from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


try:
    from sklearn.cluster import KMeans as _SklearnKMeans  # type: ignore
except Exception:  # pragma: no cover
    _SklearnKMeans = None


@dataclass
class _SimpleKMeans2D:
    """Tiny 2D kmeans fallback (only what we need: centers_ and predict)."""

    centers_: np.ndarray  # (k,2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must have shape (n,2)")
        d2 = ((X[:, None, :] - self.centers_[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d2, axis=1).astype(int)


@dataclass
class OracleDifficultyModelKMeans:
    """Oracle difficulty buckets derived from KMeans over (a,b)."""

    kmeans: object
    centers_ab: np.ndarray  # (k,2), ordered by easiness
    probs: np.ndarray  # (k,)
    cluster_to_bucket: np.ndarray  # (k,), raw cluster label -> ordered bucket index


def fit_kmeans_2d(X: np.ndarray, k: int, *, seed: int = 0) -> object:
    """Fit KMeans on 2D points with sklearn if available, else fallback."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must have shape (n,2)")
    n = int(X.shape[0])
    if n <= 0:
        raise ValueError("need at least one point")

    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, n)

    if _SklearnKMeans is not None:
        model = _SklearnKMeans(n_clusters=k, random_state=seed, n_init=10)
        model.fit(X)
        return model

    rng = np.random.default_rng(seed)
    init_idx = rng.choice(n, size=k, replace=False)
    centers = X[init_idx].copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(200):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1).astype(int)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_centers[j] = X[mask].mean(axis=0)
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    return _SimpleKMeans2D(centers_=centers)


def build_oracle_difficulty_model_from_params(
    train_params: Dict[str, Tuple[float, float]],
    *,
    score_fn: Callable[[float, float], float],
    k: int = 5,
    random_seed: int = 0,
) -> Optional[OracleDifficultyModelKMeans]:
    """Fit KMeans model on training (a,b) and return ordered centers + probs."""
    if not train_params:
        return None

    a_values = np.asarray([v[0] for v in train_params.values()], dtype=float)
    b_values = np.asarray([v[1] for v in train_params.values()], dtype=float)
    if a_values.size == 0 or b_values.size == 0:
        return None

    X = np.column_stack([a_values, b_values]).astype(float)
    finite_mask = np.isfinite(X).all(axis=1)
    X = X[finite_mask]
    if X.shape[0] == 0:
        return None

    k_eff = min(int(k), int(X.shape[0]))
    if k_eff <= 0:
        return None

    kmeans = fit_kmeans_2d(X, k_eff, seed=int(random_seed))
    centers_attr = getattr(kmeans, "cluster_centers_", None)
    if centers_attr is None:
        centers_attr = getattr(kmeans, "centers_")
    centers = np.asarray(centers_attr, dtype=float)

    labels = getattr(kmeans, "labels_", None)
    if labels is None:
        labels = kmeans.predict(X)
    labels = np.asarray(labels, dtype=int)

    counts = np.bincount(labels, minlength=k_eff).astype(float)
    total = float(counts.sum())
    if total <= 0:
        return None
    probs = counts / total

    scores = np.asarray([float(score_fn(float(a), float(b))) for a, b in centers], dtype=float)
    perm = np.argsort(-scores)
    ordered_centers = centers[perm]
    ordered_probs = probs[perm]

    cluster_to_bucket = np.empty(k_eff, dtype=int)
    for bucket_idx, cluster_label in enumerate(perm.tolist()):
        cluster_to_bucket[int(cluster_label)] = int(bucket_idx)

    return OracleDifficultyModelKMeans(
        kmeans=kmeans,
        centers_ab=np.asarray(ordered_centers, dtype=float),
        probs=np.asarray(ordered_probs, dtype=float),
        cluster_to_bucket=cluster_to_bucket,
    )


def greedy_budget_allocation_oracle_common(
    model: OracleDifficultyModelKMeans,
    *,
    average_budget: float,
    B_max: int,
    min_budget: int,
    marginal_gain_fn: Callable[[int, int, np.ndarray], float],
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """Shared greedy budget allocation over KMeans buckets.

    `marginal_gain_fn(bucket_idx, current_budget, centers_ab) -> delta`.
    """
    centers = np.asarray(model.centers_ab, dtype=float)
    probs = np.asarray(model.probs, dtype=float)
    k = int(centers.shape[0])
    if k == 0:
        return np.zeros((0,), dtype=int), 0.0

    B_max = int(B_max)
    min_budget = int(min_budget)
    if B_max < min_budget:
        B_max = min_budget
    min_budget = max(1, min(min_budget, B_max))

    B = np.full(k, int(min_budget), dtype=int)
    used_budget = float(np.sum(probs * B))

    if used_budget >= float(average_budget) - float(eps):
        return B, used_budget

    import heapq

    def marginal_gain(t: int) -> float:
        cur = int(B[t])
        if cur + 1 > int(B_max):
            return -math.inf
        return float(marginal_gain_fn(int(t), int(cur), centers))

    heap: List[Tuple[float, int]] = []
    for t in range(k):
        gain = marginal_gain(t)
        if gain > 0 and probs[t] > 0:
            heapq.heappush(heap, (-gain, t))

    while heap and used_budget + float(eps) < float(average_budget):
        neg_gain, t = heapq.heappop(heap)
        gain = -float(neg_gain)
        if gain <= 0:
            continue
        if B[t] >= int(B_max):
            continue

        cost = float(probs[t])
        if cost <= 0:
            continue
        if used_budget + cost > float(average_budget) + float(eps):
            continue

        B[t] += 1
        used_budget += cost

        next_gain = marginal_gain(t)
        if next_gain > 0 and B[t] < int(B_max):
            heapq.heappush(heap, (-next_gain, t))

    candidates = [t for t in range(k) if float(probs[t]) > 0.0 and int(B[t]) < int(B_max)]
    candidates.sort(key=lambda t: float(probs[t]))

    if candidates:
        while used_budget + float(eps) < float(average_budget):
            slack = float(average_budget) - used_budget
            progressed = False
            for t in candidates:
                if int(B[t]) >= int(B_max):
                    continue
                cost = float(probs[t])
                if cost <= 0:
                    continue
                if cost <= slack + float(eps):
                    B[t] += 1
                    used_budget += cost
                    progressed = True
            if not progressed:
                break

    return B, used_budget


def locate_param_bin_oracle(a_value: float, b_value: float, model: OracleDifficultyModelKMeans) -> int:
    """Assign (a,b) to nearest KMeans center and return ordered bucket index [0..k-1]."""
    x = np.asarray([[float(a_value), float(b_value)]], dtype=float)
    if not np.isfinite(x).all():
        return 0
    cluster_label = int(model.kmeans.predict(x)[0])
    if cluster_label < 0 or cluster_label >= int(model.cluster_to_bucket.size):
        return 0
    return int(model.cluster_to_bucket[cluster_label])
