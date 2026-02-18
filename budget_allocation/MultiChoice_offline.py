#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Usage:
  python3 gpqa_offline_output.py \
    --preds gpqa_conf_qwen3_64.jsonl \
    --multi_runs 10 \
    --consistency_plot gpqa_offline_multi_consistency.png \
    --accuracy_plot gpqa_offline_multi_accuracy.png \
    --multi_run_jsonl gpqa_offline_multi_curves.jsonl \
        --B 64 --with_baseline
"""

from __future__ import annotations
import json, argparse, math
from collections import Counter
from typing import Dict, List, Tuple, Optional, Sequence, Union
import numpy as np
from collections import defaultdict

from multi_run_export import export_multi_run_curves_jsonl
from plots.offline_curves import (
    aggregate_multi_run_curve_stats,
    plot_accuracy_multi_run_curves,
    plot_multi_run_curves,
)


# ==================== Helpers ====================

def vote_majority(answers: List[str]) -> str:
   
    if not answers:
        return ""
    cnt = Counter(answers)
    max_count = max(cnt.values())
    tied = {a for a, c in cnt.items() if c == max_count}
    for a in answers:
        if a in tied:
            return a
    return ""



def weighted_vote_majority(answers, trace_confs):
  
    if not answers or not trace_confs:
        return ""
    #mean_conf_list = [d["mean_confidence"] for d in trace_confs]
    weighted_cnt = defaultdict(float)
    for a, conf in zip(answers, trace_confs):
        weighted_cnt[a] += conf
    if not weighted_cnt:
        return ""
    max_weight = max(weighted_cnt.values())
    tied = {a for a,w in weighted_cnt.items() if w == max_weight}
    firstpos = {}
    for i,a in enumerate(answers):
        if a in tied and a not in firstpos: firstpos[a] = i
    return min(tied, key=lambda a: firstpos[a])



def weighted_top_percent_vote_majority(answers, trace_confs, frac: float):
    if not answers or not trace_confs:
        return ""
    indexed = [(idx, ans, float(conf)) for idx, (ans, conf) in enumerate(zip(answers, trace_confs))]
    if not indexed:
        return ""
    frac = max(0.0, min(1.0, float(frac)))
    top_k = max(1, int(len(indexed) * frac))
    top_k = min(top_k, len(indexed))
    top_samples = sorted(indexed, key=lambda x: x[2], reverse=True)[:top_k]
    top_indices = {idx for idx, _, _ in top_samples}
    weighted_cnt = defaultdict(float)
    for _, ans, conf in top_samples:
        weighted_cnt[ans] += conf
    if not weighted_cnt:
        return ""
    max_weight = max(weighted_cnt.values())
    tied = {a for a, w in weighted_cnt.items() if w == max_weight}
    firstpos = {}
    for idx, ans in enumerate(answers):
        if idx in top_indices and ans in tied and ans not in firstpos:
            firstpos[ans] = idx
    return min(tied, key=lambda a: firstpos[a]) if firstpos else ""


def weighted_top10percent_vote_majority(answers, trace_confs):
    return weighted_top_percent_vote_majority(answers, trace_confs, 0.1)


def weighted_top30percent_vote_majority(answers, trace_confs):
    return weighted_top_percent_vote_majority(answers, trace_confs, 0.3)


def weighted_top50percent_vote_majority(answers, trace_confs):
    return weighted_top_percent_vote_majority(answers, trace_confs, 0.5)


def weighted_top70percent_vote_majority(answers, trace_confs):
    return weighted_top_percent_vote_majority(answers, trace_confs, 0.7)


def weighted_top90percent_vote_majority(answers, trace_confs):
    return weighted_top_percent_vote_majority(answers, trace_confs, 0.9)


VARIANT_FUNC_MAP = {
    "weighted": weighted_vote_majority,
    "top10": weighted_top10percent_vote_majority,
    "top30": weighted_top30percent_vote_majority,
    "top50": weighted_top50percent_vote_majority,
    "top70": weighted_top70percent_vote_majority,
    "top90": weighted_top90percent_vote_majority,
}

# Confidence metrics available inside trace_confidence entries
# Default mean_confidence metric is now keyed by "Conf" (kept "mean" as alias for backward compatibility).
CONF_METRIC_KEYS = {
    "Conf": "mean_confidence",
    "mean": "mean_confidence",  # alias to avoid breaking older configs
    "tail": "tail_2048_mean_conf",
    "bottom": "bottom_0.1_sliding_2048_mean_conf",
}

DEFAULT_CONF_VALS = {
    "mean_confidence": 1.0,
    "tail_2048_mean_conf": 1.0,
    "bottom_0.1_sliding_2048_mean_conf": 1.0,
}


def _default_conf_entry() -> dict:
 
    return dict(DEFAULT_CONF_VALS)

# Default method names to evaluate for OKG (excluding plain OKG majority)
DEFAULT_OKG_METHODS = [
    "Conf",          # mean_confidence (renamed) weighted majority
    "tail_top90",    # tail metric + top90 filtering weighted majority
    "bottom_top90",  # bottom metric + top90 filtering weighted majority
]

# Default baseline methods (include mv by default)
DEFAULT_BASELINE_METHODS = ["mv"] + DEFAULT_OKG_METHODS

# Canonical "conf" variants expected by downstream summary consumers.
# We map Base_Conf/OKG_Conf to this method when exporting curves.
DEFAULT_REPORT_CONF_METHOD = "tail_top70"


def _ensure_required_methods(method_names: List[str], required: Sequence[str]) -> List[str]:
    """Return a new list where `required` methods are present (if supported).

    We only add a required method if it is recognized by `build_method_functions`.
    """
    out: List[str] = list(method_names or [])
    seen = set(out)
    for m in required or []:
        if not m or m in seen:
            continue
        if build_method_functions([m]):
            out.append(m)
            seen.add(m)
    return out


def _get_metric_weights(answers: List[str], conf_dicts: Optional[List[dict]], metric: str) -> List[float]:
    """Return a list of weights aligned with answers for the given metric.

    When the requested metric is missing, fall back to mean_confidence or 1.0.
    """
    n = len(answers)
    if n == 0:
        return []
    key = CONF_METRIC_KEYS.get(metric, "mean_confidence")
    if not conf_dicts:
        return [1.0] * n
    weights: List[float] = []
    for i in range(n):
        entry = conf_dicts[i] if i < len(conf_dicts) else None
        if not isinstance(entry, dict):
            try:
                weights.append(float(entry))
            except Exception:
                weights.append(1.0)
            continue
        value = entry.get(key)
        if value is None:
            value = entry.get("mean_confidence", 1.0)
        try:
            weights.append(float(value))
        except Exception:
            weights.append(1.0)
    return weights


def build_method_functions(method_names: Sequence[str]) -> List[Tuple[str, callable]]:
    """Convert method name strings into (name, callable) pairs.

    Supported naming patterns:
      - "mv": plain majority voting
      - Variant names from VARIANT_FUNC_MAP (e.g. "weighted", "top70"): use mean_confidence
      - Metric-prefixed names (e.g. "mean", "mean_top90", "tail_top10"): use specified metric
    """
    result: List[Tuple[str, callable]] = []
    seen = set()

    for raw_name in method_names:
        if raw_name in seen:
            continue
        if raw_name == "mv":
            result.append((raw_name, lambda answers, confs=None: vote_majority(answers)))
            seen.add(raw_name)
            continue

        # Backwards-compatible variant name (assume mean_confidence weights)
        if raw_name in VARIANT_FUNC_MAP:
            base_func = VARIANT_FUNC_MAP[raw_name]

            def make_wrapper(func=base_func):
                return lambda answers, confs: func(answers, _get_metric_weights(answers, confs, "mean"))

            result.append((raw_name, make_wrapper()))
            seen.add(raw_name)
            continue

        # Metric-prefixed name, e.g. "mean" or "mean_top90"
        if "_" in raw_name:
            metric, variant = raw_name.split("_", 1)
        else:
            metric, variant = raw_name, "weighted"

        if metric not in CONF_METRIC_KEYS:
            # Unknown pattern; skip gracefully
            continue

        if variant in ("weighted", "mean"):
            variant_key = "weighted"
        else:
            variant_key = variant

        base_variant = VARIANT_FUNC_MAP.get(variant_key)
        if base_variant is None:
            continue

        def make_metric_wrapper(func=base_variant, metric_name=metric):
            # Example: metric="tail", variant="top90" computes tail-based weights, keeps the top 90% highest-weight answers, then majority votes
            return lambda answers, confs: func(answers, _get_metric_weights(answers, confs, metric_name))

        result.append((raw_name, make_metric_wrapper()))
        seen.add(raw_name)

    return result


LabelsType = Union[Dict[str, str], Dict[str, Dict[str, str]]]


def _default_labels(labels: LabelsType) -> Dict[str, str]:

    if not labels:
        return {}
    try:
        if all(isinstance(v, dict) for v in labels.values()):
            default = labels.get("__default__")  # type: ignore[assignment]
            if isinstance(default, dict):
                return default
            return next(iter(labels.values()))  # type: ignore[return-value]
    except Exception:
        pass
    return labels  # type: ignore[return-value]


def _label_for_method(labels: LabelsType, *, method_name: str, qid: str) -> str:

    base = _default_labels(labels)
    if not base:
        return ""
    try:
        if labels and all(isinstance(v, dict) for v in labels.values()):
            m = labels.get(method_name)  # type: ignore[assignment]
            if isinstance(m, dict):
                v = m.get(qid)
                if v is not None:
                    return str(v)
    except Exception:
        pass
    return str(base.get(qid, ""))


def _evaluate_methods_on_qs(qids: List[str],
                             labels: LabelsType,
                             answers_map: Dict[str, List[str]],
                             confs_map: Optional[Dict[str, List[dict]]],
                             methods: List[Tuple[str, callable]]) -> Dict[str, float]:
    """Evaluate a list of (name, func) methods over qids and return accuracy per method.

    - answers_map: qid -> list of observed answers (strings)
    - confs_map: qid -> list of per-sample confidence dicts; may be None
    - methods: list of (name, func). func may accept (answers, confs) or (answers,) only.
    """
    base_labels = _default_labels(labels)
    valid_qids = [q for q in qids if q in base_labels]
    if not valid_qids:
        return {name: 0.0 for name, _ in methods}

    correct_counts = {name: 0 for name, _ in methods}
    total = 0
    for q in valid_qids:
        answers = answers_map.get(q, [])
        confs = (confs_map.get(q, []) if confs_map is not None else [])
        for name, func in methods:
            # call func with (answers, confs) when possible; fall back to (answers,)
            try:
                try:
                    pred = func(answers, confs)
                except TypeError:
                    pred = func(answers)
            except Exception:
                pred = ""
            if pred == _label_for_method(labels, method_name=name, qid=q):
                correct_counts[name] += 1
        total += 1

    return {name: (correct_counts[name] / total) for name, _ in methods}




def load_labels(path: str, answer_map: Dict[str, int]) -> Dict[str, str]:


    labels: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            qid = item.get("id")
            if not qid:
                continue
            raw = str(item.get("final")).strip().upper()
            if raw and raw in answer_map:
                labels[qid] = raw
    return labels


# Helper to record OKG curve points
def maybe_record_curve(current_budget: int,
                       curve_records: List[Tuple[int, Dict[str, float]]],
                       ordered_qids: List[str],
                       labels: LabelsType,
                       collected_answers: Dict[str, List[str]],
                       collected_confs: Optional[Dict[str, List[dict]]],
                       eval_every: int = 1,
                       force: bool = False,
                       method_names: Optional[Sequence[str]] = None) -> None:

    if not _default_labels(labels):
        return
    if not force and current_budget % max(1, eval_every) != 0:
        return

    # build methods list: OKG majority + selected weighted variants
    names = list(dict.fromkeys(method_names or DEFAULT_OKG_METHODS))
    # OKG already covers plain majority; exclude mv if present in names
    names = [n for n in names if n and n != "mv"]
    methods: List[Tuple[str, callable]] = [("OKG", lambda a, c=None: vote_majority(a))]
    methods.extend(build_method_functions(names))

    # evaluate methods over current collected answers/confs using shared helper
    acc_dict = _evaluate_methods_on_qs(ordered_qids, labels, collected_answers, collected_confs, methods)

    # store into curve_records at current_budget (replace if same t)
    if curve_records and curve_records[-1][0] == current_budget:
        curve_records[-1] = (current_budget, acc_dict)
    else:
        curve_records.append((current_budget, acc_dict))


# ==================== Baseline ====================

def compute_baseline_curve(qids: List, pools: Dict[str, List[str]], 
                           labels: LabelsType, budgets: List[int], warm_up: int,
                           warmup_answers: Dict[str, List[str]],
                           warmup_indices: Dict[str, List[int]],
                           confs_pools: Optional[Dict[str, List[dict]]] = None,
                           methods: Optional[List[Tuple[str, callable]]] = None) -> Dict[str, Dict[int, float]]:

    K = len(qids)
    base_labels = _default_labels(labels)
    if K == 0 or not base_labels:
        return {}

    valid_qids = [q for q in qids if q in base_labels]
    if not valid_qids:
        return {}

    # default methods
    if methods is None:
        methods = [("mv", lambda a, c=None: vote_majority(a))]

    # prepare result structure
    curves: Dict[str, Dict[int, float]] = {name: {} for name, _ in methods}

    for budget in budgets:
        b = budget // K
        if b <= 0:
            continue

        selected_map: Dict[str, List[str]] = {}
        selected_confs_map: Dict[str, List[dict]] = {} if confs_pools is not None else {}

        for q in valid_qids:
            pool_answers = pools.get(q, [])
            if not pool_answers:
                continue

            warmup_obs = warmup_answers.get(q, [])
            warmup_take = min(warm_up, len(warmup_obs), b)
            answers: List[str] = list(warmup_obs[:warmup_take]) if warmup_take > 0 else []
            confs: List[dict] = []

            if confs_pools is not None and warmup_take > 0:
                used_indices = warmup_indices.get(q, [])[:warmup_take]
                cpool = confs_pools.get(q, [])
                for idx in used_indices:
                    if idx < len(cpool) and isinstance(cpool[idx], dict):
                        confs.append(dict(cpool[idx]))
                    else:
                        confs.append(_default_conf_entry())

            remaining_need = max(0, b - warmup_take)
            if remaining_need > 0:
                used_indices = warmup_indices.get(q, [])[:warmup_take]
                used_set = set(used_indices)
                added = 0
                for idx, ans in enumerate(pool_answers):
                    if idx in used_set:
                        continue
                    answers.append(ans)
                    if confs_pools is not None:
                        cpool = confs_pools.get(q, [])
                        if idx < len(cpool) and isinstance(cpool[idx], dict):
                            confs.append(dict(cpool[idx]))
                        else:
                            confs.append(_default_conf_entry())
                    added += 1
                    if added >= remaining_need:
                        break

            if not answers:
                continue

            selected_map[q] = answers
            if confs_pools is not None:
                selected_confs_map[q] = confs

        selected_confs_arg = selected_confs_map if confs_pools is not None else None
        accs = _evaluate_methods_on_qs(valid_qids, labels, selected_map, selected_confs_arg, methods)
        for name in accs:
            curves[name][budget] = accs[name]


    return curves


#===============================================================
# ----------------- OKG Allocator -----------------
#===============================================================

class OKGAllocator:
  
    def __init__(self, M: int, nsamples: int = 500, seed: Optional[int] = None):
        self.M = int(M)
        self.nsamples = int(nsamples)
        self.rng = np.random.default_rng(seed)

    def _estimate_I_vector(self, alpha: Sequence[float]) -> np.ndarray:
        alpha = np.asarray(alpha, dtype=float)
        if alpha.ndim != 1 or alpha.size != self.M:
            raise ValueError("alpha must be 1-D with length M")
        # Gamma sampling + argmax counting
        X = self.rng.gamma(shape=alpha, scale=1.0, size=(self.nsamples, self.M))
        argmax = np.argmax(X, axis=1)
        counts = np.bincount(argmax, minlength=self.M)
        probs = counts / self.nsamples
        return probs

    @staticmethod
    def _compute_h(I_vector: np.ndarray) -> float:
      
        return float(np.max(I_vector))

    def select_next(self, alpha_list: Sequence[Sequence[float]], c: float = 1.0) -> int:
        A = np.asarray(alpha_list, dtype=float)
        if A.ndim != 2 or A.shape[1] != self.M:
            raise ValueError("alpha_list must be shape (K, M)")
        K = A.shape[0]
        R_plus_vals = np.empty(K, dtype=float)
        for i in range(K):
            alpha_i = A[i]
            r_vals = np.empty(self.M, dtype=float)
            I_base = self._estimate_I_vector(alpha_i)
            h_base = self._compute_h(I_base)
            for m in range(self.M):
                alpha_plus = alpha_i.copy()
                alpha_plus[m] += c
                I_plus = self._estimate_I_vector(alpha_plus)
                h_plus = self._compute_h(I_plus)
                r_vals[m] = h_plus - h_base
            R_plus_vals[i] = np.max(r_vals)
        return int(np.argmax(R_plus_vals))

    @staticmethod
    def update(
        alpha_list: np.ndarray,
        i: int,
        y: int,
        c: Union[float, int] = 1.0,
    ) -> None:
        """Update alpha for one (question, label) observation.

        `c` can be a confidence-scaled pseudo-count.
        """
        try:
            cc = float(c)
        except Exception:
            cc = 1.0
        alpha_list[i, y] += cc


class OKGAllocator_Batch(OKGAllocator):
    """Batch variant that selects the top-s questions each round."""

    def __init__(self, M: int, batch_size: int, nsamples: int = 500, seed: Optional[int] = None):
        super().__init__(M=M, nsamples=nsamples, seed=seed)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)

    def select_next(self, alpha_list: Sequence[Sequence[float]], c: float = 1.0) -> List[int]:
        A = np.asarray(alpha_list, dtype=float)
        if A.ndim != 2 or A.shape[1] != self.M:
            raise ValueError("alpha_list must be shape (K, M)")
        K = A.shape[0]
        R_plus_vals = np.empty(K, dtype=float)
        for i in range(K):
            alpha_i = A[i]
            gains = np.empty(self.M, dtype=float)
            I_base = self._estimate_I_vector(alpha_i)
            h_base = self._compute_h(I_base)
            for m in range(self.M):
                alpha_plus = alpha_i.copy()
                alpha_plus[m] += c
                I_plus = self._estimate_I_vector(alpha_plus)
                gains[m] = self._compute_h(I_plus) - h_base
            R_plus_vals[i] = np.max(gains)

        top_k = min(self.batch_size, K)
        indices = np.argpartition(-R_plus_vals, top_k - 1)[:top_k]
        return indices.tolist()

    @staticmethod
    def update(
        alpha_list: np.ndarray,
        indices: Sequence[int],
        options: Sequence[int],
        c: Union[float, int, Sequence[Union[float, int]]] = 1.0,
    ) -> None:
        if len(indices) != len(options):
            raise ValueError("indices and options must have the same length")

        if isinstance(c, (list, tuple, np.ndarray)):
            if len(c) != len(indices):
                raise ValueError("c must have the same length as indices when provided as a sequence")
            cs = c
        else:
            cs = [c] * len(indices)

        for idx, opt, ci in zip(indices, options, cs):
            try:
                cc = float(ci)
            except Exception:
                cc = 1.0
            alpha_list[idx, opt] += cc


#===============================================================
# ----------------- main run -----------------
#===============================================================
def run(preds_path: str,
    out_path: str = "preds_okg.jsonl",
    B: int = 64,
    nsamples: int = 500,
    seed: Optional[int] = 2025,
    choices: str = "A,B,C,D",
    eval_every: int = 1,
    batch_size: int = 1,
    with_baseline: bool = False,
    warm_up: int = 0,
    labels_path: Optional[str] = None,
    baseline_methods: Optional[Sequence[str]] = None,
    smoke_n: Optional[int] = None,
    subsample_pool_size: Optional[int] = None,
    relabel_from_subsample_mv: bool = False,
    return_curve_data: bool = False,
    save_outputs: bool = True) -> Union[Tuple[list, np.ndarray], Tuple[list, np.ndarray, Optional[dict]]]:

    # rng: for OKG / warmup / internal randomness
    rng = np.random.default_rng(seed)
    # rng_pool: dedicated RNG for per-run answer-pool shuffling/subsampling so toggling
    # subsampling doesn't inadvertently change OKG randomness.
    rng_pool = np.random.default_rng(int(seed or 0) + 1000003)

    # 1) Parse the choices list and build mappings
    label_list = [s.strip() for s in choices.split(",") if s.strip()]
    if not label_list:
        raise ValueError("--choices parsed to an empty list")
    answer_map = {lab: i for i, lab in enumerate(label_list)}
    rev_map = {i: lab for lab, i in answer_map.items()}
    M = len(label_list)

    def _extract_pseudo_label_from_final(item: dict) -> Optional[str]:
        """Pseudo label for self-consistency evaluation (from item['final'])."""
        if not isinstance(item, dict):
            return None
        raw_final = item.get("final")
        if raw_final is None:
            return None
        s = str(raw_final).strip().upper()
        return s if s in answer_map else None

    def _extract_true_label_from_answer(item: dict) -> Optional[str]:
        """True gold label for accuracy evaluation (from item['answer'] or item['answer_index'])."""
        if not isinstance(item, dict):
            return None

        raw = item.get("answer")
        if raw is None:
            raw = item.get("answer_index")

        # Case 1: already a letter label
        if isinstance(raw, str):
            s = raw.strip().upper()
            return s if s in answer_map else None

        # Case 2: index
        try:
            idx = int(raw)
        except Exception:
            idx = None
        if idx is not None and 0 <= idx < M:
            return rev_map.get(idx)
        return None

    #
    ordered_qids: List = []     # qid list in the order they appear in preds (for deterministic processing and optional smoke-testing)
    questions_map: Dict = {}    # qid → question text
    answer_pools: Dict = {}     # qid → list of valid answer letters (mapped from raw answers in preds)
    extracted_labels: Dict = {}  # backward-compatible: will hold the *primary* label set
    pseudo_labels: Dict = {}     # qid -> pseudo label from final (consistency)
    true_labels: Dict = {}       # qid -> dataset gold label (accuracy)
    label_source = "preds file (final)"
    skipped_unmapped = 0

    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            # qid
            qid = item.get("id")
            if not qid:
                qid = f"gpqa_{len(ordered_qids):05d}"
            ordered_qids.append(qid)

            # question text
            questions_map[qid] = item.get("question", "")

            # Extract labels for both metrics
            pseudo = _extract_pseudo_label_from_final(item)
            if pseudo:
                pseudo_labels[qid] = pseudo
            gold = _extract_true_label_from_answer(item)
            if gold:
                true_labels[qid] = gold

            # answers pool
            pool = []
            for ans in item.get("answers", []):
                ans_letter = str(ans).strip().upper()
                if ans_letter in answer_map:
                    pool.append(answer_map[ans_letter])
                else:
                    skipped_unmapped += 1
            answer_pools[qid] = pool

    
    K = len(ordered_qids)
    if K == 0:
        raise RuntimeError("No questions found in preds file")

    B = max(1, int(B))
    T = K * B

    # Smoke-test mode: optionally restrict to first N questions to run faster
    if smoke_n is not None and int(smoke_n) > 0 and int(smoke_n) < K:
        smoke_n = int(smoke_n)
        print(f"[smoke] restricting to first {smoke_n} questions for a quick run")
        keep = set(ordered_qids[:smoke_n])
        ordered_qids = [q for q in ordered_qids if q in keep]
        questions_map = {q: questions_map[q] for q in ordered_qids}
        # filter answer pools and labels
        answer_pools = {q: answer_pools[q] for q in ordered_qids}
        pseudo_labels = {q: pseudo_labels[q] for q in ordered_qids if q in pseudo_labels}
        true_labels = {q: true_labels[q] for q in ordered_qids if q in true_labels}
        K = len(ordered_qids)
        T = K * B

    print(f"[load] K={K}, M={M}, skipped_unmapped_answers={skipped_unmapped}")



    
    answer_pools_str: Dict[str, List[str]] = {}
    # also build per-answer confidence pools if trace_confidence exists in preds
    confs_pools: Dict[str, List[dict]] = {}
    with open(preds_path, "r", encoding="utf-8") as f:
        rec_idx = 0
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            # IMPORTANT: keep qid consistent with the first pass.
            # Many datasets have id=None; first pass auto-generates gpqa_00000... based on record index.
            if rec_idx < len(ordered_qids):
                qid = ordered_qids[rec_idx]
            else:
                qid = item.get("id") or f"gpqa_{rec_idx:05d}"
            rec_idx += 1
            # 
            raw_answers = item.get("answers", [])
            filtered_answers: List[str] = []
            trace_list = item.get("trace_confidence", [])
            for idx, ans in enumerate(raw_answers):
                letter = str(ans).strip().upper()
                if letter in answer_map:
                    filtered_answers.append(letter)
                    # extract multiple confidence metrics if available
                    entry = trace_list[idx] if idx < len(trace_list) else {}
                    entry = entry or {}
                    # Support nested schema: {"conf_summary": {...}} (e.g. server_stats)
                    conf_src = entry.get("conf_summary") if isinstance(entry, dict) else None
                    if isinstance(conf_src, dict):
                        entry = {**entry, **conf_src}
                    conf_entry = _default_conf_entry()
                    try:
                        conf_entry["mean_confidence"] = float(entry.get("mean_confidence", 1.0))
                    except Exception:
                        conf_entry["mean_confidence"] = 1.0
                    try:
                        conf_entry["tail_2048_mean_conf"] = float(entry.get("tail_2048_mean_conf", entry.get("tail_mean_conf", 1.0)))
                    except Exception:
                        conf_entry["tail_2048_mean_conf"] = 1.0
                    try:
                        conf_entry["bottom_0.1_sliding_2048_mean_conf"] = float(entry.get("bottom_0.1_sliding_2048_mean_conf", entry.get("bottom_mean_conf", 1.0)))
                    except Exception:
                        conf_entry["bottom_0.1_sliding_2048_mean_conf"] = 1.0
                    confs_pools.setdefault(qid, []).append(conf_entry)
            answer_pools_str[qid] = filtered_answers

    # 2.7) multi-run helper: per-run shuffle the full pool, take the first K as the pool,
    # and (optionally) treat MV(pool[:K]) as this run's gold label (equivalent to original "final").
    if subsample_pool_size is not None and int(subsample_pool_size) > 0:
        pool_k = int(subsample_pool_size)
        changed = 0
        too_short = 0
        mismatched_conf = 0
        for qid in ordered_qids:
            pool_letters = list(answer_pools_str.get(qid, []) or [])
            if not pool_letters:
                continue
            n = len(pool_letters)
            k_eff = min(pool_k, n)
            if n < pool_k:
                too_short += 1

            idx = np.arange(n, dtype=int)
            rng_pool.shuffle(idx)
            idx = idx[:k_eff]
            # "shuffle then take prefix" semantics
            pool_letters_sub = [pool_letters[int(i)] for i in idx.tolist()]
            answer_pools_str[qid] = pool_letters_sub
            # rebuild int pool from letters to guarantee alignment
            answer_pools[qid] = [answer_map[a] for a in pool_letters_sub if a in answer_map]

            # subsample confidence pool if aligned; otherwise drop to avoid misalignment
            if qid in confs_pools:
                cpool = confs_pools.get(qid, []) or []
                if len(cpool) == n:
                    confs_pools[qid] = [cpool[int(i)] for i in idx.tolist()]
                else:
                    confs_pools[qid] = []
                    mismatched_conf += 1

            # Optional: MV relabeling ONLY affects pseudo-labels (consistency),
            # never overrides dataset gold labels (accuracy).
            if relabel_from_subsample_mv:
                pseudo_labels[qid] = vote_majority(pool_letters_sub)
            changed += 1

        print(
            f"[multi-run] subsample_pool_size={pool_k}, changed={changed}, "
            f"too_short<{pool_k}={too_short}, mismatched_conf={mismatched_conf}, "
            f"relabel_mv={bool(relabel_from_subsample_mv)}"
        )

    # 3) 
    alphas = np.ones((K, M), dtype=float)
    alloc = OKGAllocator_Batch(M=M, batch_size=max(1, batch_size), nsamples=nsamples, seed=seed)
    used_indices = {qid: [] for qid in ordered_qids}     # 
    collected_answers = {qid: [] for qid in ordered_qids}  # 
    collected_confs = {qid: [] for qid in ordered_qids}   # 
    exhausted_questions = set()  # 

    # 4) labels for online evaluation curves
    labels_consistency: Dict[str, str] = {}
    if labels_path:
        try:
            labels_consistency = load_labels(labels_path, answer_map)
            label_source = f"labels file: {labels_path}"
        except Exception as e:
            print(f"[warning] failed to load labels_path={labels_path}: {e}")
            labels_consistency = dict(pseudo_labels)
            label_source = "preds file (final)"
    else:
        labels_consistency = dict(pseudo_labels)

    labels_accuracy: Dict[str, str] = dict(true_labels)

    if labels_consistency:
        print(f"[labels] consistency labels: {len(labels_consistency)} from {label_source}")
    else:
        print("[warning] No pseudo labels available; consistency curve will be empty.")
    if labels_accuracy:
        print(f"[labels] accuracy labels: {len(labels_accuracy)} from preds file (answer/answer_index)")
    else:
        print("[warning] No gold labels available; accuracy curve will be empty.")

    warm_up = max(0, int(warm_up))
    eval_every = max(1, eval_every)

    # Determine which additional methods (beyond OKG majority) to evaluate
    if baseline_methods:
        requested_methods = [m for m in baseline_methods if m]
    else:
        requested_methods = list(DEFAULT_OKG_METHODS)

    # Always ensure we can export Base_Conf / OKG_Conf for downstream scripts.
    # If the user didn't request any conf method, we inject a supported default.
    requested_methods = _ensure_required_methods(requested_methods, [DEFAULT_REPORT_CONF_METHOD])
    # deduplicate while preserving order
    seen_methods = set()
    requested_methods = [m for m in requested_methods if not (m in seen_methods or seen_methods.add(m))]
    baseline_method_names = ["mv"] + requested_methods
    seen_baseline = set()
    baseline_method_names = [m for m in baseline_method_names if not (m in seen_baseline or seen_baseline.add(m))]
    okg_method_names = [m for m in requested_methods if m != "mv"]

    # Consistency labels: for confidence-weighted methods, use a run-specific pseudo label
    # computed from the full (current-run) answer pool + confidence weights.
    labels_consistency_by_method: Dict[str, Dict[str, str]] = {
        "__default__": dict(labels_consistency),
        "OKG": dict(labels_consistency),
        "mv": dict(labels_consistency),
    }
    weighted_method_names: List[str] = [
        m
        for m in list(dict.fromkeys((baseline_method_names + okg_method_names)))
        if m and m not in ("mv", "OKG")
    ]
    method_funcs = {name: fn for name, fn in build_method_functions(weighted_method_names)}
    if weighted_method_names and method_funcs and labels_consistency:
        for m in weighted_method_names:
            fn = method_funcs.get(m)
            if fn is None:
                continue
            per_method: Dict[str, str] = {}
            for qid in ordered_qids:
                if qid not in labels_consistency:
                    continue
                pool_answers = answer_pools_str.get(qid, []) or []
                pool_confs = confs_pools.get(qid, []) if confs_pools is not None else []
                try:
                    pseudo_m = fn(pool_answers, pool_confs)
                except Exception:
                    pseudo_m = ""
                per_method[qid] = pseudo_m if pseudo_m else labels_consistency.get(qid, "")
            labels_consistency_by_method[m] = per_method

    selections: List[int] = []
    curve_records_consistency: List[Tuple[int, Dict[str, float]]] = []  # (t, {method: consistency})
    curve_records_accuracy: List[Tuple[int, Dict[str, float]]] = []     # (t, {method: accuracy})

    total_budget = 0
    warmup_budget_end = 0

    #===============================================================
    # 5) Warm-up stage: uniform allocation per question (optional)
    #===============================================================
    if warm_up > 0 and total_budget < T:
        print(f"[warmup] allocating up to {warm_up} answers per question before OKG")
        for idx, qid in enumerate(ordered_qids):
            if total_budget >= T:
                break
            pool = answer_pools.get(qid, [])
            if not pool:
                exhausted_questions.add(qid)
                continue
            available_answers = [j for j in range(len(pool)) if j not in used_indices[qid]]
            if not available_answers:
                exhausted_questions.add(qid)
                continue
            num_samples = min(warm_up, len(available_answers), T - total_budget)
            if num_samples <= 0:
                break
            sample_indices = rng.choice(available_answers, size=num_samples, replace=False)
            sample_indices = np.atleast_1d(sample_indices).tolist()
            for ans_idx in sample_indices:
                ans_idx = int(ans_idx)
                used_indices[qid].append(ans_idx)
                y_obs = int(pool[ans_idx])
                sampled = rev_map[y_obs]
                collected_answers[qid].append(sampled)
                # collect confidence metrics for reuse in weighted variants
                if qid in confs_pools and ans_idx < len(confs_pools[qid]) and isinstance(confs_pools[qid][ans_idx], dict):
                    conf_entry = dict(confs_pools[qid][ans_idx])
                else:
                    conf_entry = _default_conf_entry()
                collected_confs[qid].append(conf_entry)
                # Confidence-scaled alpha update: use mean_confidence / 16 to keep updates ~1.
                try:
                    conf_val = float(conf_entry.get("mean_confidence", 16.0))
                except Exception:
                    conf_val = 16.0
                alphas[idx, y_obs] += (conf_val / 16.0)
                selections.append(int(idx))
                total_budget += 1
                maybe_record_curve(total_budget, curve_records_consistency, ordered_qids, labels_consistency_by_method, collected_answers, collected_confs, eval_every, method_names=okg_method_names)
                maybe_record_curve(total_budget, curve_records_accuracy, ordered_qids, labels_accuracy, collected_answers, collected_confs, eval_every, method_names=okg_method_names)
                if total_budget >= T:
                    break
            if len(used_indices[qid]) >= len(pool):
                exhausted_questions.add(qid)
            if total_budget >= T:
                break
        warmup_budget_end = total_budget  # we want to start curve from here
        maybe_record_curve(total_budget, curve_records_consistency, ordered_qids, labels_consistency_by_method, collected_answers, collected_confs, eval_every, force=True, method_names=okg_method_names)
        maybe_record_curve(total_budget, curve_records_accuracy, ordered_qids, labels_accuracy, collected_answers, collected_confs, eval_every, force=True, method_names=okg_method_names)
        print(f"[warmup] consumed {total_budget} budget during warm-up")
    else:
        warmup_budget_end = total_budget
        if warm_up <= 0 and total_budget == 0:
            maybe_record_curve(0, curve_records_consistency, ordered_qids, labels_consistency_by_method, collected_answers, collected_confs, eval_every, force=True, method_names=okg_method_names)
            maybe_record_curve(0, curve_records_accuracy, ordered_qids, labels_accuracy, collected_answers, collected_confs, eval_every, force=True, method_names=okg_method_names)

    # Snapshot warm-up observations so the baseline can reuse the exact samples
    warmup_answers_snapshot: Dict[str, List[str]] = {qid: list(collected_answers[qid]) for qid in ordered_qids}
    warmup_indices_snapshot: Dict[str, List[int]] = {qid: list(used_indices[qid]) for qid in ordered_qids}

    # 6) Main loop

    while total_budget < T:
        if len(exhausted_questions) >= K:
            print(f"[t={total_budget}] All questions exhausted, stopping early", flush=True)
            break

        available_indices: List[int] = []
        for i in range(K):
            qid = ordered_qids[i]
            if qid in exhausted_questions:
                continue
            pool = answer_pools.get(qid, [])
            if not pool or len(used_indices[qid]) >= len(pool):
                exhausted_questions.add(qid)
                continue
            available_indices.append(i)

        if not available_indices:
            print(f"[t={total_budget}] No available questions, stopping early", flush=True)
            break

        temp_alphas = alphas[available_indices]
        selected_local = alloc.select_next(temp_alphas, c=1.0) 
        if not selected_local:
            print(f"[t={total_budget}] Allocator returned empty selection, stopping", flush=True)
            break

        remaining_budget = min(len(selected_local), alloc.batch_size, T - total_budget)
        selected_local = selected_local[:remaining_budget]

        update_indices: List[int] = []
        update_options: List[int] = []
        update_cs: List[float] = []

        for local_idx in selected_local:
            idx = available_indices[local_idx]
            qid = ordered_qids[idx]
            pool = answer_pools.get(qid, [])
            available_answers = [j for j in range(len(pool)) if j not in used_indices[qid]]
            if not available_answers:
                exhausted_questions.add(qid)
                continue

            ans_idx = int(rng.choice(available_answers))
            used_indices[qid].append(ans_idx)
            y_obs = int(pool[ans_idx])
            sampled = rev_map[y_obs]
            collected_answers[qid].append(sampled)
            # record confidence metrics for this observation (used by weighted variants)
            if qid in confs_pools and ans_idx < len(confs_pools[qid]) and isinstance(confs_pools[qid][ans_idx], dict):
                conf_entry = dict(confs_pools[qid][ans_idx])
            else:
                conf_entry = _default_conf_entry()
            collected_confs[qid].append(conf_entry)
            # Confidence-scaled alpha update: use mean_confidence / 16 to keep updates ~1.
            try:
                conf_val = float(conf_entry.get("mean_confidence", 16.0))
            except Exception:
                conf_val = 16.0
            c_update = float(conf_val / 16.0)
            selections.append(int(idx))
            update_indices.append(idx)
            update_options.append(y_obs)
            update_cs.append(c_update)

            total_budget += 1
            print(f"[t={total_budget}] selected idx={idx} qid={qid} sampled_ans={sampled} (ans_idx={ans_idx})", flush=True)

            maybe_record_curve(total_budget, curve_records_consistency, ordered_qids, labels_consistency_by_method, collected_answers, collected_confs, eval_every, method_names=okg_method_names)
            maybe_record_curve(total_budget, curve_records_accuracy, ordered_qids, labels_accuracy, collected_answers, collected_confs, eval_every, method_names=okg_method_names)

            if total_budget >= T:
                break

            if len(used_indices[qid]) >= len(pool):
                exhausted_questions.add(qid)

        if update_indices:
            OKGAllocator_Batch.update(alphas, update_indices, update_options, c=update_cs)

    # 7) Build curve payloads for both metrics
    start_budget = warmup_budget_end  # 
    curve_consistency: List[Tuple[int, Dict[str, float]]] = [entry for entry in curve_records_consistency if entry[0] >= start_budget]
    curve_accuracy: List[Tuple[int, Dict[str, float]]] = [entry for entry in curve_records_accuracy if entry[0] >= start_budget]

    def _build_curves_dict_for_metric(
        curve: List[Tuple[int, Dict[str, float]]],
        metric_labels: LabelsType,
        *,
        with_baseline_flag: bool,
    ) -> Tuple[Dict[str, List[Tuple[int, float]]], Dict[int, float], Dict[str, Dict[int, float]], List[str]]:
        mv_curve_local: Dict[int, float] = {}
        weighted_baselines_local: Dict[str, Dict[int, float]] = {}
        selected_okg_variants_local: List[str] = [name for name in okg_method_names]

        if with_baseline_flag and metric_labels and curve:
            eval_budgets = [t for t, _ in curve]  # 
            baseline_method_defs = build_method_functions(baseline_method_names)
            if not baseline_method_defs:
                baseline_method_defs = [("mv", lambda answers, confs=None: vote_majority(answers))]
            baseline_results = compute_baseline_curve(
                qids=ordered_qids,
                pools=answer_pools_str,
                labels=metric_labels,
                budgets=eval_budgets,
                warm_up=warm_up,
                warmup_answers=warmup_answers_snapshot,
                warmup_indices=warmup_indices_snapshot,
                confs_pools=confs_pools,
                methods=baseline_method_defs,
            )
            mv_curve_local = baseline_results.get("mv", {})
            weighted_baselines_local = {name: accs for name, accs in baseline_results.items() if name != "mv"}

        # Determine final OKG variant order based on recorded curves
        curve_variant_order: List[str] = []
        for _, acc_dict in curve:
            for key in acc_dict.keys():
                if key == "OKG" or not key:
                    continue
                if key not in curve_variant_order:
                    curve_variant_order.append(key)
        if selected_okg_variants_local:
            selected_okg_variants_local = [name for name in selected_okg_variants_local if name in curve_variant_order]
            for name in curve_variant_order:
                if name not in selected_okg_variants_local:
                    selected_okg_variants_local.append(name)
        else:
            selected_okg_variants_local = curve_variant_order

        curves_dict_local: Dict[str, List[Tuple[int, float]]] = {}
        if curve:
            curves_dict_local["OKG"] = [(t, acc_dict.get("OKG", 0.0)) for t, acc_dict in curve]
            if with_baseline_flag and mv_curve_local:
                curves_dict_local["Base"] = [(t, mv_curve_local[t]) for t in sorted(mv_curve_local)]
            for name, table in (weighted_baselines_local or {}).items():
                if name == "mv":
                    continue
                curves_dict_local[f"Base_{name}"] = [(t, table.get(t, 0.0)) for t, _ in curve]
            for name in selected_okg_variants_local:
                curves_dict_local[f"OKG_{name}"] = [(t, acc_dict.get(name, 0.0)) for t, acc_dict in curve]

            # Alias the default conf-like method to canonical keys expected by summary scripts.
            base_conf_key = f"Base_{DEFAULT_REPORT_CONF_METHOD}"
            okg_conf_key = f"OKG_{DEFAULT_REPORT_CONF_METHOD}"
            if "Base_Conf" not in curves_dict_local and base_conf_key in curves_dict_local:
                curves_dict_local["Base_Conf"] = curves_dict_local[base_conf_key]
            if "OKG_Conf" not in curves_dict_local and okg_conf_key in curves_dict_local:
                curves_dict_local["OKG_Conf"] = curves_dict_local[okg_conf_key]

        return curves_dict_local, mv_curve_local, weighted_baselines_local, selected_okg_variants_local

    curves_dict_consistency, mv_curve_consistency, weighted_baselines_consistency, selected_okg_variants_consistency = _build_curves_dict_for_metric(
        curve_consistency, labels_consistency_by_method, with_baseline_flag=with_baseline
    )
    curves_dict_accuracy, mv_curve_accuracy, weighted_baselines_accuracy, selected_okg_variants_accuracy = _build_curves_dict_for_metric(
        curve_accuracy, labels_accuracy, with_baseline_flag=with_baseline
    )

    if with_baseline:
        if curves_dict_consistency:
            print("[Baseline] Computed baseline variants (consistency)")
        if curves_dict_accuracy:
            print("[Baseline] Computed baseline variants (accuracy)")

    # 8)
    if save_outputs:
        with open(out_path, "w", encoding="utf-8") as f:
            for qid in ordered_qids:
                answers_list = collected_answers.get(qid, [])
                final = vote_majority(answers_list)
                obj = {
                    "id": qid,
                    "question": questions_map.get(qid, ""),
                    "answers": answers_list,
                    "final": final
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print(f"[OKG] Saved OKG outputs to: {out_path}")






    curve_payload: Optional[dict] = None
    if return_curve_data:
        curve_payload = {
            "curve_consistency": curve_consistency,
            "curve_accuracy": curve_accuracy,
            "curves_dict_consistency": curves_dict_consistency,
            "curves_dict_accuracy": curves_dict_accuracy,
            "mv_curve_consistency": mv_curve_consistency,
            "mv_curve_accuracy": mv_curve_accuracy,
            "weighted_baselines_consistency": weighted_baselines_consistency,
            "weighted_baselines_accuracy": weighted_baselines_accuracy,
            "selected_okg_variants_consistency": selected_okg_variants_consistency,
            "selected_okg_variants_accuracy": selected_okg_variants_accuracy,
            "start_budget": start_budget,
        }

    if return_curve_data:
        return selections, alphas, curve_payload
    return selections, alphas


# ----------------- CLI -----------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="gpqa_QwenLong-L1.5-30B-A3B_128.jsonl", help="Input preds.jsonl")
    ap.add_argument("--B", type=int, default=64, help="Per-question budget (total budget = K*B)")
    ap.add_argument("--nsamples", type=int, default=500, help="I(alpha) MC sample count")
    ap.add_argument("--seed", type=int, default=1024, help="Random seed")
    ap.add_argument("--choices", default="A,B,C,D", help="Choice set, comma-separated (e.g., A,B,C,D)")

    ap.add_argument("--eval_every", type=int, default=200, help="Evaluate curve every N steps")
    ap.add_argument("--batch_size", type=int, default=1, help="Number of questions selected per round")
    ap.add_argument("--warm_up", type=int, default=1, help="Warm-up: uniformly sample answers per question (0 to skip)")
    ap.add_argument("--smoke_n", type=int, default=0, help="If >0, use only the first N questions for a quick smoke test")

    ap.add_argument("--with_baseline", action="store_true", help="Also compute naive majority voting baseline curve")
    ap.add_argument(
        "--baseline_methods",
        default="mv,tail_top70",
        help="Comma-separated baseline methods list. Supported: mv,weighted,top10,top30,top50,top70,top90,Conf,Conf_top90,tail,tail_top70,tail_top90,bottom,bottom_top90",
    )

    ap.add_argument("--multi_runs", type=int, default=10, help="Number of repeated runs (>=1)")
    ap.add_argument("--accuracy_plot", default="gpqa_offline_multi_accuracy.png", help="Output path for aggregated accuracy curve plot across runs")
    ap.add_argument("--consistency_plot", default="gpqa_offline_multi_consistency.png", help="Output path for aggregated consistency curve plot across runs")
    ap.add_argument("--accuracy_csv", default=None, help="Summary CSV path for the accuracy curve (default: same name as plot with .csv)")
    ap.add_argument("--consistency_csv", default=None, help="Summary CSV path for the consistency curve (default: same name as plot with .csv)")
    ap.add_argument("--multi_run_jsonl", default="gpqa_curve.jsonl", help="Export multi-run curves and stats to JSONL (optional)")
    ap.add_argument(
        "--multi_pool_size",
        type=int,
        default=64,
        help="Multi-run: shuffle answers per question, then take the first K as this run's answers pool (default 64)",
    )
    ap.add_argument(
        "--multi_relabel_mv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Multi-run: majority-vote on the subsampled pool and use it as this run's gold label (equivalent to the legacy final)",
    )

    args = ap.parse_args()

    # parse baseline methods list
    bm = None
    if args.baseline_methods:
        bm = [s.strip() for s in str(args.baseline_methods).split(",") if s.strip()]

    multi_runs = max(1, int(args.multi_runs))
    multi_curves_consistency: List[Tuple[int, Dict[str, List[Tuple[int, float]]]]] = []
    multi_curves_accuracy: List[Tuple[int, Dict[str, List[Tuple[int, float]]]]] = []
    for run_idx in range(multi_runs):
        run_seed = int(args.seed) + run_idx
        res = run(
            preds_path=str(args.preds),
            B=int(args.B),
            nsamples=int(args.nsamples),
            seed=run_seed,
            choices=str(args.choices),
            eval_every=int(args.eval_every),
            batch_size=int(args.batch_size),
            with_baseline=bool(args.with_baseline),
            warm_up=int(args.warm_up),
            labels_path=None,
            baseline_methods=bm,
            smoke_n=int(args.smoke_n) if int(args.smoke_n) > 0 else None,
            subsample_pool_size=int(args.multi_pool_size),
            relabel_from_subsample_mv=bool(args.multi_relabel_mv),
            return_curve_data=True,
            save_outputs=False,
        )

        try:
            _sel, _alpha, payload = res  # type: ignore[misc]
        except Exception:
            payload = None
        if isinstance(payload, dict):
            if payload.get("curves_dict_consistency"):
                multi_curves_consistency.append((run_idx, payload["curves_dict_consistency"]))  # type: ignore[arg-type]
            if payload.get("curves_dict_accuracy"):
                multi_curves_accuracy.append((run_idx, payload["curves_dict_accuracy"]))  # type: ignore[arg-type]

    if not multi_curves_consistency and not multi_curves_accuracy:
        print("[multi-run] No valid curve data collected; skip aggregation")
        raise SystemExit(2)

    # Plot both metrics (if available)
    if multi_curves_consistency and args.consistency_plot:
        title_line = f"Self Consistency Rate vs Budget (multi-run, n={len(multi_curves_consistency)})"
        plot_multi_run_curves(
            multi_curves_consistency,
            str(args.consistency_plot),
            title=title_line,
            csv_path=(str(args.consistency_csv) if args.consistency_csv else None),
        )
    else:
        print("[multi-run] Consistency curves not provided or --consistency_plot is empty; skip consistency plotting")

    if multi_curves_accuracy and args.accuracy_plot:
        title_line = f"Accuracy vs Budget (multi-run, n={len(multi_curves_accuracy)})"
        plot_accuracy_multi_run_curves(
            multi_curves_accuracy,
            str(args.accuracy_plot),
            title=title_line,
            csv_path=(str(args.accuracy_csv) if args.accuracy_csv else None),
        )
    else:
        print("[multi-run] Accuracy curves not provided or --accuracy_plot is empty; skip accuracy plotting")

    # Optional JSONL export with derived stats
    if args.multi_run_jsonl:
        export_multi_run_curves_jsonl(
            multi_curves_consistency,
            multi_curves_accuracy,
            str(args.multi_run_jsonl),
        )

    # Print quick summaries for main curves
    if multi_curves_consistency:
        summaries = aggregate_multi_run_curve_stats(multi_curves_consistency)
        okg_summary = summaries.get("OKG", []) or summaries.get("okg", [])
        base_summary = summaries.get("Base", []) or summaries.get("base", [])
        if okg_summary:
            msg = ", ".join(
                f"t~{entry['budget']}: {entry['mean']:.4f}±{entry['std']:.4f} (n={entry['num_runs']})"
                for entry in okg_summary
            )
            print(f"[multi-run][consistency] OKG summary: {msg}")
        if base_summary:
            msg = ", ".join(
                f"t~{entry['budget']}: {entry['mean']:.4f}±{entry['std']:.4f} (n={entry['num_runs']})"
                for entry in base_summary
            )
            print(f"[multi-run][consistency] Base summary: {msg}")

    if multi_curves_accuracy:
        summaries = aggregate_multi_run_curve_stats(multi_curves_accuracy)
        okg_summary = summaries.get("OKG", []) or summaries.get("okg", [])
        base_summary = summaries.get("Base", []) or summaries.get("base", [])
        if okg_summary:
            msg = ", ".join(
                f"t~{entry['budget']}: {entry['mean']:.4f}±{entry['std']:.4f} (n={entry['num_runs']})"
                for entry in okg_summary
            )
            print(f"[multi-run][accuracy] OKG summary: {msg}")
        if base_summary:
            msg = ", ".join(
                f"t~{entry['budget']}: {entry['mean']:.4f}±{entry['std']:.4f} (n={entry['num_runs']})"
                for entry in base_summary
            )
            print(f"[multi-run][accuracy] Base summary: {msg}")
