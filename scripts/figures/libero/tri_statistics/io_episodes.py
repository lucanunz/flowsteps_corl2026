"""Per-episode loaders that return dicts keyed by (task_key, episode_id) -> bool success.

Used for paired (McNemar) testing within a single model when comparing the
1-step variant against the max-step variant. Models with per-episode data:
  - Pi0.5 LIBERO (json with episodes[*].metrics.success per task)
  - LAP LIBERO (results_shard*.json with explicit task_id, episode_id)
  - XR-0 LIBERO base (parsed from `*_eval.log` lines)
  - X-VLA LIBERO and LIBERO+ (concatenated `{task: 0.0|1.0}` rollout stream;
    rollouts paired positionally per task on LIBERO base, by unique key on LIBERO+)

For Flower / SmolVLA only aggregate counts exist, so we return None.
"""

from __future__ import annotations

import fnmatch
import json
from collections import defaultdict
import re
import sys
from pathlib import Path
from typing import Any, Optional

EpisodeMap = dict[tuple[str, int], bool]


SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

# ---------------------------------------------------------------------------
# LIBERO+ task-index mapping
# ---------------------------------------------------------------------------

def _load_libero_task_map() -> dict[str, list[str]]:
    """Load canonical task names from the LIBERO benchmark package."""
    return json.loads((Path(__file__).resolve().parents[2] / "task_metadata.json").read_text())["base_task_names"]


_LIBERO_TASK_MAP: dict[str, list[str]] = {}   # lazily populated


def _get_base_task_names(suite: str) -> list[str]:
    """Return the 10 normalised base task names for a LIBERO suite."""
    global _LIBERO_TASK_MAP
    if not _LIBERO_TASK_MAP:
        _LIBERO_TASK_MAP = _load_libero_task_map()
    raw_list = _LIBERO_TASK_MAP.get(suite, [])
    return [
        re.sub(r"^[A-Z_]+SCENE\d+_", "", t).replace("_", " ").lower().strip()
        for t in raw_list
    ]


_NGRAM_STOPWORDS = frozenset({
    "the", "a", "an", "to", "of", "and", "or", "it", "in", "on", "is", "up",
    "at", "by", "as", "be", "do", "so", "if", "from", "with", "please", "pick",
    "place", "black", "bowl", "put", "get", "take", "bring", "move", "transfer",
    "retrieve", "position", "carefully", "into", "onto", "that", "this", "which",
    "their", "your", "our", "need", "could", "would", "should", "must", "will",
    "can", "have", "has", "had", "been", "were", "was", "let", "then", "its",
    "are", "over", "after", "before", "during", "for", "not", "no",
})


def _content_ngrams(text: str) -> frozenset[str]:
    ws = re.sub(r"[^a-z0-9 ]", "", text.lower()).split()
    uni = frozenset(w for w in ws if w not in _NGRAM_STOPWORDS and len(w) > 2)
    bi = frozenset(f"{ws[i]} {ws[i+1]}" for i in range(len(ws) - 1))
    return uni | bi


def _strip_liberoplus_metadata(key: str) -> str:
    """Remove trailing initial-state metadata tokens (table N, view N, etc.).

    Kept for diagnostic use (e.g. the sanity-check confidence score). The
    aggregation path now goes through the BDDL-backed resolver below.
    """
    return re.sub(
        r"\s+(?:table|tb|view|initstate|noise|light|add)\b.*",
        "", key, flags=re.IGNORECASE
    ).strip()


# ---------------------------------------------------------------------------
# BDDL-backed resolver
#
# Each LIBERO+ variant has a BDDL file whose filename encodes the canonical
# base task and whose `(:language ...)` clause holds the prompt the model
# was shown. We build {normalised prompt -> base_idx} per suite once and look
# up each eval prompt after stripping the runtime-suffix the eval harness
# concatenates (e.g. "... language 1 view 0 0 100 0 0 initstate 0").
# ---------------------------------------------------------------------------

_BDDL_ROOT = (
    Path(__file__).resolve().parents[3]
    / "openpi" / "examples" / "LIBERO-plus" / "libero" / "libero" / "bddl_files"
)

_BDDL_LANG_RE = re.compile(r"\(:language\s+(.*?)\)", re.IGNORECASE | re.DOTALL)
_BDDL_NAME_VARIANT = re.compile(
    r"^(?P<base>.+?)_(?:add|table|tb|light|language)_\d+$"
)
_BDDL_NAME_LEVEL = re.compile(r"^(?P<base>.+?)(?:_moved)?_level\d+_sample\d+$")
_SCENE_PREFIX = re.compile(r"^[A-Z_]+SCENE\d+_")
_PUNCT_RE = re.compile(r"[^a-z0-9 ]")
_SUFFIX_KEYWORDS = (
    "language", "view", "initstate", "noise",
    "add", "tb", "table", "light", "sample", "level",
)
_RUNTIME_SUFFIX_RE = re.compile(
    r"\s+(?:" + "|".join(_SUFFIX_KEYWORDS) + r")(?:\s+-?\d+|\d+)+\s*$",
    re.IGNORECASE,
)
_TRAILING_MOVED_RE = re.compile(r"\s+moved\s*$", re.IGNORECASE)

_BDDL_INDEX: dict[str, dict[str, int]] = {}      # suite -> {norm prompt: idx}
_BDDL_INDEX_FAILED: set[str] = set()             # suites we couldn't build


def _normalise_prompt(text: str) -> str:
    return " ".join(_PUNCT_RE.sub(" ", text.lower()).split())


def _canon_filename_base(name: str) -> str:
    return _SCENE_PREFIX.sub("", name).replace("_", " ").lower().strip()


def _strip_runtime_suffix(prompt: str) -> str:
    """Peel stacked runtime suffixes ("view 0 0 100 0 0", "language N", ...)."""
    prev = None
    cur = prompt
    while cur != prev:
        prev = cur
        cur = _RUNTIME_SUFFIX_RE.sub("", cur).rstrip()
        cur = _TRAILING_MOVED_RE.sub("", cur).rstrip()
    return cur


def _build_bddl_index(suite: str) -> dict[str, int]:
    """Build {normalised :language text -> base_idx} for one suite.

    Walks every `*.bddl` in `LIBERO-plus/.../bddl_files/<suite>/`. For each
    file: derive `base_idx` from the filename prefix (matched against the 10
    canonical names in `libero_task_map[suite]`), read `(:language ...)` text,
    and register both the language text and the canonical filename base as
    lookup keys for that index.
    """
    return json.loads((Path(__file__).resolve().parents[2] / "task_metadata.json").read_text())["bddl_index"][suite]


def _get_bddl_index(suite: str) -> dict[str, int]:
    if suite in _BDDL_INDEX:
        return _BDDL_INDEX[suite]
    if suite in _BDDL_INDEX_FAILED:
        return {}
    try:
        _BDDL_INDEX[suite] = _build_bddl_index(suite)
    except FileNotFoundError:
        _BDDL_INDEX_FAILED.add(suite)
        return {}
    return _BDDL_INDEX[suite]


def map_liberoplus_to_task_id(key: str, suite: str) -> int | None:
    """Map a LIBERO+ task variant key to a canonical task index (0–9).

    Looks the prompt up in a per-suite BDDL index built from
    `LIBERO-plus/.../bddl_files/<suite>/*.bddl`. If the prompt isn't found
    verbatim, the eval harness's stacked runtime suffix (e.g.
    ``"... language 1 view 0 0 100 0 0 initstate 0"``) is stripped before
    retrying.

    Returns ``None`` when the suite has no canonical task map or the prompt
    cannot be resolved. (Callers already treat ``None`` as "skip".)
    """
    index = _get_bddl_index(suite)
    if not index:
        return None
    norm = _normalise_prompt(key)
    hit = index.get(norm)
    if hit is not None:
        return hit
    stripped = _normalise_prompt(_strip_runtime_suffix(key))
    return index.get(stripped)


# -- Pi0.5 -------------------------------------------------------------------

def load_pi05_episodes(numstep_dir: Path, suite: str) -> EpisodeMap:
    """Read Pi0.5 LIBERO per-episode JSON for a suite.

    Pi0.5 has no integer task_id; we use the task instruction string as the key.
    """
    path = numstep_dir / f"libero_all_LIBEROBenchmark_{suite}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: EpisodeMap = {}
    for task in raw["tasks"]:
        task_key = task["task"]
        for ep in task.get("episodes", []):
            key = (task_key, int(ep["episode_id"]))
            out[key] = bool(ep["metrics"]["success"])
    return out


# -- LAP --------------------------------------------------------------------

def load_lap_episodes(suite_ns_dir: Path) -> EpisodeMap:
    """Read LAP per-episode shards for `<suite>_ns<steps>/` directory.

    Shard counts vary by suite (libero_spatial uses 04, others use 10), so we
    accept any `results_shard*.json` filename.
    """
    out: EpisodeMap = {}
    for _, raw in read_shard_group(suite_ns_dir, "results_shard*.json"):
        for ep in raw.get("episodes", []):
            key = (str(int(ep["task_id"])), int(ep["episode_id"]))
            out[key] = bool(ep["success"])
    return out


# -- XR-0 ------------------------------------------------------------------

_MIBOT_LINE = re.compile(
    r"INFO:root:Task_id\s+(\d+):\s+Episode\s+(\d+):\s+(Success|Failure)"
)


def _read_episode_records(path: Path, records_key: str = "episodes") -> Optional[EpisodeMap]:
    """Load an `episodes.json` sidecar as an EpisodeMap, preserving record order.

    These sidecars hold the per-episode outcomes that used to be re-derived by
    regex-scanning console logs on every run; see `extract_episodes_from_logs.py`.
    Returns None when the sidecar is absent, so callers fall back to the logs.
    """
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get(records_key)
    if not records:
        return None
    return {(str(r["task_id"]), int(r["episode_id"])): bool(r["success"]) for r in records}


def load_mibot_episodes(step_dir: Path, suite: str, use_json: bool = True) -> Optional[EpisodeMap]:
    """Recover per-episode bools for XR-0 on LIBERO base.

    Prefers the `episodes.json` sidecar in the suite directory, falling back to
    parsing the `[0-9]*_eval.log` console logs. Returns None if neither exists.
    """
    suite_dir = step_dir / suite
    if use_json:
        cached = _read_episode_records(suite_dir / "episodes.json")
        if cached is not None:
            return cached
    log_files = sorted(suite_dir.glob("[0-9]_eval.log")) + sorted(suite_dir.glob("[0-9][0-9]_eval.log"))
    if not log_files:
        return None
    out: EpisodeMap = {}
    for log_path in log_files:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _MIBOT_LINE.search(line)
                if not m:
                    continue
                task_id = m.group(1)
                ep_id = int(m.group(2))
                success = m.group(3) == "Success"
                out[(task_id, ep_id)] = success
    return out if out else None


# -- Flower / XR-0 / Pi0.5 LIBERO+ per-task shards -------------------------

# These three models all emit per-task records on LIBERO+ as
#   {task_string: {"success": int, "failure": int}}
# (Pi0.5 / XR-0) or
#   {"per_task": {task_string: {"num_success": int, "num_rollouts": int}}}
# (Flower). Task strings are globally unique (encode the table/init-state
# variation), so pairing is by exact key.
#
# A few tasks have multiple rollouts. Without per-rollout ordering we cannot
# reconstruct the joint distribution across variants when the within-task
# outcomes differ, so we drop multi-rollout tasks from the paired set and
# record the count in a `paired_diagnostics` dict for the report.

_PSEUDO_KEYS = {"meta", "total"}


def read_shard_group(directory: Path, pattern: str) -> list[tuple[str, Any]]:
    """Return `(stem, document)` pairs for a group of per-shard JSON files.

    The logs keep each such group as one merged object named after its own
    directory (`<dir>/<dir>.json`) mapping every original file stem to its
    document, so a directory that once held thousands of shard files now holds
    one. Loose per-shard files are still read when no merged file is present,
    so the loaders also work against untidied copies of the logs. Order is the
    merged file's key order, which is the original sorted filename order.
    """
    merged = directory / f"{directory.name}.json"
    if merged.exists():
        data = json.loads(merged.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return [(k, v) for k, v in data.items()
                    if fnmatch.fnmatch(f"{k}.json", pattern)]
    return [(path.stem, json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(directory.glob(pattern))]


def _pair_from_count_dict(
    pairs_low: dict[str, tuple[int, int]],
    pairs_high: dict[str, tuple[int, int]],
) -> tuple[EpisodeMap, EpisodeMap, int]:
    """Build paired EpisodeMaps from per-task (success, failure) counts.

    Only single-rollout tasks present in both variants are included. Multi-
    rollout tasks are skipped and counted in the returned `skipped_multi`.

    Task keys are traversed in the low variant's evaluation order, which its
    loader takes from the logs (`task_idx` for Flower, shard task ranges for
    Pi0.5/XR-0). The insertion order here becomes the pooled episode order in
    `_regroup_liberoplus`, and the sequential STEP test is order-dependent, so
    this must be the order the rollouts were actually run in — and never an
    unordered `set` traversal, whose result would vary with PYTHONHASHSEED.
    """
    low: EpisodeMap = {}
    high: EpisodeMap = {}
    skipped_multi = 0
    for task_key in (key for key in pairs_low if key in pairs_high):
        s_lo, f_lo = pairs_low[task_key]
        s_hi, f_hi = pairs_high[task_key]
        if (s_lo + f_lo) != 1 or (s_hi + f_hi) != 1:
            if (s_lo + f_lo) > 1 or (s_hi + f_hi) > 1:
                skipped_multi += 1
            continue
        low[(task_key, 0)] = bool(s_lo == 1)
        high[(task_key, 0)] = bool(s_hi == 1)
    return low, high, skipped_multi


def _order_shards_by_task_range(shard_paths: list[Path]) -> list[Path]:
    """Sort Pi0.5/XR-0 shards by the task range recorded in their `meta`.

    Each shard covers a contiguous `meta.shard_tasks` = [start, end) slice of
    the harness task list and writes its entries in evaluation order, so
    concatenating the shards by ascending start reproduces that order. Falls
    back to the given (filename) order when the metadata is absent.
    """
    keyed: list[tuple[int, Path]] = []
    for path in shard_paths:
        try:
            meta = json.loads(path.read_text(encoding="utf-8")).get("meta", {})
            start = int(meta["shard_tasks"][0])
        except Exception:
            return shard_paths
        keyed.append((start, path))
    return [path for _, path in sorted(keyed, key=lambda item: item[0])]


def _read_pi05_mibot_shards(model_dir: Path, suite: str) -> dict[str, tuple[int, int]]:
    """Read Pi0.5/XR-0 LIBERO+ per-suite shards as {task_key: (success, failure)}."""
    # Pi0.5: pi05/libero_to_liberoplus/videos_pi05_flowstepN/<suite>/results_shard_*.json
    # XR-0: mibot/logs_liberoplus_Nstep[s]/<suite>/<suite>/results_shard_*.json
    out: dict[str, tuple[int, int]] = {}
    candidate_dirs = [model_dir / suite, model_dir / suite / suite]
    shard_paths: list[Path] = []
    for d in candidate_dirs:
        if not d.is_dir():
            continue
        found = sorted(d.glob("results_shard_*.json"))
        if found:
            shard_paths = _order_shards_by_task_range(found)
            break
    for shard_path in shard_paths:
        raw = json.loads(shard_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            continue
        for k, v in raw.items():
            if k in _PSEUDO_KEYS or not isinstance(v, dict):
                continue
            succ = int(v.get("success", 0))
            fail = int(v.get("failure", 0))
            if succ + fail == 0:
                continue
            prev = out.get(k)
            if prev is None:
                out[k] = (succ, fail)
            else:
                out[k] = (prev[0] + succ, prev[1] + fail)
    return out


def load_pi05_liberoplus_episodes(
    flowstep_dir: Path, suite: str
) -> tuple[EpisodeMap, dict[str, tuple[int, int]]]:
    """Return per-task counts for one Pi0.5 LIBERO+ variant.

    `flowstep_dir` is e.g. `pi05/libero_to_liberoplus/videos_pi05_flowstep1`.
    Returns (episode_map_placeholder, raw_counts). The placeholder is empty —
    the actual EpisodeMaps are built by `pair_pi05_liberoplus_episodes`.
    """
    return {}, _read_pi05_mibot_shards(flowstep_dir, suite)


def pair_pi05_liberoplus_episodes(
    low_counts: dict[str, tuple[int, int]],
    high_counts: dict[str, tuple[int, int]],
) -> tuple[EpisodeMap, EpisodeMap, int]:
    return _pair_from_count_dict(low_counts, high_counts)


def load_mibot_liberoplus_episodes(
    step_dir: Path, suite: str
) -> tuple[EpisodeMap, dict[str, tuple[int, int]]]:
    """Return per-task counts for one XR-0 LIBERO+ variant.

    `step_dir` is e.g. `mibot/logs_liberoplus_1step` or `mibot/logs_liberoplus_5steps`.
    """
    return {}, _read_pi05_mibot_shards(step_dir, suite)


def load_flower_liberoplus_episodes(
    suite_dir: Path, steps: int
) -> dict[str, tuple[int, int]]:
    """Read Flower LIBERO+ `nsteps_<steps>/summary_shard_*.json` for one suite.

    `suite_dir` is `flower/liberoplus_logs/<suite>`.
    Returns {normalised_task_name: (successes, total_rollouts)}.
    Keys are normalised (SCENE prefix stripped, underscores → spaces) so they
    can be matched by `map_liberoplus_to_task_id`.

    Keys are returned in harness evaluation order, i.e. ascending `task_idx`.
    The shards are strided by task (shard i holds tasks i, i+num_shards, …),
    so shard-filename order is *not* task order and the index must be used.
    """
    out: dict[str, tuple[int, int]] = {}
    order: dict[str, int] = {}
    nsteps_dir = suite_dir / f"nsteps_{steps}"
    for _, raw in read_shard_group(nsteps_dir, "summary_shard_*.json"):
        per_task = raw.get("per_task", {})
        for raw_key, payload in per_task.items():
            key = normalize_flower_task_name(raw_key)
            n = int(payload.get("num_rollouts", 0))
            s = int(payload.get("num_success", 0))
            if n == 0:
                continue
            f = n - s
            prev = out.get(key)
            out[key] = (s, f) if prev is None else (prev[0] + s, prev[1] + f)
            task_idx = payload.get("task_idx")
            if task_idx is not None:
                order.setdefault(key, int(task_idx))
    if len(order) == len(out):
        return {k: out[k] for k in sorted(out, key=order.__getitem__)}
    return out


def pair_flower_liberoplus_episodes(
    low_counts: dict[str, tuple[int, int]],
    high_counts: dict[str, tuple[int, int]],
) -> tuple[EpisodeMap, EpisodeMap, int]:
    return _pair_from_count_dict(low_counts, high_counts)


# -- X-VLA ------------------------------------------------------------------

def _read_rollout_stream(path: Path) -> list[dict]:
    """Read an X-VLA per-rollout results file as a list of records, in order.

    These files are plain JSON arrays. They were once a bare concatenation of
    JSON objects (`{...}{...}{...}`), which needed incremental `raw_decode`
    scanning; the fallback below still reads that legacy layout so the loader
    works against untidied copies of the logs. Rollout order is significant —
    it supplies the positional episode ids — so both paths preserve it.
    """
    text = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except ValueError:
        dec = json.JSONDecoder()
        out: list[dict] = []
        i = 0
        while i < len(text):
            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text):
                break
            record, i = dec.raw_decode(text, i)
            out.append(record)
        return out
    if not isinstance(obj, list):
        raise ValueError(f"{path}: expected a JSON array of rollout records")
    return obj


def load_xvla_episodes(steps_dir: Path, suite: str) -> Optional[EpisodeMap]:
    """Read X-VLA per-rollout stream for one suite under `libero_<N>steps/`.

    The file format is a sequence of JSON objects, each `{task_string: 0.0|1.0}`.
    Summary entries `sim_summary/...` are skipped.

    LIBERO base: tasks repeat 50 times each; we assign a positional episode_id
        (0..49) per task. Pairing across 1-step vs max-step assumes the harness
        enumerates the same deterministic seed sequence in the same order — true
        for the X-VLA eval harness based on identical task ordering.
    LIBERO+: each task key is unique (e.g., "task table 1", "task table 10"),
        so episode_id is always 0 and pairing is by exact key.
    """
    suite_path = steps_dir / suite / "results.json"
    if not suite_path.exists():
        return None
    entries = _read_rollout_stream(suite_path)
    counters: dict[str, int] = defaultdict(int)
    out: EpisodeMap = {}
    for obj in entries:
        if not obj:
            continue
        key = next(iter(obj))
        val = obj[key]
        if key.startswith("sim_summary/"):
            continue
        if not isinstance(val, (int, float)):
            continue
        # Drop the "sim/<suite>/" prefix so the task key matches across variants.
        task_key = key.split("/", 2)[-1] if "/" in key else key
        ep_id = counters[task_key]
        counters[task_key] += 1
        out[(task_key, ep_id)] = bool(val == 1.0)
    return out if out else None


# -- Per-task grouping and name resolution ----------------------------------

def group_episodes_by_task(
    ep_map: EpisodeMap,
) -> dict[str, dict[int, bool]]:
    """Split an EpisodeMap into {task_key: {episode_id: success}}."""
    out: dict[str, dict[int, bool]] = defaultdict(dict)
    for (task_key, ep_id), success in ep_map.items():
        out[task_key][ep_id] = success
    return dict(out)


def resolve_lap_task_descriptions(suite_ns_dir: Path) -> dict[str, str]:
    """Return {str(task_id): task_description} from LAP shards."""
    mapping: dict[str, str] = {}
    for _, raw in read_shard_group(suite_ns_dir, "results_shard*.json"):
        for ep in raw.get("episodes", []):
            tid = str(int(ep["task_id"]))
            if tid not in mapping:
                mapping[tid] = ep["task_description"]
    return mapping


def resolve_mibot_task_descriptions(step_dir: Path, suite: str) -> dict[str, str]:
    """Return {str(task_id): instruction} from XR-0 merged_results.json."""
    path = step_dir / suite / "merged_results.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(r["task_id"]): r["instruction"] for r in raw.get("individual_results", [])}


def normalize_flower_task_name(key: str) -> str:
    """Strip SCENE_N_ prefix from Flower task keys and return lowercase instruction."""
    return re.sub(r"^[A-Z_]+SCENE\d+_", "", key).replace("_", " ").lower()


# Per-task aggregate counts for models that don't expose per-episode data.

def load_flower_base_per_task(
    suite_dir: Path, steps: int, suite: str
) -> dict[str, tuple[int, int]]:
    """Read Flower LIBERO base per-task counts from nsteps_<steps>/summary*.json.

    Returns {normalised_task_name: (successes, total_trials)}.
    """
    out: dict[str, tuple[int, int]] = {}
    nsteps_dir = suite_dir / suite / f"nsteps_{steps}"
    documents = [doc for _, doc in read_shard_group(nsteps_dir, "summary_shard_*.json")]
    if not documents:
        summary = nsteps_dir / "summary.json"
        if summary.exists():
            documents = [json.loads(summary.read_text(encoding="utf-8"))]
    for raw in documents:
        for raw_key, payload in raw.get("per_task", {}).items():
            key = normalize_flower_task_name(raw_key)
            n = int(payload.get("num_rollouts", 0))
            s = int(payload.get("num_success", 0))
            if n == 0:
                continue
            prev = out.get(key)
            out[key] = (s, n) if prev is None else (prev[0] + s, prev[1] + n)
    return out


def load_flower_libero_episodes(
    libero_logs_dir: Path, steps: int, suite: str
) -> EpisodeMap:
    """Read Flower LIBERO base per-episode rollouts.

    `libero_logs_dir` is `flower/libero_logs`. Prefers per-shard summary files,
    falling back to the aggregated `summary.json`. Emits keys of the form
    `(normalize_flower_task_name(raw_key), episode_idx) -> bool(success)`.
    """
    out: EpisodeMap = {}
    nsteps_dir = libero_logs_dir / suite / f"nsteps_{steps}"
    documents = [doc for _, doc in read_shard_group(nsteps_dir, "summary_shard_*.json")]
    if not documents:
        summary = nsteps_dir / "summary.json"
        if summary.exists():
            documents = [json.loads(summary.read_text(encoding="utf-8"))]
    for raw in documents:
        for raw_key, payload in raw.get("per_task", {}).items():
            key = normalize_flower_task_name(raw_key)
            for rollout in payload.get("rollouts", []) or []:
                ep_id = int(rollout["episode_idx"])
                out[(key, ep_id)] = bool(int(rollout["success"]) == 1)
    return out


def load_smolvla_per_task(summary_path: Path, suite: str) -> dict[str, tuple[int, int]]:
    """Read SmolVLA per-task counts from summary.json.

    Returns {task_name: (successes, total_trials)}.
    """
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    tasks = raw.get("suites", {}).get(suite, {}).get("tasks", {})
    return {
        name: (int(v["success"]), int(v["total"]))
        for name, v in tasks.items()
        if int(v.get("total", 0)) > 0
    }


def load_smolvla_paired_episodes(paired_json_path: Path, suite: str) -> EpisodeMap:
    """Read SmolVLA per-rollout outcomes for one LIBERO suite.

    `paired_json_path` is e.g. `smolvla/libero_logs/paired/videos_smolvla_flowstep1_actionstep10_episodes.json`
    (or `smolvla/liberoplus_logs/paired/...` for the LIBERO+ counterpart).
    The file is a flat list of `{task_suite_name, task_id, task_description, episode_idx, success}`.
    Returns `{(str(task_id), episode_idx): bool(success)}` for the requested suite.
    """
    raw = json.loads(paired_json_path.read_text(encoding="utf-8"))
    out: EpisodeMap = {}
    for ep in raw:
        if ep.get("task_suite_name") != suite:
            continue
        key = (str(int(ep["task_id"])), int(ep["episode_idx"]))
        out[key] = bool(ep["success"])
    return out


def resolve_smolvla_task_descriptions(paired_json_path: Path, suite: str) -> dict[str, str]:
    """Return `{str(task_id): task_description}` for one SmolVLA suite."""
    raw = json.loads(paired_json_path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for ep in raw:
        if ep.get("task_suite_name") != suite:
            continue
        tid = str(int(ep["task_id"]))
        if tid not in mapping:
            mapping[tid] = ep["task_description"]
    return mapping


# -- Helpers ----------------------------------------------------------------

def discordant_counts(
    low: EpisodeMap, high: EpisodeMap
) -> tuple[int, int, int, int, int]:
    """Return (n_paired, both_succ, b, c, both_fail) over the intersection of keys.

    Following McNemar conventions:
      b = low success, high failure
      c = low failure, high success
    """
    keys = set(low.keys()) & set(high.keys())
    n_paired = len(keys)
    both_s = both_f = b = c = 0
    for k in keys:
        sl = low[k]
        sh = high[k]
        if sl and sh:
            both_s += 1
        elif (not sl) and (not sh):
            both_f += 1
        elif sl and not sh:
            b += 1
        else:
            c += 1
    return n_paired, both_s, b, c, both_f


# -- DiT --------------------------------------------------------------------

_DIT_EP_LINE = re.compile(
    r"\[(?P<suite>[a-zA-Z0-9_]+)/task_(?P<task>\d+)\]\s+ep\s+(?P<ep>\d+):"
)
_DIT_DONE_LINE = re.compile(r"done=(?P<done>True|False)")


def load_dit_log_episodes(shards_dir: Path, suite: str, use_json: bool = True) -> EpisodeMap:
    """Per-episode outcomes for DiT.

    Prefers the `episodes.json` sidecar beside the shards, falling back to
    parsing the sharded eval logs.

    Each shard log emits pairs of lines like
        [libero_spatial/task_2] ep 17: <description>
        ...  done=True  task_rate=...
    Pairing is by (task_id, episode_idx); the fixed init-state seed makes this
    stable across the two flow-step variants.
    """
    if use_json:
        path = shards_dir / "episodes.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("suites", {}).get(suite)
            if records:
                return {(str(r["task_id"]), int(r["episode_id"])): bool(r["success"])
                        for r in records}
    out: EpisodeMap = {}
    log_files = sorted(shards_dir.glob("shard*-of-*.log"))
    for log_path in log_files:
        pending: tuple[str, int] | None = None
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m_ep = _DIT_EP_LINE.search(line)
                if m_ep:
                    if m_ep.group("suite") == suite:
                        pending = (m_ep.group("task"), int(m_ep.group("ep")))
                    else:
                        pending = None
                    continue
                if pending is None:
                    continue
                m_done = _DIT_DONE_LINE.search(line)
                if m_done:
                    out[pending] = m_done.group("done") == "True"
                    pending = None
    return out


def resolve_dit_task_descriptions(final_results_path: Path) -> dict[str, str]:
    """Map task id (no `task_` prefix) -> description from DiT per-suite JSON."""
    raw = json.loads(final_results_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for key, payload in raw.get("per_task", {}).items():
        if not key.startswith("task_"):
            continue
        tid = key[len("task_"):]
        desc = payload.get("description")
        if isinstance(desc, str):
            out[tid] = desc
    return out
