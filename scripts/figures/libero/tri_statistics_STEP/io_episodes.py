"""Episode-data loaders for the STEP variant.

Re-export of `tri_statistics.io_episodes` plus one extra helper —
`paired_sequences(low, high)` — which returns the aligned per-key success
vectors that STEP needs to operate on. The original module exposes
`discordant_counts` (counts only); STEP requires the raw sequences.
"""

from __future__ import annotations

from typing import Sequence

from tri_statistics.io_episodes import (  # noqa: F401
    EpisodeMap,
    _get_base_task_names,
    discordant_counts,
    group_episodes_by_task,
    load_dit_log_episodes,
    load_flower_base_per_task,
    load_flower_libero_episodes,
    load_flower_liberoplus_episodes,
    load_lap_episodes,
    load_mibot_episodes,
    load_mibot_liberoplus_episodes,
    load_pi05_episodes,
    load_pi05_liberoplus_episodes,
    load_smolvla_paired_episodes,
    load_smolvla_per_task,
    load_xvla_episodes,
    map_liberoplus_to_task_id,
    pair_flower_liberoplus_episodes,
    pair_pi05_liberoplus_episodes,
    resolve_dit_task_descriptions,
    resolve_lap_task_descriptions,
    resolve_mibot_task_descriptions,
    resolve_smolvla_task_descriptions,
)


def paired_sequences(low: EpisodeMap, high: EpisodeMap) -> tuple[list[bool], list[bool]]:
    """Return aligned (seq_low, seq_high) over the intersection of keys.

    Keys are visited in a deterministic order (sorted by their string form) so
    that the resulting sequences are stable across runs. STEP itself doesn't
    care about order — it's a sequential test on i.i.d.-style Bernoulli
    streams — but a stable order makes results reproducible.
    """
    shared = sorted(set(low) & set(high), key=lambda k: str(k))
    seq_low = [bool(low[k]) for k in shared]
    seq_high = [bool(high[k]) for k in shared]
    return seq_low, seq_high


def paired_sequences_per_task(
    ep_low_task: dict[int, bool],
    ep_high_task: dict[int, bool],
) -> tuple[list[bool], list[bool]]:
    """Aligned per-task sequences (shared episode IDs only)."""
    shared = sorted(set(ep_low_task) & set(ep_high_task))
    seq_low = [bool(ep_low_task[eid]) for eid in shared]
    seq_high = [bool(ep_high_task[eid]) for eid in shared]
    return seq_low, seq_high


__all__ = [
    "EpisodeMap",
    "_get_base_task_names",
    "discordant_counts",
    "group_episodes_by_task",
    "load_dit_log_episodes",
    "load_flower_base_per_task",
    "load_flower_libero_episodes",
    "load_flower_liberoplus_episodes",
    "load_lap_episodes",
    "load_mibot_episodes",
    "load_mibot_liberoplus_episodes",
    "load_pi05_episodes",
    "load_pi05_liberoplus_episodes",
    "load_smolvla_paired_episodes",
    "load_smolvla_per_task",
    "load_xvla_episodes",
    "map_liberoplus_to_task_id",
    "pair_flower_liberoplus_episodes",
    "pair_pi05_liberoplus_episodes",
    "resolve_dit_task_descriptions",
    "resolve_lap_task_descriptions",
    "resolve_mibot_task_descriptions",
    "resolve_smolvla_task_descriptions",
    "paired_sequences",
    "paired_sequences_per_task",
]
