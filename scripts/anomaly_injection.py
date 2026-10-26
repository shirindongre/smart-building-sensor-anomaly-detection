"""
Synthetic anomaly injection for evaluating the detector.

Real data has no labels, so we corrupt a copy of the hourly energy series in known places and check whether the
detector flags them. Four event types (settings in ``src.config.EVAL``):

* spike        - a short burst: energy + N x std(energy)          (1-3 h)
* dropout      - supply / sensor outage: energy x ~0               (6-24 h)
* flatline     - stuck sensor: the reading freezes at its last value (12-36 h)
* level_shift  - sustained increase: energy x 1.3-1.6              (48-72 h)

Events are placed ONLY inside the evaluation regions passed in, one per "slot", so they never touch training data.

Realism: an outage in a building that is idle anyway (a school at night, in the holidays) changes nothing, so a
detector cannot be expected to find it. Therefore
* dropout / flatline / level_shift are placed where the clean load is active (``min_active_frac`` of the building's
  ``active_quantile`` level, averaged over the event); spikes can occur anywhere (they are added on top of the load);
* ``inject`` measures each event's real effect (mean |change| in units of the series' std). Events below
  ``min_effect_std`` are flagged ``inert``: they stay in the data, are listed in the events CSV, but are left out of
  recall / precision / AUC (their windows count as "ignored", not as clean).
All positions are row positions in the building's FEATURE frame; ``inject`` maps them to timestamps on the raw series.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Event:
    event_id: int
    type: str
    start: int  # first affected row (feature-frame row position)
    end: int  # last affected row (inclusive)
    magnitude: float  # spike: amplitude in energy units; dropout / level_shift: factor; flatline: held value (filled in by inject)
    effect_std: float = float("nan")  # mean |injected - clean| over the event, in std of the building's energy (set by inject)
    inert: bool = False  # effect below min_effect_std: the event changed (almost) nothing (set by inject)
    active_frac: float = float("nan")  # mean clean load over the event / active level (set by place_events)

    def as_dict(self):
        return asdict(self)


def place_events(regions, window: int, cfg: dict, rng: np.random.Generator, slot_rows: int, first_id: int = 0,
                 activity: np.ndarray | None = None):
    """
    regions: list of (start_row, end_row_exclusive) evaluation regions (held-out blocks, or the test period).
    Each region is cut into slots of ``slot_rows`` rows and gets one event per full slot. An event starts at least
    ``window`` rows after its slot start and ends at least ``window`` rows before its slot end, so every window that
    touches it lies fully inside the region.

    ``activity`` (optional): per feature row, clean energy divided by the building's active level (see
    ``activity_profile``). When given, dropout / flatline / level_shift events try up to ``placement_tries`` start
    positions in their slot and take the first whose mean activity is >= ``min_active_frac``; if none qualifies the most
    active candidate is kept (``inject`` then flags it ``inert`` if it really changed nothing).
    """
    min_act = float(cfg.get("min_active_frac", 0.0))
    tries = max(int(cfg.get("placement_tries", 1)), 1)
    types = list(cfg["event_types"])
    events, eid, order = [], first_id, []
    for (rs, re_) in regions:
        n_slots = max((re_ - rs) // slot_rows, 0)
        for k in range(n_slots):
            order.append((rs + k * slot_rows, rs + (k + 1) * slot_rows))
    # deal the types out evenly (shuffled cycle) so every type appears about equally often
    type_seq = []
    while len(type_seq) < len(order):
        perm = list(rng.permutation(types))
        type_seq.extend(perm)
    for (ss, se), typ in zip(order, type_seq):
        lo_h, hi_h = cfg[typ]["hours"]
        length = int(rng.integers(lo_h, hi_h + 1))
        a_min, a_max = ss + window, se - window - length - 1
        if a_max < a_min:  # slot too short for this event - skip it
            continue
        constrained = activity is not None and typ != "spike" and min_act > 0
        best = None
        for _ in range(tries if constrained else 1):
            cand = int(rng.integers(a_min, a_max + 1))
            act = float(np.mean(activity[cand:cand + length])) if activity is not None else float("nan")
            if best is None or (constrained and act > best[1]):
                best = (cand, act)
            if not constrained or act >= min_act:
                best = (cand, act)
                break
        start, act_frac = best
        end = start + length - 1
        if typ == "spike":
            lo, hi = cfg["spike"]["amplitude_std"]
            mag = float(rng.uniform(lo, hi))
        elif typ == "dropout":
            lo, hi = cfg["dropout"]["factor"]
            mag = float(rng.uniform(lo, hi))
        elif typ == "level_shift":
            lo, hi = cfg["level_shift"]["factor"]
            mag = float(rng.uniform(lo, hi))
        else:  # flatline
            mag = float("nan")
        events.append(Event(eid, typ, start, end, mag, active_frac=act_frac))
        eid += 1
    return events


def activity_profile(clean_energy: np.ndarray, quantile: float = 0.9) -> np.ndarray:
    """Clean energy per row relative to the building's active level (the ``quantile`` of its energy)."""
    level = float(np.quantile(clean_energy, quantile))
    return np.asarray(clean_energy, dtype=float) / max(level, 1e-9)


def inject(raw_energy: pd.Series, feature_index: pd.DatetimeIndex, events, energy_std: float,
           min_effect_std: float = 0.0) -> tuple[pd.Series, list]:
    """
    Return (corrupted copy of ``raw_energy``, events with magnitudes resolved to energy units).
    ``feature_index`` maps an event's row positions to timestamps; ``raw_energy`` is the hourly UTC series the features
    were built from (it starts earlier than the feature frame because of the 168 h look-back).
    """
    out = raw_energy.copy().astype(float)
    resolved = []
    for ev in events:
        t0, t1 = feature_index[ev.start], feature_index[ev.end]
        sl = (out.index >= t0) & (out.index <= t1)
        if not sl.any():
            continue
        before = out[out.index < t0]
        if ev.type == "spike":
            out[sl] = out[sl] + ev.magnitude * energy_std
            mag = ev.magnitude * energy_std
        elif ev.type == "dropout":
            out[sl] = out[sl] * ev.magnitude
            mag = ev.magnitude
        elif ev.type == "level_shift":
            out[sl] = out[sl] * ev.magnitude
            mag = ev.magnitude
        else:  # flatline: freeze at the last reading before the event
            held = float(before.iloc[-1]) if len(before) else float(out[sl].iloc[0])
            out[sl] = held
            mag = held
        out[sl] = out[sl].clip(lower=0.0)
        effect = float((out[sl] - raw_energy[sl].astype(float)).abs().mean() / max(energy_std, 1e-9))
        resolved.append(Event(ev.event_id, ev.type, ev.start, ev.end, float(mag), effect_std=effect,
                              inert=bool(effect < min_effect_std), active_frac=ev.active_frac))
    out = out.clip(lower=0.0)
    return out, resolved
