"""Offline labels only; forecasting runtime must never import annotations."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RhythmInterval:
    start: float
    end: float
    rhythm: str

    def __post_init__(self):
        if (
            not math.isfinite(self.start)
            or not math.isfinite(self.end)
            or self.start < 0
            or self.end <= self.start
            or self.rhythm not in {"AF", "other", "unknown"}
        ):
            raise ValueError("Invalid rhythm interval")


def _intervals(intervals, duration):
    result = []
    for interval in sorted(intervals, key=lambda i: i.start):
        if interval.end > duration or (result and interval.start < result[-1].end):
            raise ValueError("Overlapping or out-of-record rhythm intervals")
        if result and result[-1].end == interval.start and result[-1].rhythm == interval.rhythm:
            previous = result.pop()
            interval = RhythmInterval(previous.start, interval.end, interval.rhythm)
        result.append(interval)
    return result


def _covered(intervals, start, end, allowed):
    cursor = start
    for interval in intervals:
        if interval.end <= cursor:
            continue
        if interval.start > cursor or interval.rhythm not in allowed:
            return False
        cursor = min(end, interval.end)
        if cursor >= end:
            return True
    return cursor >= end


def eligible_onsets(intervals, duration, config):
    """Events with at least one valid forecast time and continuous labelled non-AF history."""
    target = config["target"]
    history, gap, horizon = (target[k] for k in ["history_seconds", "gap_seconds", "horizon_seconds"])
    intervals = _intervals(intervals, duration)
    onsets = []
    previous_af_end = -math.inf
    for index, interval in enumerate(intervals):
        if interval.rhythm != "AF":
            continue
        preceding = intervals[index - 1] if index else None
        if preceding and preceding.rhythm == "other" and preceding.end == interval.start:
            # Earliest valid warning avoids requiring labels beyond its own forecast horizon.
            t = max(
                preceding.start + history,
                interval.start - horizon,
                previous_af_end + target["washout_seconds"],
            )
            if (
                interval.end - interval.start >= target["min_episode_seconds"]
                and t <= min(interval.start - gap, duration - horizon)
                and _covered(intervals, t - history, interval.start, {"other"})
                and _covered(intervals, t, t + horizon, {"other", "AF"})
            ):
                onsets.append(interval.start)
        previous_af_end = interval.end
    return onsets


def label_at(t, intervals, duration, config):
    """Inclusive onset target [t+gap,t+horizon]; None means excluded, never negative."""
    target = config["target"]
    history, gap, horizon = (target[k] for k in ["history_seconds", "gap_seconds", "horizon_seconds"])
    if not math.isfinite(t) or t < 0:
        raise ValueError("Prediction time must be finite and nonnegative")
    intervals = _intervals(intervals, duration)
    if any(i.start <= t < i.end and i.rhythm == "AF" for i in intervals):
        return None, "current_AF"
    if t < history:
        return None, "history"
    if t + horizon > duration:
        return None, "future"
    if any(i.rhythm == "AF" and i.end <= t and t - i.end < target["washout_seconds"] for i in intervals):
        return None, "washout"
    if not _covered(intervals, t - history, t + horizon, {"other", "AF"}):
        return None, "unknown"
    if not _covered(intervals, t - history, t, {"other"}):
        return None, "history"
    if any(i.rhythm == "AF" and t < i.start < t + gap for i in intervals):
        return None, "gap"
    future = [i.start for i in intervals if i.rhythm == "AF" and t + gap <= i.start <= t + horizon]
    if future:
        eligible = eligible_onsets(intervals, duration, config)
        return (1, None) if any(onset in eligible for onset in future) else (None, "ineligible_event")
    return 0, None
