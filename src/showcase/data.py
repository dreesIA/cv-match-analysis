"""Load pipeline bundles and turn their events into showcase-ready clip windows.

A bundle is intentionally simple: one ``events.json``, one MP4, and an optional
``run_summary.json`` in the same directory.  Keeping that contract small makes it
easy to copy a training run into the showcase without a conversion step.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


VIDEO_PREFERENCE = (
    "showcase.mp4",
    "tracking.mp4",
    "annotated.mp4",
)
SKIP_VIDEO_TOKENS = ("radar_", "voronoi", "triangulation")


@dataclass(frozen=True)
class ShowcaseBundle:
    """A validated training-output bundle available to the dashboard."""

    name: str
    directory: Path
    video_path: Path
    events_path: Optional[Path]
    summary_path: Optional[Path]
    radar_path: Optional[Path]
    events: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    duration_s: float

    @property
    def key(self) -> str:
        return str(self.directory.resolve())


@dataclass(frozen=True)
class ClipWindow:
    """A seekable view over a section of a source video."""

    key: str
    label: str
    detail: str
    start_s: float
    end_s: float
    focus_event_ids: tuple[int, ...] = ()

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _point(row: dict[str, Any], stem: str) -> Optional[list[float]]:
    """Read either nested ``start_xy_m`` or flattened parquet-style columns."""
    value = row.get(stem)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [_number(value[0]), _number(value[1])]
    x, y = row.get(f"{stem}_x"), row.get(f"{stem}_y")
    if x is None or y is None:
        return None
    return [_number(x), _number(y)]


def normalise_event(row: dict[str, Any], event_id: int) -> dict[str, Any]:
    """Return the compact, JSON-safe shape consumed by the browser player."""
    start_s = max(0.0, _number(row.get("start_ts")))
    end_s = max(start_s, _number(row.get("end_ts"), start_s))
    player_ids = row.get("player_ids") or []
    attrs = row.get("attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}
    return {
        "id": int(event_id),
        "type": str(row.get("type") or "event"),
        "start": start_s,
        "end": end_s,
        "team": row.get("team_id"),
        "players": [int(p) for p in player_ids if p is not None],
        "start_xy": _point(row, "start_xy_m"),
        "end_xy": _point(row, "end_xy_m"),
        "length": _number(row.get("length_m")),
        "confidence": min(1.0, max(0.0, _number(row.get("confidence")))),
        "status": str(row.get("review_status") or "review"),
        "attributes": attrs,
    }


def _load_events(path: Path) -> tuple[dict[str, Any], ...]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("events", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("events.json must contain a list or an object with an events list")
    events = [normalise_event(row, i) for i, row in enumerate(rows) if isinstance(row, dict)]
    return tuple(sorted(events, key=lambda event: (event["start"], event["end"], event["id"])))


def _load_summary(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _choose_video(directory: Path) -> Optional[Path]:
    candidates = [path for path in directory.glob("*.mp4") if path.is_file()]
    if not candidates:
        return None
    by_name = {path.name.lower(): path for path in candidates}
    for preferred in VIDEO_PREFERENCE:
        if preferred in by_name:
            return by_name[preferred]
    annotated = [
        path for path in candidates
        if not any(token in path.name.lower() for token in SKIP_VIDEO_TOKENS)
    ]
    if annotated:
        return sorted(annotated, key=lambda path: ("gs" not in path.stem.lower(), path.name))[0]
    return sorted(candidates)[0]


def _probe_duration(video_path: Path, events: Iterable[dict[str, Any]]) -> float:
    """Read video duration without making OpenCV a hard dependency for the UI."""
    duration = 0.0
    try:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if fps > 0 and frames > 0:
            duration = frames / fps
    except Exception:
        pass
    event_end = max((float(event["end"]) for event in events), default=0.0)
    if duration > 0:
        return max(duration, event_end, 1.0)
    return max(event_end + 1.0, 1.0)


def load_bundle(events_path: Path) -> ShowcaseBundle:
    """Validate and load one directory containing the showcase bundle contract."""
    events_path = events_path.expanduser().resolve()
    directory = events_path.parent
    video_path = _choose_video(directory)
    if video_path is None:
        raise ValueError(f"No MP4 found next to {events_path.name}")
    summary_path = directory / "run_summary.json"
    if not summary_path.exists():
        summary_path = None
    events = _load_events(events_path)
    return ShowcaseBundle(
        name=directory.name.replace("_", " ").strip().title(),
        directory=directory,
        video_path=video_path.resolve(),
        events_path=events_path,
        summary_path=summary_path,
        radar_path=(directory / "radar_combined.mp4").resolve()
        if (directory / "radar_combined.mp4").exists() else None,
        events=events,
        summary=_load_summary(summary_path),
        duration_s=_probe_duration(video_path, events),
    )


def default_search_roots(workspace: Optional[Path] = None) -> list[Path]:
    """Portable defaults plus the local Downloads convention used by training runs."""
    base = (workspace or Path.cwd()).resolve()
    roots = [base / "data" / "showcase", base / "outputs"]
    downloads = Path.home() / "Downloads"
    if downloads.exists():
        roots.append(downloads)
    configured = os.environ.get("UPDT_SHOWCASE_ROOT", "")
    for raw in configured.split(os.pathsep):
        if raw.strip():
            roots.append(Path(raw.strip()).expanduser())
    return roots


def discover_bundles(
    roots: Optional[Iterable[Path | str]] = None,
    *,
    max_depth: int = 3,
) -> list[ShowcaseBundle]:
    """Find valid bundles, ignoring incomplete pipeline output directories."""
    search_roots = [Path(root).expanduser() for root in (roots or default_search_roots())]
    seen: set[Path] = set()
    found: list[ShowcaseBundle] = []
    for root in search_roots:
        if not root.exists():
            continue
        event_paths = [root] if root.is_file() and root.name == "events.json" else []
        if root.is_dir():
            event_paths.extend(root.glob("events.json"))
            for depth in range(1, max_depth + 1):
                pattern = "/".join(["*"] * depth + ["events.json"])
                event_paths.extend(root.glob(pattern))
        for path in event_paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                found.append(load_bundle(resolved))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
    return sorted(found, key=lambda bundle: bundle.name.lower())


def discover_bundesliga_clips(
    roots: Optional[Iterable[Path | str]] = None,
) -> list[ShowcaseBundle]:
    """Load only complete DFL/Bundesliga pipeline bundles.

    ``events.json`` is mandatory.  In particular, this loader never reconstructs
    event timestamps from a rendered overlay: every chart point must be an artifact
    emitted by :mod:`src.tagging.events`.
    """
    search_roots = [Path(root).expanduser() for root in (roots or default_search_roots())]
    directories: set[Path] = set()
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        has_video = any((root / f"showcase{suffix}").exists() for suffix in (".m4v", ".mp4"))
        if has_video and (root / "events.json").exists() and "bundesliga" in root.parts:
            directories.add(root.resolve())
        candidates = list(root.glob("bundesliga/*"))
        for candidate in candidates:
            has_video = any(
                (candidate / f"showcase{suffix}").exists()
                for suffix in (".m4v", ".mp4")
            )
            if candidate.is_dir() and has_video and (candidate / "events.json").exists():
                directories.add(candidate.resolve())

    found: list[ShowcaseBundle] = []
    for directory in directories:
        video_path = next(
            directory / f"showcase{suffix}"
            for suffix in (".m4v", ".mp4")
            if (directory / f"showcase{suffix}").exists()
        )
        radar_path = next((
            directory / f"radar_combined{suffix}"
            for suffix in (".m4v", ".mp4")
            if (directory / f"radar_combined{suffix}").exists()
        ), None)
        events_path = (directory / "events.json").resolve()
        summary_path = directory / "run_summary.json"
        if not summary_path.exists():
            summary_path = None
        try:
            events = _load_events(events_path)
            summary = _load_summary(summary_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        duration_s = _probe_duration(video_path, events)
        # The deployed UI intentionally has a minimal dependency set and may not
        # have OpenCV available to probe media. Pipeline metadata still gives an
        # exact duration for sampled runs.
        processed = _number(summary.get("frames_processed"))
        effective_fps = _number(summary.get("effective_fps"))
        if processed > 0 and effective_fps > 0:
            duration_s = max(duration_s, processed / effective_fps)
        source_name = str(summary.get("source", directory.name)).replace("_", " ")
        found.append(ShowcaseBundle(
            name=f"{Path(source_name).stem} · {int(round(duration_s))}s",
            directory=directory,
            video_path=video_path.resolve(),
            events_path=events_path,
            summary_path=summary_path,
            radar_path=radar_path.resolve() if radar_path else None,
            events=events,
            summary=summary,
            duration_s=duration_s,
        ))
    # Prefer genuinely per-frame tracking bundles when they exist. This keeps a
    # lower-quality sampled experiment from appearing beside the interview-ready
    # result, while retaining backward compatibility for older bundle folders.
    full_rate = [
        bundle for bundle in found
        if int(_number(bundle.summary.get("sampling_stride"), 0)) == 1
    ]
    return sorted(full_rate or found, key=lambda bundle: bundle.name.lower())


def _clock(seconds: float) -> str:
    minutes, remainder = divmod(max(0, int(round(seconds))), 60)
    return f"{minutes:02d}:{remainder:02d}"


def _friendly_type(event_type: str) -> str:
    return {
        "zone_entry": "Entry",
        "possession_change": "Possession change",
    }.get(event_type, event_type.replace("_", " ").title())


def _plural_type(event_type: str, count: int) -> str:
    singular = _friendly_type(event_type).lower()
    if count == 1:
        return singular
    return {
        "pass": "passes",
        "entry": "entries",
        "possession change": "possession changes",
    }.get(singular, f"{singular}s")


def event_label(event: dict[str, Any]) -> str:
    event_type = event["type"]
    players = event.get("players") or []
    attrs = event.get("attributes") or {}
    if event_type == "pass" and players:
        path = " → ".join(f"#{player}" for player in players[:2])
        return f"Pass · {path}"
    if event_type == "zone_entry":
        zone = str(attrs.get("zone") or "pitch zone").replace("_", " ").title()
        return f"Entry · {zone}"
    team = event.get("team")
    suffix = f" · Team {team}" if team is not None else ""
    return f"{_friendly_type(event_type)}{suffix}"


def build_clip_windows(
    events: Iterable[dict[str, Any]],
    duration_s: float,
    *,
    padding_s: float = 3.5,
) -> list[ClipWindow]:
    """Create full-session, action-sequence, and individual-action selections."""
    ordered = sorted(events, key=lambda event: (event["start"], event["end"]))
    duration_s = max(1.0, float(duration_s))
    clips = [ClipWindow(
        key="full",
        label="Full training clip",
        detail=f"Complete session · {_clock(duration_s)}",
        start_s=0.0,
        end_s=duration_s,
        focus_event_ids=tuple(int(event["id"]) for event in ordered),
    )]

    # Cluster discrete actions into coherent sequences. Long possession intervals are
    # excluded because they otherwise swallow every sequence in the clip.
    discrete = [event for event in ordered if event["type"] != "possession"]
    clusters: list[list[dict[str, Any]]] = []
    for event in discrete:
        if not clusters or event["start"] - clusters[-1][-1]["end"] > 8.0:
            clusters.append([event])
        else:
            clusters[-1].append(event)
    for sequence_no, cluster in enumerate((c for c in clusters if len(c) >= 2), start=1):
        start = max(0.0, cluster[0]["start"] - padding_s)
        end = min(duration_s, max(cluster[-1]["end"], cluster[-1]["start"] + 1.0) + padding_s)
        counts: dict[str, int] = {}
        for event in cluster:
            name = _friendly_type(event["type"]).lower()
            counts[name] = counts.get(name, 0) + 1
        breakdown = " · ".join(
            f"{count} {_plural_type(name, count)}" for name, count in counts.items()
        )
        clips.append(ClipWindow(
            key=f"sequence-{sequence_no}",
            label=f"Action sequence {sequence_no}",
            detail=f"{_clock(start)}–{_clock(end)} · {breakdown}",
            start_s=start,
            end_s=end,
            focus_event_ids=tuple(int(event["id"]) for event in cluster),
        ))

    for event in ordered:
        start = max(0.0, event["start"] - padding_s)
        event_end = max(event["end"], event["start"] + 1.0)
        end = min(duration_s, event_end + padding_s)
        clips.append(ClipWindow(
            key=f"event-{event['id']}",
            label=event_label(event),
            detail=f"{_clock(event['start'])} · {event['confidence']:.0%} confidence",
            start_s=start,
            end_s=max(start + 0.5, end),
            focus_event_ids=(int(event["id"]),),
        ))
    return clips
