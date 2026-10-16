#!/usr/bin/env python
"""Produce the annotated showcase + tactical videos of what the pipeline does.

Runs perception + tagging once, caches the per-frame `FrameState`s and `Event`s, then
renders a set of videos from that single cache (see src/visualization/annotate.py):

  showcase.mp4                broadcast overlay: tracks, ball, HUD, event log, radar inset
  triangulation_overlay.mp4   Delaunay team-shape mesh on the real video
  voronoi_overlay.mp4         Voronoi control regions warped onto the real field
  radar_voronoi.mp4           top-down 2D map — Voronoi only
  radar_triangulation.mp4     top-down 2D map — Delaunay only
  radar_combined.mp4          top-down 2D map — Voronoi + Delaunay

Sampling is set in seconds (`--interval-s`), derived to a frame stride from the clip
fps, so "every tenth of a second" is fps-independent.

Examples:
  # full run + all videos, sampling every 0.1 s of footage
  python scripts/render_video.py --source data/raw_matches/2e57b9_0.mp4 \
      --device cuda --interval-s 0.1 --max-frames 250 --output-dir outputs/showcase

  # re-render the videos from the cache only (no models loaded)
  python scripts/render_video.py --source data/raw_matches/2e57b9_0.mp4 \
      --output-dir outputs/showcase --from-cache
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.video import get_video_info
from src.pipeline import run
from src.visualization.annotate import densify_frame_states, render_radar_video, render_video

ALL_OUTPUTS = ("showcase", "triangulation", "voronoi",
               "radar_voronoi", "radar_delaunay", "radar_combined")


def main() -> None:
    ap = argparse.ArgumentParser(description="render the showcase + tactical videos")
    ap.add_argument("--source", required=True, help="path to match video")
    ap.add_argument("--output-dir", default="outputs/showcase")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--interval-s", type=float, default=None,
                    help="sample every N seconds of footage (overrides --stride via fps)")
    ap.add_argument("--max-frames", type=int, default=None, help="cap processed frames")
    ap.add_argument("--device", default=None, help="override device (mps|cpu|cuda)")
    ap.add_argument("--ball-inference", choices=("sliced", "full_frame"), default=None,
                    help="dedicated-ball inference mode (default: configured sliced mode)")
    ap.add_argument("--ball-imgsz", type=int, default=None,
                    help="ball-model input size for full-frame inference")
    ap.add_argument("--player-imgsz", type=int, default=None,
                    help="override player-model input size")
    ap.add_argument("--classify-every", type=int, default=1,
                    help="run visual team classification every N processed frames")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="batch local model inference while preserving sequential tracking")
    ap.add_argument("--ball-every", type=int, default=1,
                    help="run ball detection every N processed frames")
    ap.add_argument("--pitch-every", type=int, default=1,
                    help="refresh pitch calibration every N processed frames")
    ap.add_argument("--warmup-every", type=int, default=30,
                    help="sample every Nth source frame for team-cluster fitting")
    ap.add_argument("--warmup-cap", type=int, default=600,
                    help="maximum player crops used to fit team clusters")
    ap.add_argument("--from-cache", action="store_true",
                    help="skip perception; load frame_states/events from the cache")
    ap.add_argument("--dense-output", action="store_true",
                    help="render at source fps by interpolating tracked display positions")
    ap.add_argument("--only", default=None,
                    help=f"comma list of outputs to render (default all): {','.join(ALL_OUTPUTS)}")
    ap.add_argument("--calib-gate", type=float, default=0.5,
                    help="gate tactical overlays below this calibration confidence")
    ap.add_argument("--radar-scale", type=float, default=0.06, help="inset radar render scale")
    ap.add_argument("--radar-panel-scale", type=float, default=0.10,
                    help="standalone radar-video render scale (higher = crisper)")
    ap.add_argument("--voronoi-opacity", type=float, default=0.35,
                    help="opacity of the Voronoi overlay warped onto the video")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "frame_states.pkl"
    want = set(ALL_OUTPUTS if args.only is None else args.only.split(","))

    info = get_video_info(args.source)
    if args.interval_s:
        stride = max(1, round(info.fps * args.interval_s))
        print(f"[sample] every {args.interval_s}s @ {info.fps:.1f}fps -> stride {stride} "
              f"({stride / info.fps:.3f}s)")
    else:
        stride = args.stride
    out_fps = max(1.0, info.fps / stride)

    if args.from_cache:
        if not cache.exists():
            raise SystemExit(f"no cache at {cache}; run once without --from-cache first")
        with open(cache, "rb") as f:
            blob = pickle.load(f)
        frame_states, events = blob["frame_states"], blob["events"]
        print(f"[cache] loaded {len(frame_states)} frame states, {len(events)} events")
    else:
        result = run(source=args.source, output_dir=args.output_dir,
                     stride=stride, max_frames=args.max_frames, device=args.device,
                     ball_inference=args.ball_inference, ball_imgsz=args.ball_imgsz,
                     player_imgsz=args.player_imgsz, classify_every=args.classify_every,
                     batch_size=args.batch_size, warmup_every=args.warmup_every,
                     warmup_cap=args.warmup_cap, ball_every=args.ball_every,
                     pitch_every=args.pitch_every)
        frame_states, events = result.frame_states, result.events
        with open(cache, "wb") as f:
            pickle.dump({"frame_states": frame_states, "events": events}, f)
        print(f"[cache] wrote {cache}")

    render_states = frame_states
    render_stride = stride
    render_max_frames = args.max_frames
    render_fps = out_fps
    if args.dense_output:
        dense_count = info.total_frames if args.max_frames is None else min(
            info.total_frames, max(1, (args.max_frames - 1) * stride + 1)
        )
        render_states = densify_frame_states(frame_states, dense_count, info.fps)
        render_stride = 1
        render_max_frames = dense_count
        render_fps = info.fps
        print(f"[display] interpolated {len(frame_states)} observed states -> "
              f"{len(render_states)} frames @ {render_fps:.1f} fps")

    common = dict(source=args.source, frame_states=render_states, events=events,
                  stride=render_stride, max_frames=render_max_frames,
                  calib_gate=args.calib_gate)
    written = []

    if "showcase" in want:
        written.append(render_video(
            out_path=str(out_dir / "showcase.mp4"), radar_scale=args.radar_scale,
            desc="showcase", **common))
    if "triangulation" in want:
        written.append(render_video(
            out_path=str(out_dir / "triangulation_overlay.mp4"),
            draw_radar=False, draw_events=False, draw_triangulation=True,
            desc="triangulation", **common))
    if "voronoi" in want:
        written.append(render_video(
            out_path=str(out_dir / "voronoi_overlay.mp4"),
            draw_radar=False, draw_events=False, draw_voronoi_overlay=True,
            voronoi_overlay_opacity=args.voronoi_opacity, desc="voronoi", **common))

    # standalone top-down maps (no source decode needed)
    radar_common = dict(frame_states=render_states, fps=render_fps,
                        scale=args.radar_panel_scale)
    if "radar_voronoi" in want:
        written.append(render_radar_video(
            out_path=str(out_dir / "radar_voronoi.mp4"),
            draw_voronoi=True, draw_delaunay=False, title="VORONOI", **radar_common))
    if "radar_delaunay" in want:
        written.append(render_radar_video(
            out_path=str(out_dir / "radar_triangulation.mp4"),
            draw_voronoi=False, draw_delaunay=True, title="DELAUNAY", **radar_common))
    if "radar_combined" in want:
        written.append(render_radar_video(
            out_path=str(out_dir / "radar_combined.mp4"),
            draw_voronoi=True, draw_delaunay=True, title="VORONOI + DELAUNAY", **radar_common))

    print("\n[done] wrote:")
    for p in written:
        print("  ", p)


if __name__ == "__main__":
    main()
