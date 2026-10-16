# updt.pro CV MVP — Assisted Tagging Pipeline

Perception + assisted-tagging MVP for soccer match video, built on **Roboflow
Universe models + Supervision**. It ingests a clip and emits a structured **event
log** (possession, possession changes, passes, carries, zone entries) where every
event carries detection / tracking / team / calibration confidences aggregated into
one score and routed to **accept / review / hidden** for analyst review.

Runs entirely on **pretrained models** on an Apple **M1** (`--device mps`) — no
training. See [plan.md](plan.md) for the full program; this repo implements the
Phase 0 baseline through the Section 9 tagging slice.

## Pipeline

```
video -> ingestion (fps norm + camera-cut segments)
      -> detection (players + dedicated ball model, tiled)        [Roboflow Universe YOLOv8]
      -> tracking (ByteTrack)                                     [Supervision]
      -> team & role classification (SigLIP->UMAP->KMeans)        [roboflow/sports]
      -> pitch calibration (per-segment homography -> metres)     [Supervision + OpenCV]
      -> possession / pass / carry / zone tagging                 [this repo]
      -> confidence aggregation + review-queue routing            [this repo]
      -> events.json / events.parquet + run_summary.json
```

What's reused vs. built: the perception backbone (`TeamClassifier`,
`ViewTransformer`, `SoccerPitchConfiguration`, the model weights) comes from
[roboflow/sports](https://github.com/roboflow/sports); the **tagging logic,
confidence routing, and evaluation harness** are this repo's contribution.

## Setup (Apple M1, Python 3.12)

```bash
brew install ffmpeg                              # reliable video I/O (required)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-cv.txt               # exact pins in requirements.lock.txt
python scripts/download_assets.py                # weights + one DFL sample clip (no API key)
```

> `sv.ByteTrack` is deprecated in supervision 0.28 and **removed in 0.30** (logic
> moves to the `trackers` package). We cap `supervision<0.30`; the tracking wrapper
> is adaptive. Treat any tracker upgrade as a deliberate, tested change.

## Run

```bash
# first run also downloads SigLIP (~1.5 GB) for team classification
PYTORCH_ENABLE_MPS_FALLBACK=1 python run.py \
    --source data/raw_matches/2e57b9_0.mp4 --output-dir outputs/run1

# quick wiring check on 60 frames:
python run.py --source data/raw_matches/2e57b9_0.mp4 --max-frames 60 --output-dir outputs/smoke
```

Outputs: `outputs/<run>/events.json` (+ `.parquet`) and `run_summary.json`
(event counts by type/status, ball-valid fraction, calibration coverage).

## Interactive Bundesliga showcase

The Streamlit showcase is limited to two quality-gated, full-rate DFL/Bundesliga
clips and keeps each tracking video synchronized with its paired tactical-map
render and cumulative pass, carry, and possession-gain views. Passes on the
action-location pitch use the same blue/red team colors as the CV overlays.

Run:

```bash
streamlit run streamlit_app.py
```

The compact browser-safe M4V assets under `data/showcase/bundesliga/` are kept
with the deployable app. Raw footage, MP4 intermediates, model weights, and
per-frame inference caches remain ignored.

The checked-in clips run player detection and ByteTrack on every source frame
(25 FPS). The small-object ball model and pitch calibration run every fifth
frame; ball positions are interpolated only between two valid measurements,
retain an explicit `observed=false` flag, and receive a confidence penalty.
Interpolation is disabled across camera cuts. Event tagging uses these same
frame states—there is no separate hand-authored showcase path.

```bash
python scripts/render_video.py --source <bundesliga.mp4> \
    --output-dir <bundle-dir> --stride 1 --device mps --batch-size 8 \
    --ball-inference full_frame --ball-imgsz 1536 --ball-every 5 \
    --pitch-every 5 --player-imgsz 640 --classify-every 10 \
    --only showcase,radar_combined
```

## Validate (the point of the project — plan §10)

The tagging core has synthetic tests (no models needed):

```bash
python tests/test_tagging.py
```

Against a **golden set** (`data/golden_set/<clip>/`, never trained on — format in
[src/evaluation/golden_set.py](src/evaluation/golden_set.py)), the harness scores
every stage:

| Component | Module | Metric |
|---|---|---|
| Detection | [evaluation/detection.py](src/evaluation/detection.py) | mAP@0.5/.5:.95, per-class P/R, **ball separate** |
| Tracking | [evaluation/tracking.py](src/evaluation/tracking.py) | MOTA/IDF1, id-switches (HOTA via TrackEval) |
| Calibration | [evaluation/calibration.py](src/evaluation/calibration.py) | **reprojection error in metres**, coverage |
| Team | [evaluation/team.py](src/evaluation/team.py) | per-track accuracy (by role), label stability |
| Tagging | [evaluation/tagging.py](src/evaluation/tagging.py) | event agreement, possession-change P/R |
| Confidence | [evaluation/confidence_calibration.py](src/evaluation/confidence_calibration.py) | reliability curve, ECE, temperature |

## Honesty guarantees (plan §5.3, §8.2)

- **No valid ball** (below deploy conf) → possession `unknown`, **no events emitted**.
- **Low-confidence calibration segment** → flagged; its zone events inherit that.
- Every event is routed, not asserted: weak components push it to review/hidden.

## Config

All thresholds live in [configs/](configs/) — `detection.yaml` (model ids, deploy
conf), `tracking.yaml` (ByteTrack), `tagging.yaml` (proximity/velocity/hysteresis,
**per-event-type** routing thresholds). Tune routing thresholds from the calibrated
reliability curves, not by guessing.

## Layout

`src/{ingestion,detection,tracking,team_classification,calibration,tagging,evaluation}`,
`configs/`, `data/{golden_set,raw_matches,models}`, `reports/`, `tests/`.
See [plan.md](plan.md) §12.3.
