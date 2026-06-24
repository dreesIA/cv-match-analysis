# Golden Set (plan §10.1)

A small, fixed, fully manually annotated set of clips that is **never used for
training** and only changes deliberately (version-controlled). It is the regression
test suite for models: every model change is re-scored here before deployment.

## What to include
- At least one segment fully annotated for **detection, tracking, team labels, and
  pitch keypoints**, plus an analyst **event log**.
- Several 2–5 min hard-case clips: goalmouth scramble, set piece, **similar-coloured
  kits**, degraded/compressed video, camera cuts.
- At least one clip from conditions **not** in training (different venue/broadcast).

## How to build the first clip (seed from the sample footage)
1. `mkdir data/golden_set/gs_2e57b9_0` and copy a 1–2 min cut of a sample clip in as
   `video.mp4` (media is gitignored; keep it out-of-band).
2. Annotate in Roboflow (boxes + classes, pitch keypoints) and assign **consistent
   track ids** across the clip for tracking GT.
3. Export to the formats in [src/evaluation/golden_set.py](../../src/evaluation/golden_set.py):
   `detections.json`, `tracks_gt.txt` (MOT), `team_labels.json`, `keypoints.json`,
   `events_gt.json`, and a `manifest.yaml`.
4. Score: `python scripts/evaluate.py --clip gs_2e57b9_0 --pred-events outputs/run1/events.json`.

Annotation throughput is usually the real bottleneck (plan §13) — start small and
grow coverage via the active-learning loop.
