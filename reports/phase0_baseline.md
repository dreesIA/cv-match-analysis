# Phase 0 Baseline Report

> Fill this in after running the pipeline + evaluation harness on the golden set.
> This report is the **go/no-go gate** (plan §13): it decides whether the bottleneck
> is detection, tracking, team classification, or calibration on updt.pro footage,
> and therefore what (if anything) to fine-tune first. The MVP itself ships on
> pretrained models — training is only justified by evidence collected here.

## 1. Setup

- Models: `players.pt`, `ball.pt`, `pitch.pt` (Roboflow Universe, pretrained — see `data/models/`)
- Clips evaluated: _list golden-set clip ids_
- Date / commit: _..._
- Hardware: Apple M1 / 16 GB, `--device mps`
- Per-match processing time (throughput, not latency — plan §12.4): _... min / match_

## 2. Per-component results (from `src/evaluation/*`)

| Component | Metric | Result | Threshold | Pass? |
|---|---|---|---|---|
| Detection (overall) | mAP@0.5 / @0.5:0.95 | | | |
| Detection — **ball** | recall / precision @ deploy conf | | | |
| Detection — GK vs player | confusion (similar kits) | | | |
| Tracking | MOTA / IDF1 | | | |
| Tracking | id-switches / fragmentation per clip | | | |
| Team classification | per-track accuracy (outfield / GK / ref) | | | |
| Team classification | label changes per track (stability) | | | |
| Calibration | reprojection error (median / p90, metres) | | | |
| Calibration | coverage % of duration | | | |
| Tagging | event agreement vs analyst log | | | |
| Tagging | possession-change precision / recall | | | |
| Confidence | ECE before / after temperature | | | |

## 3. Qualitative failure modes observed

_Review 200–500 frames + the hard-case clips. Log each failure with cause:
occlusion / motion blur / camera cut / small object / lighting / kit similarity /
calibration drift. Cross-reference `reports/error_taxonomy.md`._

## 4. Prioritized fix list (what custom training must address first)

1. _e.g. ball recall under occlusion → highest leverage; candidate for tiled
   inference tuning or fine-tuning the ball model on updt.pro footage._
2. _..._

## 5. Decision

- [ ] Ship MVP on pretrained models for these conditions
- [ ] Fine-tune _<which model>_ on updt.pro footage (Roboflow cloud / Colab — not M1)
- [ ] Expand golden set to cover _<missing condition>_
