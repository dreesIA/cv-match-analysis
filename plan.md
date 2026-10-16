# Player Tracking and Assisted Tagging: Computer Vision Action Plan

- **Project:** updt.pro soccer analytics platform computer vision task
- **Scope:** Phase 1 to Phase 2 of the platform roadmap (Perception Foundation, Possession and Passing), with a narrow assisted-tagging slice
- **Core principle:** Nothing ships until it is validated against a held-out, condition-diverse "golden set." Speed of feature delivery is secondary to model trustworthiness, since every downstream stat, clip, and report inherits the errors of the perception layer.

---

## 1. Goal and Definition of Completion ("Done")

The objective of this phase is not "a working demo." It is a perception and tracking pipeline whose outputs (player/ball/referee detections, track IDs, team assignments, pitch coordinates) are accurate enough, and honest enough about their own uncertainty, that the analyst review layer described in the product spec can function as designed.

A component is "done" when:

1. It has a documented operational definition (what counts as a correct detection, a correct track, a correct team label).
2. It has been measured on a held-out validation set that was never used for training or threshold tuning.
3. Its confidence scores are calibrated, meaning a 0.9 confidence prediction is right roughly 90 percent of the time, not just "usually higher than a 0.5 confidence prediction."
4. Its known failure modes are documented and either mitigated or surfaced to the analyst as a flagged segment.

---

## 2. Architecture at a Glance

```
Match video
  -> Ingestion and normalization (resolution, frame rate, period split)
  -> Object detection (player, goalkeeper, referee, ball)             [Roboflow trained model]
  -> Multi-object tracking (consistent track IDs across frames)       [Supervision + ByteTrack/trackers]
  -> Team and role classification (analyzed team, opponent, GK, ref)  [SigLIP embeddings + clustering]
  -> Pitch keypoint detection and homography                          [Roboflow pitch model + OpenCV]
  -> Coordinate projection (pixel -> normalized pitch coordinates)
  -> Possession state machine and pass candidate generation
  -> Confidence aggregation and review-queue routing
  -> Analyst review (accept/reject/rectify)
  -> Structured event data + clips
```

Tooling choices:

- **Roboflow** for dataset management, annotation, versioning, and training (YOLO family and RF-DETR checkpoints, with the option to start from existing Universe models such as the football player/ball/goalkeeper/referee detector and the pitch keypoint detector).
- **Supervision** for tracking, zone logic, annotation overlays, and dataset evaluation utilities (confusion matrices, mAP).
- **OpenCV** for homography computation from keypoint correspondences.
- A lightweight **embedding plus clustering** approach (SigLIP or similar vision-language embeddings, dimensionality reduction, KMeans) for team classification, following the pattern Roboflow has already open sourced for soccer.

---

## 3. Phase 0: Baseline and Infrastructure (Weeks 1 to 2)

<u>Goal:</u> Get an end-to-end pipeline running on real updt.pro footage using existing open-source models, before training anything custom. This establishes a baseline and, more importantly, surfaces what is actually hard about updt.pro's specific footage (camera angles, broadcast quality, pitch types, kit colors).

Tasks:

1. Collect 3 to 5 representative full or partial matches from updt.pro, covering different venues, camera setups, and lighting conditions if possible. This becomes the seed for the golden set (Section 8).
2. Stand up a Roboflow workspace and project structure (see Section 11).
3. Run an existing pretrained football detection model (player, goalkeeper, referee, ball classes) and an existing pitch keypoint model on the seed clips, using Roboflow Inference.
4. Run Supervision's ByteTrack on the raw detections to get an initial sense of track stability and ID switch frequency.
5. Manually review 200 to 500 frames across the seed clips and log every failure: missed players, missed ball, wrong class, broken homography, lost tracks. Categorize by cause (occlusion, motion blur, camera cut, small object, lighting, kit similarity).
6. Write up a baseline report: per-component qualitative pass/fail and a prioritized list of what custom training needs to fix first.

<u>Output of this phase:</u> A short internal document (not the model itself) that tells us whether the bottleneck is detection, tracking, team classification, or calibration for updt.pro's footage specifically. This determines where Phase 2 effort goes first.

---

## 4. Phase 1: Data Foundation

This is the section that determines whether the rest of the plan succeeds. A good model trained on the wrong data distribution will fail silently in production, which is worse than failing loudly.

### 4.1 Footage diversity requirements

For both training and validation data, deliberately sample across:

- **Camera type:** Tactical/elevated cam versus broadcast-style if both exist in updt.pro's pipeline.
- **Venue and pitch surface** (grass color, line visibility, markings).
- **Lighting:** daytime, floodlit, overcast, mixed shadow.
- **Kit combinations,** including same-color or similar-color kit clashes between team and goalkeeper or referee.
- **Video quality tiers:** At minimum a "good" tier and a "degraded" tier (lower resolution, compression artifacts), since the product spec explicitly requires the system to disclose reduced confidence on degraded footage rather than pretend it is fine.

### 4.2 Annotation scheme

Define this once, in writing, before annotating anything:

- **Classes:** player (analyzed team), player (opponent), goalkeeper (analyzed team), goalkeeper (opponent), referee, assistant referee (if in scope), ball.
- **Bounding box convention** for partially occluded players (annotate the visible extent, do not guess the occluded portion).
- **Ball annotation rules** for partial visibility, motion blur, and the ball in flight versus on the ground (a single class is fine for detection, but log these cases for later analysis).
- **Pitch keypoints:** Use a fixed, numbered keypoint schema (line intersections, penalty box corners, center circle points, goalpost bases) so every annotator places point 17 in the same physical location every time.
- **Team identity is a separate annotation pass from detection.** Detection gives us "player"; team classification needs a per-track or per-crop "Team A / Team B / GK A / GK B / Referee" label for evaluation.

### 4.3 Train, validation, and test split strategy

The single most important rule: **split by match, not by frame or clip.**

If frames from the same match appear in both training and validation, the validation metrics will be inflated by the model memorizing that match's specific kits, pitch markings, and camera position. This will not be caught until a genuinely new match performs much worse than validation suggested, at which point trust in the system is already damaged.

Recommended split for an initial dataset of, say, 10 to 15 matches:

- 60 to 70 percent of matches for training
- 15 to 20 percent for validation (used for hyperparameter and threshold tuning)
- 15 to 20 percent for test, including at least one match from a venue, lighting condition, or kit combination not represented in training, used only for final reporting

### 4.4 Active learning loop

Rather than annotating everything from scratch:

1. Run the Phase 0 baseline (or first custom-trained) model on new, unlabeled footage.
2. Use Roboflow's annotation tooling to review and correct the model's predictions rather than drawing boxes from zero. This is dramatically faster for high-recall classes like "player."
3. Prioritize human attention on low-confidence predictions and on the rare classes (ball, goalkeeper, referee), which benefit most from additional examples.
4. Periodically retrain and re-measure on the fixed validation set so we can see whether new data is actually moving the needle, not just growing the dataset.

---

## 5. Phase 2: Detection Models

### 5.1 Starting point

Start from an existing open-source soccer detection checkpoint (player, goalkeeper, referee, ball classes) available on Roboflow Universe, available in YOLOv8/v9/v11 and RF-DETR variants. Fine-tune this on updt.pro's own annotated footage rather than training from a generic COCO checkpoint. This gives a meaningful head start on the player and referee classes, while the ball and goalkeeper classes will likely need the most updt.pro-specific data because they are rarer and more visually variable.

### 5.2 Architecture choice: YOLO family versus RF-DETR

Run both on the validation set before committing:

- **YOLOv8/v11**: mature, fast, well integrated with Supervision, large amount of community tooling for sports use cases. Generally a safe default for real-time or near-real-time inference.
- **RF-DETR**: transformer-based, can offer better accuracy on small and crowded objects in some benchmarks, at higher compute cost.

Given that this pipeline does not need to run live (it processes uploaded match video), inference speed is less critical than accuracy. If RF-DETR gives a meaningful precision and recall improvement on the ball and goalkeeper classes in the validation set, the extra compute cost is likely worth it. Decide based on our own measured numbers, not on general benchmarks, since soccer broadcast and tactical-cam footage has its own characteristics.

### 5.3 The ball problem

The ball is small, fast, frequently occluded, and the single highest-leverage object for almost every downstream tag (passes, shots, crosses, possession). Treat it as a separate engineering problem from player detection:

- Consider training at higher input resolution, or using tiled/sliced inference, specifically to improve small-object recall.
- Track ball detection recall and precision as a separate, headline metric, not buried inside an overall mAP.
- Build in explicit "no valid ball track" states. The product spec is correct that the system must preserve uncertainty rather than hallucinate a ball location, since a fabricated ball position silently corrupts every possession and pass calculation downstream.

### 5.4 Detection validation metrics

For each class, on the held-out test set:

- mAP at IoU 0.5 and mAP at IoU 0.5 to 0.95
- Precision and recall at the confidence threshold we actually plan to deploy (not just at the threshold that maximizes F1 in the abstract)
- Per-class breakdown, since an overall mAP of 0.85 can hide a ball mAP of 0.4
- Confusion analysis between visually similar classes (for example, goalkeeper versus outfield player when both teams wear similar colors, or referee versus a player in a similarly colored kit)

---

## 6. Phase 3: Multi-Object Tracking

### 6.1 Approach

Use Supervision's ByteTrack implementation on top of the detection model's outputs to assign persistent track IDs across frames.

One implementation note worth flagging now: Supervision is in the process of moving tracker implementations (including ByteTrack) into a separate `trackers` package, with the in-library `sv.ByteTrack` scheduled for eventual removal. Pin our Supervision version explicitly in the project's dependency file, and treat a tracker upgrade as a deliberate, tested change rather than something that happens via a routine `pip install --upgrade`, since a tracker behavior change would silently shift every downstream track-based metric.

### 6.2 Identity persistence across occlusion

ByteTrack handles short-term occlusion reasonably well via motion prediction, but soccer has frequent, longer occlusions (players bunching for set pieces, goalmouth scrambles). Two complementary mitigations:

1. Re-identification using the same appearance embeddings used for team classification (Section 7). When a track is lost and a new track appears nearby shortly after, compare appearance embeddings to decide whether it is plausibly the same player.
2. For the analyzed team specifically, use shirt-number recognition and squad constraints (only 11, plus substitutes, possible identities) as a strong prior, exactly as the product spec describes. This is a higher-effort item and can come after the core tracking and possession pipeline is stable.

### 6.3 Tracking validation metrics

This is the area most teams under-invest in, and it is where "the model looks fine in a demo clip" diverges most sharply from "the model produces trustworthy statistics over 90 minutes."

- **ID switches per match**: how often a single physical player's track ID changes.
- **Track fragmentation**: how often a continuous physical presence is split into multiple track segments.
- **MOTA (Multiple Object Tracking Accuracy)** and **IDF1**: standard tracking metrics that combine detection and identity-consistency errors.
- **HOTA (Higher Order Tracking Accuracy)**: a more balanced metric that separately credits detection accuracy and association accuracy, useful for diagnosing whether tracking failures stem from detection misses versus identity confusion.

To compute these, we need ground-truth tracks for at least a subset of the golden set: a small number of clips (a few minutes each, covering a goalmouth scramble, a set piece, and open play) with every player manually assigned a consistent ID across the clip. This is labor-intensive but high value, since it is the only way to know whether our possession and pass logic is being fed reliable identity information.

---

## 7. Phase 4: Team and Role Classification

### 7.1 Approach

Follow the pattern that has already proven workable for soccer: extract an appearance embedding (for example, using a vision-language model such as SigLIP) for each detected player crop, reduce dimensionality (UMAP or similar), and cluster (KMeans) into team groups. Goalkeepers and referees are typically handled separately, since their kits are usually visually distinct from both outfield teams and a two-cluster KMeans over all players would otherwise misclassify them.

### 7.2 Required improvements over the naive baseline

A single-frame color cluster is not reliable in production because of lighting changes, shadows, and partial occlusion. Build in:

- **Temporal smoothing**: classify each detection independently, but assign the final team label per track as the majority vote (or confidence-weighted vote) across all detections in that track. A single bad frame should not flip a player's team for the whole match.
- **Goalkeeper and referee handling**: identify these roles using a combination of detection class (if our detector outputs separate GK/referee classes), pitch position priors (goalkeepers spend most of their time near their own goal), and kit color, rather than relying on the same two-cluster logic used for outfield players.
- **Kit clash handling**: when team and goalkeeper kits, or team and referee kits, are similar, fall back to secondary cues (shorts color, position on pitch, the analyst-confirmed setup information from Step 2 of the workflow). This is exactly the kind of case the product spec routes to analyst confirmation, so the model's job is to flag low-confidence cases rather than guess.

### 7.3 Validation

- Per-track team classification accuracy against analyst-labeled ground truth, reported separately for outfield players, goalkeepers, and referees.
- A confusion matrix specifically for matches with similar-colored kits, since this is the known hard case and aggregate accuracy will hide it.
- **Stability metric:** How often a track's team label changes over its lifetime (should be at or near zero after temporal smoothing).

---

## 8. Phase 5: Pitch Calibration and Coordinate Mapping

### 8.1 Approach

Use a pitch keypoint detection model (an existing open-source soccer pitch keypoint model is a reasonable starting point, fine-tuned on updt.pro footage if needed) to detect a fixed set of pitch landmarks per frame, then compute a homography matrix mapping image coordinates to normalized pitch coordinates using OpenCV.

Critically, follow the product spec's guidance: estimate homography per stable camera segment, not once for the whole video. Detect camera cuts or significant pans/zooms, and recompute calibration after each.

### 8.2 Validation

- **Reprojection error in meters**: for a held-out set of frames with manually annotated keypoints, project them through the computed homography and measure the distance, in real-world pitch units, between the projected and known true positions. This is the single most informative calibration metric, since "the lines look roughly right" in a visualization can hide errors of several meters that materially affect zone-entry and pass-distance calculations.
- **Calibration confidence per segment**: a segment with few visible keypoints or a poor homography fit should be explicitly flagged as low-confidence, propagating that uncertainty to any spatial statistic computed from it (as required by the false-precision principle in the product spec).
- **Segment coverage**: percentage of match duration with valid, high-confidence calibration. Gaps should be visible to the analyst, not silently interpolated.

---

## 9. Phase 6: From Tracks to Tags (Assisted Tagging MVP)

Once detection, tracking, team classification, and calibration are each independently validated, combine them into the smallest useful set of tags, matching the MVP scope in the product spec:

1. **Possession**: which team's player has controlled the ball, based on ball-player proximity, ball velocity relative to the nearest player, and continuity over time.
2. **Possession changes**: transitions where the controlling team changes, with a minimum-evidence requirement (sustained control, not a single deflection) to avoid noisy flapping.
3. **Passes**: a controlled touch by Player A, ball separation toward a teammate, and subsequent control by Player B, with length and direction labels derived from the calibrated pitch coordinates.
4. **Carries**: sustained player-ball proximity with shared movement direction and minimum displacement.
5. **Attacking-third entries and halfway-line progressions**: derived directly from calibrated coordinates plus possession state, so these inherit whatever calibration confidence applies to that segment.

Each of these should carry forward the component-level confidence scores (detection confidence, tracking confidence, team classification confidence, calibration confidence) into a single per-event confidence, so the review-queue prioritization described in the product spec (Section 8.1) can function.

---

## 10. Validation and Reliability Framework

This section is the backbone of the project. Every phase above produces a model or pipeline stage; this section is what tells us whether to trust it.

### 10.1 The golden set

Maintain a small, fixed, fully manually annotated set of clips that is never used for training and is only updated deliberately (with version control). It should include:

- At least one full match, fully annotated for detection, tracking, team labels, and pitch keypoints.
- Several shorter clips (2 to 5 minutes each) chosen specifically to cover hard cases: goalmouth scrambles, set pieces, similar-colored kits, degraded video quality, camera cuts.
- At least one match from conditions not seen in training (different venue, different broadcast setup), to catch overfitting to our training distribution.

Every model change, no matter how small, gets re-evaluated against the golden set before being considered for deployment. Treat this as a regression test suite for models, the same way we would treat unit tests for code.

### 10.2 Component-level validation matrix

| Component | Primary metric(s) | Secondary metric(s) | Validation data |
|---|---|---|---|
| Player/GK/referee detection | mAP@0.5, mAP@0.5:0.95, per-class precision/recall | Confusion between similar-kit classes | Held-out matches, golden set |
| Ball detection | Recall and precision at deployment threshold | Recall under occlusion and motion blur subsets | Golden set hard-case clips |
| Tracking | MOTA, IDF1, HOTA | ID switches per match, fragmentation rate | Manually tracked golden set clips |
| Team classification | Per-track accuracy | Stability (label changes per track), kit-clash subset accuracy | Analyst-labeled tracks |
| Pitch calibration | Reprojection error (meters) | Segment coverage percentage, confidence distribution | Manually annotated keypoints |
| Possession/pass logic | Agreement rate with analyst ground truth | Precision/recall on possession changes specifically | Analyst-reviewed event log |

### 10.3 Confidence calibration

Raw model confidence scores are not automatically meaningful probabilities. Before using them to drive the review-queue thresholds in Section 12.2 of the product spec (high confidence auto-accepted, medium confidence queued, low confidence hidden), check calibration:

- Bin predictions by confidence score and compare the bin's average confidence to its actual accuracy on the validation set.
- If the model is overconfident or underconfident, apply a calibration adjustment (such as temperature scaling) before using raw scores for thresholding.
- Set thresholds per event type, not globally. A 0.7 confidence pass and a 0.7 confidence shot do not carry the same reliability, since the underlying detection logic and base rates differ.

### 10.4 Continuous evaluation

As the active learning loop (Section 4.4) brings in new annotated data and the models are retrained:

1. Every retrained model is evaluated on the unchanged golden set and the unchanged held-out test matches, producing a comparable metric trail over time.
2. A model is only promoted to production if it improves or maintains performance on every component metric in Section 10.2, not just an aggregate score. A model that improves overall mAP by overfitting further to common, easy cases while regressing on the ball or on similar-kit team classification should not ship.
3. Keep a simple model registry (even a spreadsheet or a Roboflow project's version history is enough initially) recording which model version is deployed, its golden-set scores, and the date.

### 10.5 Human-in-the-loop ground truth

Analyst accept/reject/rectify decisions (Section 8.3 of the product spec) are not just a UI feature, they are the project's most valuable ongoing data source. Set up the data pipeline from day one so that every analyst decision is captured in a structured form (machine prediction, machine confidence, final analyst decision) and periodically reviewed for:

- Recurring rectification patterns (if the model consistently mislabels a specific situation as cross potential when it is actually a long pass, that is a targeted retraining signal).
- Whether analyst review time per match is trending down over successive model versions, which is ultimately the metric that matters to updt.pro's customers.

### 10.6 Error taxonomy and review cadence

Maintain a living document of known failure modes, each tagged with: affected component, example clip reference, current mitigation (model fix, threshold adjustment, or "flagged for analyst review by design"), and status. Review this list at the end of each phase before moving on, since a failure mode that is "flagged for analyst review" at MVP scale may need an actual model fix once match volume increases.

---

## 11. Risk Register

| Risk (from product spec Section 16) | Primary impact | Mitigation | Validation tie-in |
|---|---|---|---|
| Low-resolution, compressed, or poorly lit footage | All detection and tracking degrades | Explicit "degraded" tier in training and validation data; confidence scores reflect footage quality | Stress-test subset of golden set |
| Ball occlusion and small size | Possession, pass, shot, cross logic all depend on ball state | Dedicated ball metrics, "no valid ball track" state, possible higher-resolution or tiled inference | Ball-specific recall/precision tracked separately |
| Player identity drift (shirt numbers) | Player-level statistics misattributed | Track-level identity propagation, squad-number constraints, analyst correction loop | Track stability metrics, identity coverage percentage |
| Homography failure (camera motion, partial pitch view) | All spatial statistics and zone maps | Per-segment recalibration, explicit confidence and coverage reporting | Reprojection error on golden set |
| Subjective events (duels, take-ons, crosses) | Disagreement between model and analyst on borderline cases | Treat as potential tags by design, prioritize recall, rely on analyst adjudication | Agreement rate with analyst ground truth |
| False precision in dashboards | Erodes coach trust if stats look exact but are not | Surface confidence and coverage alongside every spatial or count statistic | Coverage and confidence reporting built into Section 10.2 outputs |

---

## 12. Tooling, Infrastructure, and Repo Layout

### 12.1 Roboflow workspace organization

- One project per detection task that has a distinct class set or annotation type: player/GK/referee/ball detection, pitch keypoints. Keep team-label annotation as a separate, lightweight labeling pass tied to track crops rather than a third detection project.
- Use Roboflow's dataset versioning so every trained model can be traced back to the exact dataset version it was trained on, which is essential for the model registry in Section 10.4.

### 12.2 Dependency management

- Pin Supervision, Ultralytics (or RF-DETR's library), and OpenCV versions explicitly. Given the in-progress migration of ByteTrack out of Supervision into a separate `trackers` package, decide deliberately whether to stay on a pinned older Supervision version for now or migrate early, and document the choice.
- Keep inference and training environments reproducible (a `requirements.txt` or `pyproject.toml` with pinned versions, plus a short setup README).

### 12.3 Potential repo structure

```
updt-cv/
  data/
    golden_set/              # never used for training, version-controlled separately
    raw_matches/
    annotations/
  configs/
    detection_*.yaml
    tracking_*.yaml
  src/
    ingestion/
    detection/
    tracking/
    team_classification/
    calibration/
    tagging/
    evaluation/
  notebooks/
    baseline_eval.ipynb
    error_analysis.ipynb
  reports/
    phase0_baseline.md
    model_registry.md
    error_taxonomy.md
```

### 12.4 Compute considerations

- **Training:** A single modern GPU is sufficient for fine-tuning YOLO-family or RF-DETR models on a dataset of this size; budget for iterative retraining as the active learning loop brings in new data.
- **Inference:** Since this is offline, batch processing of uploaded matches, _**throughput matters more than latency.**_ Estimate processing time per match early (Phase 0) so it can be communicated to coaches as part of the "submit and forget" experience.

---

## 13. Timeline and Milestones

| Weeks | Milestone | Go/No-Go Gate |
|---|---|---|
| 1 to 2 | Phase 0 baseline complete; golden set seed collected | Baseline failure modes documented; clear priority order for custom training established |
| 3 to 6 | Detection model fine-tuned (player/GK/referee/ball) and validated | Per-class precision/recall on held-out test matches meets agreed thresholds, especially for ball |
| 5 to 8 | Tracking pipeline integrated and validated (overlaps with detection work) | MOTA/IDF1/HOTA on manually tracked golden-set clips meets thresholds; ID switch rate documented |
| 6 to 9 | Team classification with temporal smoothing validated | Per-track accuracy meets threshold including kit-clash subset |
| 7 to 10 | Pitch calibration pipeline validated | Reprojection error within agreed meter tolerance; coverage percentage acceptable |
| 9 to 12 | Possession, pass, and carry logic integrated; confidence aggregation in place | Agreement rate with analyst ground truth on a held-out match meets threshold; review-queue routing demonstrably reduces analyst review time versus full manual tagging |

These ranges assume one focused engineer/data scientist with analyst availability for ground-truth annotation starting in week 1. Adjust based on actual annotation throughput, which is usually the real bottleneck.

---

## 14. Immediate Next Steps (First Two Weeks)

1. Get access to 3 to 5 representative full or partial match videos from updt.pro, prioritizing variety over quantity.
2. Stand up the Roboflow workspace and the repo skeleton in Section 12.3.
3. Run an existing open-source soccer detection model and pitch keypoint model on the seed footage via Roboflow Inference, and run Supervision's ByteTrack on the resulting detections.
4. Manually review a sample of frames and clips from the seed footage, logging failure modes by category (occlusion, small object, lighting, kit similarity, camera cut, calibration drift).
5. Draft the annotation guideline document (Section 4.2) and the operational definitions referenced in Section 1, since both are needed before any meaningful annotation work begins.
6. Begin selecting and annotating the golden set (Section 10.1) in parallel, since it takes time and is needed before the first custom-trained model can be properly evaluated.
