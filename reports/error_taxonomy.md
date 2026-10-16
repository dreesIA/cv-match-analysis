# Error Taxonomy (living document — plan §10.6)

Every known failure mode, its affected component, an example clip reference, the
current mitigation, and status. Review at the end of each phase: a failure that is
"flagged for analyst review" at MVP scale may need a real model fix at higher volume.

| ID | Component | Failure mode | Example (clip @ frame) | Cause | Mitigation | Status |
|---|---|---|---|---|---|---|
| E-01 | Ball detection | Ball missed under occlusion / motion blur | | small object + occlusion | tiled inference; "no valid ball" state; ball metrics tracked separately | open |
| E-01b | Ball detection | **Ball not detected when far away / on the far side of the pitch** (observed on match1 ~20:00 clip, 1080p) | match1 @ 00:20:00 clip | ball only a few px at distance → below detector recall even with 640px tiles | finer InferenceSlicer tiles (e.g. 384px); "no valid ball" state correctly shown (no fabricated position) | open |
| E-02 | Calibration | Homography drifts after camera pan | | few visible keypoints | per-segment recalibration; flag low-confidence segment | open |
| E-03 | Team classification | Mislabel on similar-colour kits | | kit clash | temporal majority vote; route low-margin to analyst | open |
| E-03b | Team classification | **Two teams not separated at all** on real footage (observed on match1 ~20:00 clip) | match1 @ 00:20:00 clip | distant/low-res crops → weak appearance embeddings; cluster may split on lighting/pitch-region not kit | more/diverse warmup crops; verify kit colours are separable; route low-margin to analyst | open |
| E-04 | Tracking | ID switch in goalmouth scramble | | dense occlusion | (later) appearance re-id; squad-number prior | open |
| E-05 | Tagging | Possession flaps on deflection | | brief contact | min-evidence window + hysteresis | mitigated |

Status values: `open` | `mitigated` | `flagged-by-design` | `needs-model-fix` | `resolved`.
