# Error Taxonomy (living document — plan §10.6)

Every known failure mode, its affected component, an example clip reference, the
current mitigation, and status. Review at the end of each phase: a failure that is
"flagged for analyst review" at MVP scale may need a real model fix at higher volume.

| ID | Component | Failure mode | Example (clip @ frame) | Cause | Mitigation | Status |
|---|---|---|---|---|---|---|
| E-01 | Ball detection | Ball missed under occlusion / motion blur | | small object + occlusion | tiled inference; "no valid ball" state; ball metrics tracked separately | open |
| E-02 | Calibration | Homography drifts after camera pan | | few visible keypoints | per-segment recalibration; flag low-confidence segment | open |
| E-03 | Team classification | Mislabel on similar-colour kits | | kit clash | temporal majority vote; route low-margin to analyst | open |
| E-04 | Tracking | ID switch in goalmouth scramble | | dense occlusion | (later) appearance re-id; squad-number prior | open |
| E-05 | Tagging | Possession flaps on deflection | | brief contact | min-evidence window + hysteresis | mitigated |

Status values: `open` | `mitigated` | `flagged-by-design` | `needs-model-fix` | `resolved`.
