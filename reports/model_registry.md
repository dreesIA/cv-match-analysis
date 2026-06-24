# Model Registry (plan §10.4)

Every model version that is evaluated or deployed, traceable to its dataset version
and its golden-set scores. A model is only promoted if it improves or maintains
**every** component metric (§10.2) — not just an aggregate — so we never ship a model
that improves overall mAP by regressing on the ball or on similar-kit team labels.

| Version | Date | Model(s) | Dataset version | Golden-set scores (ball R / mAP50 / MOTA / reproj_m / event F1) | Deployed? | Notes |
|---|---|---|---|---|---|---|
| v0-pretrained | 2026-06-24 | roboflow/sports `players.pt`, `ball.pt`, `pitch.pt` | Roboflow Universe (pretrained) | _baseline — fill from Phase 0_ | candidate | MVP baseline; no fine-tuning |

Process: download/train → evaluate on the **unchanged** golden set + held-out test
matches → record the row above → promote only on no-regression.
