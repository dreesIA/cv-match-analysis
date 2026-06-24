"""Detection wrappers (plan §5).

Runs two detectors per frame:
  * players model  -> player / goalkeeper / referee boxes  (ball class dropped here)
  * dedicated ball model via tiled InferenceSlicer        -> high-recall small-object ball
plus the pitch-keypoint model for calibration.

Each model loads from a local ultralytics `.pt` if present, else falls back to the
Roboflow-hosted model via `inference.get_model`. Ball is first-class: when no ball
clears the deploy threshold we return None, never a fabricated position (plan §5.3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import supervision as sv


def _resolve_device(device: str) -> str:
    try:
        import torch

        if device == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
    except Exception:
        return "cpu"
    return device


class _Model:
    """Thin adapter over ultralytics (local .pt) or Roboflow inference (hosted)."""

    def __init__(self, spec: dict[str, Any], device: str):
        self.spec = spec
        self.device = device
        self.backend, self.model = self._load(spec, device)

    @staticmethod
    def _load(spec: dict[str, Any], device: str):
        import os

        weights = spec.get("local_weights")
        if weights and os.path.exists(weights):
            from ultralytics import YOLO

            return "ultralytics", YOLO(weights)
        # fallback: hosted Roboflow Universe model
        try:
            from inference import get_model

            api_key = os.environ.get("ROBOFLOW_API_KEY")
            return "inference", get_model(spec["roboflow_id"], api_key=api_key)
        except Exception as e:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"No local weights at {weights!r} and could not load hosted model "
                f"{spec.get('roboflow_id')!r}. Run scripts/download_assets.py or set "
                f"ROBOFLOW_API_KEY. Original error: {e}"
            )

    def detect(self, image: np.ndarray, conf: float, imgsz: int | None = None) -> sv.Detections:
        if self.backend == "ultralytics":
            kw: dict[str, Any] = {"conf": conf, "device": self.device, "verbose": False}
            if imgsz:
                kw["imgsz"] = imgsz
            result = self.model.predict(image, **kw)[0]
            return sv.Detections.from_ultralytics(result)
        result = self.model.infer(image, confidence=conf)[0]
        return sv.Detections.from_inference(result)

    def keypoints(self, image: np.ndarray, conf: float) -> sv.KeyPoints:
        if self.backend == "ultralytics":
            result = self.model.predict(image, conf=conf, device=self.device, verbose=False)[0]
            return sv.KeyPoints.from_ultralytics(result)
        result = self.model.infer(image, confidence=conf)[0]
        return sv.KeyPoints.from_inference(result)


@dataclass
class FrameDetections:
    players: sv.Detections                 # gk/player/referee (ball class removed)
    ball_xy: Optional[tuple[float, float]]  # best ball center in pixels, or None
    ball_conf: float
    keypoints: sv.KeyPoints                # pitch landmarks (may be empty)


class Detector:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.class_map = cfg["class_map"]
        self.deploy_conf = cfg["deploy_conf"]
        device = _resolve_device(cfg.get("device", "cpu"))
        self.device = device
        m = cfg["models"]
        self.players = _Model(m["players"], device)
        self.ball = _Model(m["ball"], device)
        self.pitch = _Model(m["pitch"], device)
        self._ball_slicer = self._make_ball_slicer(m["ball"])

    def _make_ball_slicer(self, ball_spec: dict[str, Any]) -> sv.InferenceSlicer:
        conf = ball_spec.get("conf", 0.3)
        slice_wh = tuple(ball_spec.get("slice_wh", (640, 640)))

        def callback(image_slice: np.ndarray) -> sv.Detections:
            return self.ball.detect(image_slice, conf=conf)

        return sv.InferenceSlicer(callback=callback, slice_wh=slice_wh)

    def players_only(self, image: np.ndarray) -> sv.Detections:
        """Players-model detections (ball class dropped). Used for the cheap team-
        classifier warmup pass so we don't run ball+pitch models during warmup."""
        pm = self.cfg["models"]["players"]
        det = self.players.detect(image, conf=pm["conf"], imgsz=pm.get("imgsz"))
        return self._filter_players(det)

    def detect(self, image: np.ndarray) -> FrameDetections:
        pm = self.cfg["models"]["players"]
        players = self.players.detect(image, conf=pm["conf"], imgsz=pm.get("imgsz"))
        players = self._filter_players(players)

        ball_xy, ball_conf = self._detect_ball(image)

        keypoints = self.pitch.keypoints(image, conf=self.cfg["models"]["pitch"]["conf"])
        return FrameDetections(players=players, ball_xy=ball_xy, ball_conf=ball_conf, keypoints=keypoints)

    def _filter_players(self, det: sv.Detections) -> sv.Detections:
        """Drop the ball class from the players model and apply per-class deploy conf."""
        if len(det) == 0:
            return det
        ball_id = self.class_map["ball"]
        keep = det.class_id != ball_id
        det = det[keep]
        # per-class confidence gating at deploy thresholds
        id_to_name = {v: k for k, v in self.class_map.items()}
        thr = np.array(
            [self.deploy_conf.get(id_to_name.get(int(c), ""), 0.0) for c in det.class_id]
        )
        return det[det.confidence >= thr]

    def _detect_ball(self, image: np.ndarray) -> tuple[Optional[tuple[float, float]], float]:
        nms = self.cfg["models"]["ball"].get("nms", 0.1)
        det = self._ball_slicer(image).with_nms(threshold=nms)
        if len(det) == 0:
            return None, 0.0
        best = int(np.argmax(det.confidence))
        conf = float(det.confidence[best])
        if conf < self.deploy_conf["ball"]:
            return None, conf  # below deploy threshold -> "no valid ball" upstream
        x1, y1, x2, y2 = det.xyxy[best]
        return (float((x1 + x2) / 2), float((y1 + y2) / 2)), conf
