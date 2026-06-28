"""
Hybrid Re-ID assignment: IoU (short-term) + L2-normalized embedding cosine (longer-term).

Why pure IoU fails: fast motion, brief occlusion, or jitter can make consecutive
boxes have IoU ≈ 0, so the tracker mints a new ID every time association fails.

Why embeddings alone fail without smoothing: single-frame features are noisy;
always compare to an EMA centroid and use a conservative cosine threshold.

Optional: set VIGILRAIL_OSNET_ONNX to your OSNet ONNX path and install onnxruntime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

# Tunables (env override)
def _f(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


# Min IoU to associate with the same track as previous frame (lower = tolerate faster motion)
REID_IOU_THRESHOLD = _f("VIGILRAIL_REID_IOU", 0.18)
# Min cosine similarity between L2-normalized embeddings and track centroid
REID_COSINE_THRESHOLD = _f("VIGILRAIL_REID_COSINE", 0.68)
# EMA for centroid: centroid = α*centroid + (1-α)*emb, then L2 re-normalize
REID_EMA_ALPHA = _f("VIGILRAIL_REID_EMA", 0.88)
# If ONNX missing: max normalized center distance (fraction of frame diagonal) to match
REID_CENTER_DIST_MAX = _f("VIGILRAIL_REID_CENTER", 0.045)

# Dict-memory tracker (cosine + pixel center distance, one-to-one)
REID_MEM_SIM = _f("VIGILRAIL_REID_MEM_SIM", 0.5)
REID_MEM_DIST_PX = _f("VIGILRAIL_REID_MEM_DIST", 100.0)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x.reshape(-1), ord=2)
    if n < eps:
        return x * 0.0
    return (x / n).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine sim of two L2-normalized vectors = dot product."""
    return float(np.dot(a.reshape(-1), b.reshape(-1)))


def iou_xyxy(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ar + br - inter + 1e-6)


def center_distance_norm(a, b, frame_h: int, frame_w: int) -> float:
    cx1 = (a[0] + a[2]) * 0.5
    cy1 = (a[1] + a[3]) * 0.5
    cx2 = (b[0] + b[2]) * 0.5
    cy2 = (b[1] + b[3]) * 0.5
    diag = (frame_h**2 + frame_w**2) ** 0.5 + 1e-6
    return float((((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5) / diag)


def crop_person_bgr(frame: np.ndarray, xyxy: tuple[int, int, int, int], pad: float = 0.1):
    x1, y1, x2, y2 = xyxy
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def preprocess_osnet_imagenet(
    crop_bgr: np.ndarray, input_w: int = 128, input_h: int = 256
) -> np.ndarray:
    """NCHW float32, ImageNet norm. OSNet often uses 128x256 (W x H)."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    return np.expand_dims(x, axis=0)


class OSNetOnnxExtractor:
    """Runs OSNet (or compatible) ONNX and returns L2-normalized feature vector."""

    def __init__(self, onnx_path: str) -> None:
        import onnxruntime as ort

        self._path = onnx_path
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(
            onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._inp = self._sess.get_inputs()[0]
        self._out = self._sess.get_outputs()[0].name
        shape = self._inp.shape
        # NCHW dynamic batch
        self._h = int(shape[2]) if shape[2] not in (None, "None") else 256
        self._w = int(shape[3]) if shape[3] not in (None, "None") else 128

    @property
    def ok(self) -> bool:
        return True

    def __call__(self, frame_bgr: np.ndarray, xyxy: tuple[int, int, int, int]) -> Optional[np.ndarray]:
        crop = crop_person_bgr(frame_bgr, xyxy)
        if crop is None or crop.size < 400:
            return None
        blob = preprocess_osnet_imagenet(crop, input_w=self._w, input_h=self._h)
        name = self._inp.name
        out = self._sess.run([self._out], {name: blob})[0]
        feat = np.asarray(out).reshape(-1).astype(np.float32)
        return l2_normalize(feat)


def try_load_osnet(
    base_dir: str,
) -> tuple[Optional[Callable[[np.ndarray, tuple[int, int, int, int]], Optional[np.ndarray]]], str]:
    path = os.environ.get("VIGILRAIL_OSNET_ONNX", "").strip()
    if not path:
        for cand in (
            os.path.join(base_dir, "osnet_x1_0.onnx"),
            os.path.join(base_dir, "osnet.onnx"),
            os.path.join(base_dir, "reid_osnet.onnx"),
        ):
            if os.path.isfile(cand):
                path = cand
                break
    if not path or not os.path.isfile(path):
        return None, ""
    try:
        ext = OSNetOnnxExtractor(path)
        print(f"[ReID] Loaded OSNet ONNX: {path}")
        return ext, path
    except Exception as exc:
        print(f"[WARN] OSNet ONNX load failed ({path}): {exc}")
        return None, ""


_extractor_singleton: Optional[object] = None
_extractor_tried = False


def get_reid_extractor_cached(base_dir: str):
    """Load ONNX once per process (optional)."""
    global _extractor_singleton, _extractor_tried
    if _extractor_tried:
        return _extractor_singleton
    _extractor_tried = True
    ext, path = try_load_osnet(base_dir)
    _extractor_singleton = ext
    if ext is None and not path:
        print(
            "[ReID] No OSNet ONNX found; using IoU + center-distance fallback. "
            "Set VIGILRAIL_OSNET_ONNX or place osnet_x1_0.onnx next to app.py, "
            "and pip install onnxruntime."
        )
    return _extractor_singleton


class DictMemoryReIDTracker:
    """
    Temporal Re-ID without a heavy MOT library.

    Memory: ``prev_tracks[pid] = {"bbox": (x1,y1,x2,y2), "feat": L2-normalized np.ndarray}``

    Per frame (after YOLO gives boxes):
      1. Crop each box, run OSNet (or None), L2-normalize ``feat``.
      2. For each detection (high confidence first), pick best *unused* previous pid
         with ``dot(feat, prev_feat) > sim_threshold`` and center distance ``< max_center_dist_px``.
      3. If no match, assign ``next_id`` and increment.
      4. Replace ``prev_tracks`` with this frame's assignments (bbox + feat for next iteration).

    ``alerts.html`` unchanged — IDs are baked into the MJPEG in ``app.py``.
    """

    def __init__(
        self,
        extract_emb: Optional[Callable[[np.ndarray, tuple[int, int, int, int]], Optional[np.ndarray]]],
        sim_threshold: float = REID_MEM_SIM,
        max_center_dist_px: float = REID_MEM_DIST_PX,
    ) -> None:
        self._extract = extract_emb
        self.sim_threshold = sim_threshold
        self.max_center_dist_px = max_center_dist_px
        self.prev_tracks: dict[int, dict] = {}
        self.next_id = 0

    def update(self, frame: np.ndarray, detections: list) -> list:
        if not detections:
            self.prev_tracks.clear()
            return detections

        n = len(detections)
        feats: list[Optional[np.ndarray]] = []
        for d in detections:
            f: Optional[np.ndarray] = None
            if self._extract is not None:
                f = self._extract(frame, d["bbox"])
            if f is not None:
                f = np.asarray(f, dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(f))
                if norm > 1e-12:
                    f = f / norm
            feats.append(f)

        order = sorted(range(n), key=lambda i: -float(detections[i]["conf"]))
        used_prev_pids: set[int] = set()
        new_prev: dict[int, dict] = {}

        for ii in order:
            d = detections[ii]
            feat = feats[ii]
            x1, y1, x2, y2 = d["bbox"]
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5

            matched_id: Optional[int] = None
            best_sim = -1.0
            fallback_pid: Optional[int] = None
            best_dist = 1e18

            for pid, data in self.prev_tracks.items():
                if pid in used_prev_pids:
                    continue
                prev_feat = data.get("feat")
                px1, py1, px2, py2 = data["bbox"]
                cpx = (px1 + px2) * 0.5
                cpy = (py1 + py2) * 0.5
                dist = float(np.sqrt((cx - cpx) ** 2 + (cy - cpy) ** 2))

                if prev_feat is not None and feat is not None:
                    sim = float(np.dot(feat.reshape(-1), prev_feat.reshape(-1)))
                    if sim > best_sim and sim > self.sim_threshold and dist < self.max_center_dist_px:
                        best_sim = sim
                        matched_id = pid
                else:
                    half = self.max_center_dist_px * 0.5
                    if dist < half and dist < best_dist:
                        best_dist = dist
                        fallback_pid = pid

            if matched_id is None and fallback_pid is not None:
                matched_id = fallback_pid

            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1
            else:
                used_prev_pids.add(matched_id)

            d["track_id"] = matched_id
            old = self.prev_tracks.get(matched_id, {})
            stored_feat = feat if feat is not None else old.get("feat")
            new_prev[matched_id] = {"bbox": tuple(d["bbox"]), "feat": stored_feat}

        self.prev_tracks = new_prev
        return detections


@dataclass
class _Track:
    tid: int
    bbox: tuple[int, int, int, int]
    centroid: Optional[np.ndarray] = None  # L2-normalized embedding


class HybridReIDAssigner:
    """
    1) Greedy match by IoU between current boxes and previous-frame boxes.
    2) Remaining detections vs remaining tracks: cosine similarity (emb vs EMA centroid),
       or if embeddings unavailable, normalized center distance.
    3) Any still-unmatched detection gets a new ID.

    Centroid update: L2-normalize(α·centroid + (1-α)·emb) with α = REID_EMA_ALPHA.
    """

    def __init__(
        self,
        extract_emb: Optional[Callable[[np.ndarray, tuple[int, int, int, int]], Optional[np.ndarray]]],
        frame_hw: tuple[int, int],
        iou_thr: float = REID_IOU_THRESHOLD,
        cos_thr: float = REID_COSINE_THRESHOLD,
        ema_alpha: float = REID_EMA_ALPHA,
        center_max: float = REID_CENTER_DIST_MAX,
    ) -> None:
        self._extract = extract_emb
        self._fh, self._fw = frame_hw[0], frame_hw[1]
        self.iou_thr = iou_thr
        self.cos_thr = cos_thr
        self.ema_alpha = ema_alpha
        self.center_max = center_max
        self._next_id = 0
        self._tracks: dict[int, _Track] = {}

    def _next_track_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _merge_centroid(self, tr: _Track, emb: Optional[np.ndarray]) -> None:
        if emb is None:
            return
        if tr.centroid is None:
            tr.centroid = emb.copy()
        else:
            a = self.ema_alpha
            tr.centroid = l2_normalize(a * tr.centroid + (1.0 - a) * emb)

    def update(self, frame: np.ndarray, detections: list) -> list:
        if not detections:
            self._tracks.clear()
            return detections

        n = len(detections)
        embs: list[Optional[np.ndarray]] = []
        for d in detections:
            if self._extract is None:
                embs.append(None)
            else:
                embs.append(self._extract(frame, d["bbox"]))

        if not self._tracks:
            for i, d in enumerate(detections):
                tid = self._next_track_id()
                d["track_id"] = tid
                e = embs[i]
                self._tracks[tid] = _Track(tid=tid, bbox=d["bbox"], centroid=e.copy() if e is not None else None)
            return detections

        track_ids = list(self._tracks.keys())
        pairs: list[tuple[float, int, int]] = []
        for di in range(n):
            for tid in track_ids:
                iou = iou_xyxy(detections[di]["bbox"], self._tracks[tid].bbox)
                if iou >= self.iou_thr:
                    pairs.append((iou, di, tid))
        pairs.sort(reverse=True)

        assigned_det: set[int] = set()
        assigned_tr: set[int] = set()
        for _, di, tid in pairs:
            if di in assigned_det or tid in assigned_tr:
                continue
            detections[di]["track_id"] = tid
            assigned_det.add(di)
            assigned_tr.add(tid)
            tr = self._tracks[tid]
            tr.bbox = detections[di]["bbox"]
            self._merge_centroid(tr, embs[di])

        unmatched_det = [i for i in range(n) if i not in assigned_det]
        unmatched_tr = [t for t in track_ids if t not in assigned_tr]

        for di in list(unmatched_det):
            bbox = detections[di]["bbox"]
            emb = embs[di]
            best_tid: Optional[int] = None
            best: float = -1e9

            for tid in unmatched_tr:
                tr = self._tracks[tid]
                if emb is not None and tr.centroid is not None:
                    sim = cosine_similarity(emb, tr.centroid)
                    if sim > best:
                        best = sim
                        best_tid = tid
                else:
                    dist = center_distance_norm(bbox, tr.bbox, self._fh, self._fw)
                    neg_dist = -dist
                    if neg_dist > best:
                        best = neg_dist
                        best_tid = tid

            ok = False
            if best_tid is not None:
                tr = self._tracks[best_tid]
                if emb is not None and tr.centroid is not None:
                    ok = cosine_similarity(emb, tr.centroid) >= self.cos_thr
                else:
                    dist = center_distance_norm(bbox, tr.bbox, self._fh, self._fw)
                    ok = dist <= self.center_max

            if ok and best_tid is not None:
                tid = best_tid
                detections[di]["track_id"] = tid
                assigned_det.add(di)
                unmatched_tr.remove(tid)
                tr = self._tracks[tid]
                tr.bbox = bbox
                self._merge_centroid(tr, emb)

        for di in range(n):
            if di not in assigned_det:
                tid = self._next_track_id()
                detections[di]["track_id"] = tid
                e = embs[di]
                self._tracks[tid] = _Track(
                    tid=tid,
                    bbox=detections[di]["bbox"],
                    centroid=e.copy() if e is not None else None,
                )

        ids_present = {detections[di]["track_id"] for di in range(n)}
        self._tracks = {tid: self._tracks[tid] for tid in ids_present}
        for di in range(n):
            tid = detections[di]["track_id"]
            self._tracks[tid].bbox = detections[di]["bbox"]
        return detections
