#!/usr/bin/env python3
"""Face grouping worker for the DSH face-sorter plugin.

Reads a job JSON file (argv[1]) describing a source folder and writes into
``outDir``:

  progress.json   live counters, overwritten while running
  result.json     groups of images per identity, with per-face thumbnails
  faces.json      raw face records incl. 512-d embeddings (for re-clustering)
  thumbs/*.jpg    cropped face thumbnails served by the host plugin
  report.md       human readable summary

Detection uses SCRFD (det_10g) and recognition uses ArcFace (w600k_r50) from
the InsightFace buffalo_l model pack, through onnxruntime only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from PIL import Image as _PILImage
try:  # iPhone HEIC/HEIF, when the optional dependency is installed
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # noqa: BLE001 - HEIC simply stays unsupported
    pass

_PILImage.MAX_IMAGE_PIXELS = None  # professional cameras exceed the bomb guard

IMG_EXT = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif",
}
# Camera RAW (decoded through libraw/rawpy): the formats the mainstream bodies
# write today, plus the older ones that still show up in archived folders.
RAW_EXT = {
    ".raf", ".arw", ".srf", ".sr2", ".cr2", ".cr3", ".crw", ".nef", ".nrw",
    ".dng", ".orf", ".rw2", ".rwl", ".raw", ".pef", ".srw", ".3fr", ".fff",
    ".iiq", ".cap", ".mrw", ".kdc", ".dcr", ".x3f", ".erf", ".mef", ".mos",
    ".pxn", ".r3d",
}
IMG_EXT = IMG_EXT | RAW_EXT
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    sys.stderr.write("[face-sorter] %s\n" % msg)
    sys.stderr.flush()


def write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


RAW_SUFFIXES = tuple(sorted(RAW_EXT))


def _fit(arr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = arr.shape[:2]
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        arr = cv2.resize(
            arr,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return arr


def load_raw(path: Path, max_side: int) -> np.ndarray:
    """Decode a camera RAW file to BGR via libraw (rawpy).

    `half_size` keeps 26MP bodies around a quarter of the pixels, which is still
    well above `max_side` and roughly 4x faster than a full demosaic.
    """
    import rawpy

    arr = None
    with rawpy.imread(str(path)) as raw:
        # The camera's own JPEG preview is ~1920px and costs milliseconds,
        # whereas a demosaic of a 26MP sensor costs ~2s. Faces only need a few
        # hundred pixels, so prefer the preview and fall back to postprocess.
        try:
            thumb = raw.extract_thumb()
        except Exception:  # noqa: BLE001 - many bodies embed no preview
            thumb = None
        if thumb is not None and thumb.format == rawpy.ThumbFormat.JPEG:
            arr = cv2.imdecode(np.frombuffer(thumb.data, np.uint8), cv2.IMREAD_COLOR)
        elif thumb is not None:
            arr = cv2.cvtColor(thumb.data, cv2.COLOR_RGB2BGR)
        else:
            rgb = raw.postprocess(
                half_size=True,
                use_camera_wb=True,
                output_bps=8,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
            )
            arr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if arr is None:
        raise ValueError("无法解码 RAW：%s" % path.name)
    return np.ascontiguousarray(_fit(arr, max_side))


def load_image(path: Path, max_side: int) -> np.ndarray:
    """Read an image as BGR, honouring EXIF orientation, capped to max_side.

    Camera RAW files go through libraw; everything else through Pillow.
    """
    if path.suffix.lower() in RAW_SUFFIXES:
        return load_raw(path, max_side)

    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, float(max_side) / float(max(w, h)))
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        arr = np.asarray(im)
    return np.ascontiguousarray(arr[:, :, ::-1])


def file_key(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8", "surrogateescape")).hexdigest()[:20]


# --------------------------------------------------------------------------- #
# SCRFD detector (onnxruntime, no insightface package)
# --------------------------------------------------------------------------- #
class SCRFD:
    def __init__(self, model_path: str, input_size=(640, 640), num_anchors=2):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_size = input_size
        self.fmc = 3
        self.strides = (8, 16, 32)
        self.num_anchors = num_anchors
        self._anchor_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self._layout_ok = self._check_layout()

    def _check_layout(self) -> bool:
        shapes = [o.shape for o in self.session.get_outputs()]
        n = len(shapes)

        def kind(shape):
            if len(shape) != 2:
                return "?"
            width = shape[1]
            if isinstance(width, int):
                if width == 1:
                    return "score"
                if width == 4:
                    return "bbox"
                if width == 10:
                    return "kps"
            return "?"

        kinds = [kind(s) for s in shapes]
        ok = kinds[:3] == ["score"] * 3 and kinds[3:6] == ["bbox"] * 3
        if not ok:
            log("unexpected detector output layout: %s" % (shapes,))
        return ok

    def _anchors(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached
        centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        centers = (centers * stride).reshape((-1, 2))
        centers = np.stack([centers] * self.num_anchors, axis=1).reshape((-1, 2))
        self._anchor_cache[key] = centers
        return centers

    @staticmethod
    def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def detect(self, img: np.ndarray, det_thresh: float = 0.5) -> tuple[np.ndarray, np.ndarray, float]:
        im_ratio = float(img.shape[0]) / float(img.shape[1])
        model_ratio = float(self.input_size[1]) / float(self.input_size[0])
        if im_ratio > model_ratio:
            new_h = self.input_size[1]
            new_w = int(new_h / im_ratio)
        else:
            new_w = self.input_size[0]
            new_h = int(new_w * im_ratio)
        det_scale = float(new_h) / float(img.shape[0])
        resized = cv2.resize(img, (new_w, new_h))
        det_img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        det_img[:new_h, :new_w] = resized
        blob = cv2.dnn.blobFromImage(
            det_img, 1.0 / 128, self.input_size, (127.5, 127.5, 127.5), swapRB=True
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        fmc = self.fmc
        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(self.strides):
            if self._layout_ok:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc] * stride
                kps_preds = net_outs[idx + fmc * 2] * stride
            else:  # pragma: no cover - layout fallback
                group = net_outs[idx * 3: idx * 3 + 3]
                scores = next(o for o in group if o.shape[1] == 1)
                bbox_preds = next(o for o in group if o.shape[1] == 4) * stride
                kps_preds = next(o for o in group if o.shape[1] == 10) * stride
            height = self.input_size[1] // stride
            width = self.input_size[0] // stride
            anchor_centers = self._anchors(height, width, stride)
            pos_inds = np.where(scores[:, 0] >= det_thresh)[0]
            if pos_inds.size == 0:
                continue
            bboxes = self._distance2bbox(anchor_centers, bbox_preds)
            kpss = self._distance2kps(anchor_centers, kps_preds)
            scores_list.append(scores[pos_inds, 0])
            bboxes_list.append(bboxes[pos_inds])
            kpss_list.append(kpss[pos_inds])
        if not scores_list:
            return np.zeros((0, 4), np.float32), np.zeros((0, 5, 2), np.float32), det_scale
        scores = np.concatenate(scores_list)
        boxes = np.concatenate(bboxes_list) / det_scale
        kpss = np.concatenate(kpss_list).reshape(-1, 5, 2) / det_scale
        order = scores.argsort()[::-1]
        keep = []
        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][ovr <= 0.4]
        keep = np.array(keep, dtype=np.int32)
        kept_boxes = np.concatenate(
            [boxes[keep], scores[keep].reshape(-1, 1)], axis=1
        ).astype(np.float32)
        return kept_boxes, kpss[keep], det_scale


# --------------------------------------------------------------------------- #
# ArcFace recogniser
# --------------------------------------------------------------------------- #
def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Least-squares similarity (rotation + uniform scale + translation).

    Implemented directly instead of cv2.estimateAffinePartial2D: with five
    landmark pairs the exact fit is what we want, and OpenCV 5 rejects some
    array layouts there.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 2:
        return None
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    var_src = float((src_c ** 2).sum()) / len(src)
    if var_src < 1e-9:
        return None
    cov = (dst_c.T @ src_c) / len(src)
    try:
        u, s, vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None
    d = 1.0 if np.linalg.det(u @ vt) > 0 else -1.0
    rotation = u @ np.diag([1.0, d]) @ vt
    scale = float(s[0] + d * s[1]) / var_src
    if not np.isfinite(scale) or abs(scale) < 1e-9:
        return None
    translation = dst_mean - scale * (rotation @ src_mean)
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[:2, :2] = (scale * rotation).astype(np.float32)
    matrix[:, 2] = translation.astype(np.float32)
    return matrix


class ArcFace:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def align(self, img: np.ndarray, kps: np.ndarray) -> np.ndarray | None:
        src = np.ascontiguousarray(kps, dtype=np.float64)
        if np.linalg.norm(src[0] - src[1]) < 1e-3:
            return None
        matrix = similarity_transform(src, ARCFACE_DST.astype(np.float64))
        if matrix is None:
            return None
        return cv2.warpAffine(img, matrix, (112, 112), borderValue=0.0)

    def embed(self, aligned: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(
            aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
        )
        out = self.session.run([self.output_name], {self.input_name: blob})[0]
        vec = out.reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #
def cluster_faces(faces: list[dict], threshold: float) -> list[dict]:
    """Greedy online clustering over L2-normalised embeddings."""
    order = sorted(range(len(faces)), key=lambda i: -faces[i]["quality"])
    centroids: list[np.ndarray] = []
    sums: list[np.ndarray] = []
    counts: list[int] = []
    assign = [-1] * len(faces)
    sims = [0.0] * len(faces)
    for idx in order:
        emb = np.asarray(faces[idx]["emb"], dtype=np.float32)
        if centroids:
            sim = np.asarray([float(np.dot(emb, c)) for c in centroids])
            best = int(np.argmax(sim))
            if sim[best] >= threshold:
                assign[idx] = best
                sims[idx] = float(sim[best])
                sums[best] = sums[best] + emb
                counts[best] += 1
                merged = sums[best] / float(counts[best])
                norm = float(np.linalg.norm(merged))
                centroids[best] = merged / norm if norm > 0 else merged
                continue
        assign[idx] = len(centroids)
        sims[idx] = 1.0
        sums.append(emb.copy())
        counts.append(1)
        centroids.append(emb.copy())
    for i, face in enumerate(faces):
        face["cluster"] = assign[i]
        face["clusterSim"] = round(sims[i], 4)
    return faces


# --------------------------------------------------------------------------- #
# query mode: match dropped photos against an existing scan
# --------------------------------------------------------------------------- #
def norm_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def face_crop(img, box, kps, name, thumbs_dir, thumb_size, rec):
    """Align + embed one detection and save its padded crop as a thumbnail."""
    aligned = rec.align(img, kps)
    if aligned is None:
        return None
    try:
        emb = rec.embed(aligned)
    except Exception:  # noqa: BLE001
        return None
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    pad = 0.25 * max(x2 - x1, y2 - y1)
    height, width = img.shape[:2]
    cx1, cy1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
    cx2, cy2 = min(width, int(x2 + pad)), min(height, int(y2 + pad))
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    scale = float(thumb_size) / float(max(ch, cw))
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))))
    cv2.imwrite(str(thumbs_dir / name), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    score = float(box[4]) if len(box) > 4 else 1.0
    width_px = max(1.0, x2 - x1)
    height_px = max(1.0, y2 - y1)
    return {
        "thumb": name,
        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        "quality": round((width_px * height_px) ** 0.5 * score, 2),
        "emb": [round(float(v), 5) for v in emb],
    }


def run_match(job: dict) -> int:
    out_dir = Path(job["outDir"]).expanduser()
    thumbs_dir = out_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    query_dir = Path(job["queryDir"]).expanduser()
    ref_root = Path(job["refRoot"]).expanduser()
    ref_faces_path = Path(job["refFaces"]).expanduser()
    ref_thumbs = Path(job["refThumbs"]).expanduser()
    models_dir = Path(job["modelsDir"]).expanduser()
    threshold = float(job.get("threshold", 0.42))
    det_thresh = float(job.get("detThreshold", 0.5))
    min_face = int(job.get("minFace", 32))
    max_side = int(job.get("maxSide", 1600))
    thumb_size = int(job.get("thumbSize", 192))
    max_faces = int(job.get("maxFacesPerImage", 8))
    max_matches = int(job.get("maxMatches", 150))
    progress_path = out_dir / "progress.json"
    started = time.time()

    def fail(message: str) -> int:
        write_json(out_dir / "result.json", {"ok": False, "mode": "match", "error": message})
        return 2

    def report(stage: str, done: int, total: int, extra: dict | None = None) -> None:
        payload = {
            "stage": stage,
            "done": done,
            "total": total,
            "elapsed": round(time.time() - started, 1),
        }
        if extra:
            payload.update(extra)
        write_json(progress_path, payload)

    if not query_dir.is_dir():
        return fail("没有收到任何查询图片")
    stored: dict = {}
    if ref_faces_path.is_file():
        try:
            stored = json.loads(ref_faces_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            stored = {}
    ref_faces = stored.get("faces") or []
    if not ref_faces:
        return fail("参照图库里没有任何人脸，请先扫描一个文件夹")

    ref_emb = norm_rows(np.asarray([f["emb"] for f in ref_faces], dtype=np.float32))

    # Which library group each reference photo belongs to, so the panel can say
    # "these hits mostly come from person_001 (50 photos in that group)".
    ref_group_of: dict[str, str] = {}
    ref_group_total: dict[str, int] = {}
    ref_result_path = Path(job["refResult"]).expanduser() if job.get("refResult") else None
    if ref_result_path is not None and ref_result_path.is_file():
        try:
            ref_result = json.loads(ref_result_path.read_text(encoding="utf-8"))
            for group in ref_result.get("groups") or []:
                ref_group_total[group["id"]] = len(group.get("files") or [])
                for rel in group.get("files") or []:
                    ref_group_of.setdefault(rel, group["id"])
        except Exception:  # noqa: BLE001
            ref_group_of.clear()

    query_files = sorted(
        p
        for p in query_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXT and not p.name.startswith(".")
    )
    if not query_files:
        return fail("没有收到任何查询图片")

    report("loading-models", 0, len(query_files))
    det = SCRFD(str(models_dir / "det_10g.onnx"))
    rec = ArcFace(str(models_dir / "w600k_r50.onnx"))

    queries: list[dict] = []
    errors: list[dict] = []
    for index, path in enumerate(query_files):
        try:
            img = load_image(path, max_side)
        except Exception as exc:  # noqa: BLE001
            errors.append({"file": path.name, "error": "无法读取：%s" % exc.__class__.__name__})
            report("scanning-queries", index + 1, len(query_files))
            continue
        if img.size == 0:
            errors.append({"file": path.name, "error": "空图像"})
            report("scanning-queries", index + 1, len(query_files))
            continue
        try:
            boxes, kpss, _scale = det.detect(img, det_thresh)
        except Exception as exc:  # noqa: BLE001
            errors.append({"file": path.name, "error": "检测失败：%s" % exc.__class__.__name__})
            report("scanning-queries", index + 1, len(query_files))
            continue
        candidates = []
        for box, kps in zip(boxes, kpss):
            width_px = float(box[2] - box[0])
            height_px = float(box[3] - box[1])
            if width_px < min_face or height_px < min_face:
                continue
            candidates.append((width_px * height_px, box, kps))
        candidates.sort(key=lambda item: -item[0])
        for slot, (_area, box, kps) in enumerate(candidates[:max_faces]):
            name = "q_%s_%d.jpg" % (file_key(path), slot)
            record = face_crop(img, box, kps, name, thumbs_dir, thumb_size, rec)
            if record is None:
                continue
            queries.append({"file": str(path), "rel": path.name, **record})
        report("scanning-queries", index + 1, len(query_files))

    if not queries:
        return fail("查询图片里没有检测到人脸，换一张更清楚的照片试试")

    report("matching", len(query_files), len(query_files))
    clustered = cluster_faces(queries, threshold)
    buckets: dict[int, list[dict]] = {}
    for face in clustered:
        buckets.setdefault(face["cluster"], []).append(face)

    groups: list[dict] = []
    no_match: list[dict] = []
    copied: set[str] = set()
    for position, (_cluster_id, members) in enumerate(
        sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ):
        embs = norm_rows(np.asarray([m["emb"] for m in members], dtype=np.float32))
        centroid = embs.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm
        sims = ref_emb @ centroid
        picked: list[dict] = []
        seen_files: set[str] = set()
        files: list[str] = []
        for ref_index in np.argsort(-sims):
            score = float(sims[int(ref_index)])
            if score < threshold or len(picked) >= max_matches:
                break
            ref_face = ref_faces[int(ref_index)]
            thumb_name = "r_" + ref_face["thumb"]
            if thumb_name not in copied:
                try:
                    shutil.copyfile(ref_thumbs / ref_face["thumb"], thumbs_dir / thumb_name)
                except Exception:  # noqa: BLE001
                    continue
                copied.add(thumb_name)
            picked.append(
                {
                    "rel": ref_face["rel"],
                    "thumb": thumb_name,
                    "sim": round(score, 3),
                    "quality": ref_face.get("quality", 0),
                }
            )
            if ref_face["rel"] not in seen_files:
                seen_files.add(ref_face["rel"])
                files.append(ref_face["rel"])
        group_id = "match_%03d" % (position + 1,)
        query_thumbs = [m["thumb"] for m in members]
        query_files_rel = sorted({m["rel"] for m in members})
        if not picked:
            no_match.append({"id": group_id, "queryThumbs": query_thumbs, "queryFiles": query_files_rel})
            continue
        tally: dict[str, int] = {}
        for rel in files:
            owner = ref_group_of.get(rel)
            if owner:
                tally[owner] = tally.get(owner, 0) + 1
        ref_groups = [
            {"id": owner, "hit": hit, "total": ref_group_total.get(owner, 0)}
            for owner, hit in sorted(tally.items(), key=lambda kv: -kv[1])[:3]
        ]
        groups.append(
            {
                "id": group_id,
                "refGroups": ref_groups,
                "count": len(files),
                "faces": len(picked),
                "cover": picked[0]["thumb"],
                "coverFile": picked[0]["rel"],
                "files": files,
                "members": picked,
                "queryThumbs": query_thumbs,
                "queryFiles": query_files_rel,
            }
        )

    result = {
        "ok": True,
        "mode": "match",
        "root": str(ref_root),
        "threshold": threshold,
        "counts": {
            "images": len(query_files),
            "queries": len(queries),
            "groups": len(groups),
            "matches": sum(g["count"] for g in groups),
            "noMatch": len(no_match),
        },
        "groups": groups,
        "noMatch": no_match,
        "noFace": [],
        "images": [],
        "errors": errors,
        "elapsed": round(time.time() - started, 1),
    }
    write_json(out_dir / "result.json", result)
    write_json(
        out_dir / "faces.json",
        {"root": str(ref_root), "mode": "match", "threshold": threshold, "faces": []},
    )
    lines = [
        "# 以图找人结果",
        "",
        "- 参照图库：`%s`" % ref_root,
        "- 查询图片：%d 张，检出人脸 %d 张" % (len(query_files), len(queries)),
        "- 命中分组：%d，命中图片：%d 张" % (len(groups), result["counts"]["matches"]),
        "- 相似度阈值：%.2f，用时：%ss" % (threshold, result["elapsed"]),
        "",
        "| 分组 | 命中图片 | 命中人脸 |",
        "| --- | --- | --- |",
    ]
    for group in groups:
        lines.append("| %s | %d | %d |" % (group["id"], group["count"], group["faces"]))
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report("done", len(query_files), len(query_files))
    return 0


# --------------------------------------------------------------------------- #
# main pipeline
# --------------------------------------------------------------------------- #
# 景别（shot size）只看画面里最大的一张脸占画面高度的比例：脸越大，景别越近。
# 不额外引入模型，普通扫描就能顺带把特写 / 中景 / 全景标出来。
SHOT_CLOSEUP = 0.30
SHOT_MEDIUM = 0.12


def shot_of(ratio: float) -> str:
    if ratio >= SHOT_CLOSEUP:
        return "closeup"
    if ratio >= SHOT_MEDIUM:
        return "medium"
    return "wide"


def main() -> int:
    job_path = Path(sys.argv[1])
    job = json.loads(job_path.read_text(encoding="utf-8"))
    if job.get("mode") == "match":
        try:
            return run_match(job)
        except Exception:  # noqa: BLE001
            log(traceback.format_exc())
            write_json(
                Path(job["outDir"]).expanduser() / "result.json",
                {"ok": False, "mode": "match", "error": "匹配过程出错，见 worker.log"},
            )
            return 3
    root = Path(job["root"]).expanduser()
    out_dir = Path(job["outDir"]).expanduser()
    models_dir = Path(job["modelsDir"]).expanduser()
    det_thresh = float(job.get("detThreshold", 0.5))
    threshold = float(job.get("threshold", 0.42))
    min_face = int(job.get("minFace", 32))
    max_side = int(job.get("maxSide", 1600))
    max_faces = int(job.get("maxFacesPerImage", 8))
    thumb_size = int(job.get("thumbSize", 192))

    thumbs_dir = out_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    started = time.time()

    if not root.is_dir():
        write_json(
            out_dir / "result.json",
            {"ok": False, "error": "源文件夹不存在：%s" % root},
        )
        return 2

    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXT and not p.name.startswith(".")
    )

    def report(stage: str, done: int, extra: dict | None = None) -> None:
        payload = {
            "stage": stage,
            "total": len(files),
            "done": done,
            "elapsed": round(time.time() - started, 1),
        }
        if extra:
            payload.update(extra)
        write_json(progress_path, payload)

    report("loading-models", 0)
    det = SCRFD(str(models_dir / "det_10g.onnx"))
    rec = ArcFace(str(models_dir / "w600k_r50.onnx"))

    faces: list[dict] = []
    images: list[dict] = []
    errors: list[dict] = []
    report("scanning", 0)

    for index, path in enumerate(files):
        record = {"file": str(path), "rel": str(path.relative_to(root))}
        try:
            img = load_image(path, max_side)
        except Exception as exc:  # noqa: BLE001
            errors.append({"file": str(path), "error": "无法读取：%s" % exc.__class__.__name__})
            images.append({**record, "faces": 0, "status": "unreadable"})
            report("scanning", index + 1)
            continue
        if img.size == 0:
            errors.append({"file": str(path), "error": "空图像"})
            images.append({**record, "faces": 0, "status": "unreadable"})
            report("scanning", index + 1)
            continue
        try:
            boxes, kpss, _scale = det.detect(img, det_thresh)
        except Exception as exc:  # noqa: BLE001
            errors.append({"file": str(path), "error": "检测失败：%s" % exc.__class__.__name__})
            images.append({**record, "faces": 0, "status": "error"})
            report("scanning", index + 1)
            continue

        kept = 0
        image_faces = 0
        candidates = []
        for box, kps in zip(boxes, kpss):
            w = float(box[2] - box[0])
            h = float(box[3] - box[1])
            if w < min_face or h < min_face:
                continue
            image_faces += 1
            candidates.append((w * h, box, kps))
        candidates.sort(key=lambda c: -c[0])
        for area, box, kps in candidates[:max_faces]:
            aligned = rec.align(img, kps)
            if aligned is None:
                continue
            try:
                emb = rec.embed(aligned)
            except Exception:  # noqa: BLE001
                continue
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            pad = 0.25 * max(x2 - x1, y2 - y1)
            h_img, w_img = img.shape[:2]
            cx1 = max(0, int(x1 - pad))
            cy1 = max(0, int(y1 - pad))
            cx2 = min(w_img, int(x2 + pad))
            cy2 = min(h_img, int(y2 + pad))
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]
            scale = float(thumb_size) / float(max(ch, cw))
            if scale < 1.0:
                crop = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))))
            name = "%s_%d.jpg" % (file_key(path), kept)
            cv2.imwrite(
                str(thumbs_dir / name),
                crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), 82],
            )
            kept += 1
            faces.append(
                {
                    "file": str(path),
                    "rel": str(record["rel"]),
                    "thumb": name,
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "detScore": round(float(box[4]) if len(box) > 4 else 1.0, 4),
                    "faceW": round(w, 1),
                    "faceH": round(h, 1),
                    "quality": round((w * h) ** 0.5 * (float(box[4]) if len(box) > 4 else 1.0), 2),
                    "emb": [round(float(v), 5) for v in emb],
                }
            )
        frame_h, frame_w = img.shape[:2]
        big_face = max(
            (
                float(box[3] - box[1])
                for box in boxes
                if (box[2] - box[0]) >= min_face and (box[3] - box[1]) >= min_face
            ),
            default=0.0,
        )
        ratio = float(big_face) / float(frame_h) if frame_h else 0.0
        images.append(
            {
                **record,
                "faces": image_faces,
                "kept": kept,
                "status": "multi" if image_faces > 1 else ("single" if image_faces == 1 else "none"),
                "width": int(frame_w),
                "height": int(frame_h),
                "shot": shot_of(ratio) if big_face > 0 else "none",
                "shotRatio": round(ratio, 3),
            }
        )
        report("scanning", index + 1)

    report("clustering", len(files))
    faces = cluster_faces(faces, threshold)

    buckets: dict[int, list[dict]] = {}
    for face in faces:
        buckets.setdefault(face["cluster"], []).append(face)
    ordered = sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    groups = []
    face_group_of: dict[str, str] = {}
    for gi, (_cid, members) in enumerate(ordered):
        label = "person_%03d" % (gi + 1)
        files_in_group: dict[str, dict] = {}
        for face in members:
            face_group_of[face["file"]] = label
            entry = files_in_group.get(face["file"])
            if entry is None or face["quality"] > entry["quality"]:
                files_in_group[face["file"]] = face
        cover = max(members, key=lambda f: f["quality"])
        groups.append(
            {
                "id": label,
                "count": len(files_in_group),
                "faces": len(members),
                "cover": cover["thumb"],
                "coverFile": cover["rel"],
                "files": sorted(f["rel"] for f in files_in_group.values()),
                "members": [
                    {
                        "rel": f["rel"],
                        "thumb": f["thumb"],
                        "sim": f["clusterSim"],
                        "quality": f["quality"],
                    }
                    for f in sorted(members, key=lambda f: (-f["clusterSim"], -f["quality"]))
                ],
            }
        )

    for img in images:
        img["group"] = face_group_of.get(img["file"])

    elapsed = round(time.time() - started, 1)
    result = {
        "ok": True,
        "root": str(root),
        "outDir": str(out_dir),
        "threshold": threshold,
        "detThreshold": det_thresh,
        "elapsed": elapsed,
        "counts": {
            "images": len(images),
            "withFace": sum(1 for i in images if i["faces"] > 0),
            "noFace": sum(1 for i in images if i["faces"] == 0),
            "faces": len(faces),
            "groups": len(groups),
            "errors": len(errors),
            "shots": {
                level: sum(1 for i in images if i.get("shot") == level)
                for level in ("closeup", "medium", "wide", "none")
            },
        },
        "groups": groups,
        "noFace": sorted(i["rel"] for i in images if i["faces"] == 0),
        "multiFace": sorted(i["rel"] for i in images if i["faces"] > 1),
        "errors": errors,
        "images": [
            {
                "rel": i["rel"],
                "file": i["file"],
                "faces": i["faces"],
                "status": i.get("status", ""),
                "group": i.get("group"),
                "width": i.get("width"),
                "height": i.get("height"),
                "shot": i.get("shot"),
                "shotRatio": i.get("shotRatio"),
            }
            for i in images
        ],
    }
    write_json(out_dir / "result.json", result)
    write_json(
        out_dir / "faces.json",
        {
            "root": str(root),
            "threshold": threshold,
            "faces": [
                {
                    "file": f["file"],
                    "rel": f["rel"],
                    "thumb": f["thumb"],
                    "bbox": f["bbox"],
                    "quality": f["quality"],
                    "emb": f["emb"],
                }
                for f in faces
            ],
        },
    )

    lines = [
        "# 人脸分组报告",
        "",
        "- 源文件夹：`%s`" % root,
        "- 图片：%d（含人脸 %d，无人脸 %d）" % (result["counts"]["images"], result["counts"]["withFace"], result["counts"]["noFace"]),
        "- 人脸：%d，分组：%d，相似度阈值：%.2f" % (result["counts"]["faces"], result["counts"]["groups"], threshold),
        "- 景别：特写 %d，中景 %d，全景 %d（无人脸 %d）"
        % (
            result["counts"]["shots"]["closeup"],
            result["counts"]["shots"]["medium"],
            result["counts"]["shots"]["wide"],
            result["counts"]["shots"]["none"],
        ),
        "- 用时：%ss" % elapsed,
        "",
        "| 分组 | 图片数 | 人脸数 |",
        "| --- | --- | --- |",
    ]
    for group in groups:
        lines.append("| %s | %d | %d |" % (group["id"], group["count"], group["faces"]))
    if result["noFace"]:
        lines += ["", "## 未检测到人脸", ""] + ["- `%s`" % p for p in result["noFace"]]
    if errors:
        lines += ["", "## 读取失败", ""] + ["- `%s` — %s" % (e["file"], e["error"]) for e in errors]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report("done", len(files), {"groups": len(groups)})
    log("done: %d images, %d groups in %ss" % (len(images), len(groups), elapsed))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        try:
            out = Path(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["outDir"])
            out.mkdir(parents=True, exist_ok=True)
            write_json(out / "result.json", {"ok": False, "error": traceback.format_exc()[-2000:]})
        except Exception:  # noqa: BLE001
            pass
        sys.exit(1)
