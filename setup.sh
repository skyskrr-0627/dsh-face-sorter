#!/usr/bin/env bash
# Face sorter runtime setup: python venv + onnxruntime stack + buffalo_l models.
#
# Idempotent: safe to re-run. Honours an existing model directory and skips
# the download when det_10g.onnx is already present.
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$BASE_DIR/.venv"
MODEL_DIR="$BASE_DIR/models/buffalo_l"

# uv's default cache (~/Library/Caches/uv) can live outside the writable
# sandbox, and pypi.org is slow from here; both are pinned deliberately.
export UV_CACHE_DIR="$BASE_DIR/.uvcache"
MIRROR="${FACE_SORTER_PYPI_MIRROR:-https://mirrors.aliyun.com/pypi/simple}"
# ModelScope mirrors the insightface buffalo_l bundle at ~14MB/s from here;
# the GitHub release endpoint resets the connection.
MODEL_BASE="${FACE_SORTER_MODEL_BASE:-https://modelscope.cn/models/deepghs/insightface/resolve/master/buffalo_l}"
MODEL_FILES="det_10g.onnx w600k_r50.onnx"

echo "==> python venv: $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.11 "$VENV"
fi

echo "==> python deps (mirror: $MIRROR)"
uv pip install --python "$VENV/bin/python" --index-url "$MIRROR" \
  onnxruntime numpy opencv-python-headless pillow pillow-heif rawpy

echo "==> models: $MODEL_DIR"
mkdir -p "$BASE_DIR/models"
if [ -f "$MODEL_DIR/det_10g.onnx" ] && [ -f "$MODEL_DIR/w600k_r50.onnx" ]; then
  echo "    already present, skipping download"
else
  for f in $MODEL_FILES; do
    echo "    fetching $f"
    curl -fL --retry 3 --max-time 1800 -o "$MODEL_DIR/$f" "$MODEL_BASE/$f"
  done
fi

ls -la "$MODEL_DIR"
echo "==> done. $("$VENV/bin/python" -c 'import onnxruntime, cv2, numpy, PIL; print("onnxruntime", onnxruntime.__version__, "cv2", cv2.__version__)')"
