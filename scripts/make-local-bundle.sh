#!/bin/sh
# 打一个可以拷给别人直接安装的本地包：插件代码 + 预置模型（不需要 GitHub、不需要再下 190MB 模型）
set -eu
BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$BASE_DIR"

VERSION=$(node -p "require('./package.json').version" 2>/dev/null || grep -m1 '"version"' package.json | sed 's/.*"\([0-9][^"]*\)".*/\1/')
NAME="dsh-face-sorter-$VERSION-local"
OUT="$BASE_DIR/dist"
STAGE="$OUT/$NAME"

rm -rf "$STAGE"
mkdir -p "$STAGE"

# 1. 插件本体（根清单 + plugin/ + 说明 + 许可）
cp package.json README.md LICENSE "$STAGE"/
rsync -a --exclude 'node_modules' --exclude '.venv' plugin "$STAGE"/

# 2. 预置模型
MODELS="$BASE_DIR/models/buffalo_l"
for f in det_10g.onnx w600k_r50.onnx; do
  if [ ! -s "$MODELS/$f" ]; then
    echo "缺少模型 $f，先跑 ./setup.sh 再打包" >&2
    exit 1
  fi
done
mkdir -p "$STAGE/models/buffalo_l"
cp -c "$MODELS"/*.onnx "$STAGE/models/buffalo_l/" 2>/dev/null || cp "$MODELS"/*.onnx "$STAGE/models/buffalo_l/"

# 3. 一键放置模型的脚本
cat > "$STAGE/install.sh" <<'EOS'
#!/bin/sh
# 把预置模型放进 DSH 插件的运行时目录（默认 ~/.dsh-face-sorter），省掉 190MB 下载
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
DEST="${FACE_SORTER_HOME:-$HOME/.dsh-face-sorter}"
mkdir -p "${DEST}/models/buffalo_l"
for f in det_10g.onnx w600k_r50.onnx; do
  if [ -s "${DEST}/models/buffalo_l/$f" ]; then
    echo "已有 $f，跳过"
  else
    cp "$HERE/models/buffalo_l/$f" "${DEST}/models/buffalo_l/$f"
    echo "已放置 $f"
  fi
done
echo
echo "模型就位：${DEST}/models/buffalo_l"
echo "下一步："
echo "  1) 把本目录装成 DSH 插件（插件页安装框填本目录绝对路径：${HERE} ）"
echo "     或在任意对话里让 agent 用 plugin_manager 的 install_bundle 指向这个目录）"
echo "  2) 重启 DSH，打开「人脸分组」面板，点「准备运行环境」（此时只装约 60MB python 依赖）"
EOS
chmod +x "$STAGE/install.sh"

# 4. 说明
cat > "$STAGE/安装说明.md" <<EOS
# 人脸分组插件 · 本地安装包 v$VERSION

拷到对方机器后按顺序做三件事。

## 1. 放置预置模型（省掉 190MB 下载）

\`\`\`bash
cd 本目录
./install.sh
\`\`\`

默认放到 \`~/.dsh-face-sorter/models/buffalo_l\`；想换位置就用 \`FACE_SORTER_HOME=/some/dir ./install.sh\`。

## 2. 装成 DSH 插件

两种方式二选一：

- 在 DSH「插件」页的安装框里填**本目录的绝对路径**（本包就是一个标准插件包，根目录有 \`package.json\`）。
- 或者在任意对话里说「把 <本目录> 装成插件」，让 agent 调 \`plugin_manager\` 的 \`install_bundle\`。

装完重启 DSH，左栏会出现「人脸分组」。

## 3. 准备运行环境

打开面板点 **准备运行环境**：因为模型已经就位，这一步只建 venv 并安装约 60MB 的 python 依赖（\`onnxruntime\`、\`opencv-python-headless\`、\`pillow\`、\`pillow-heif\`、\`rawpy\`）。

## 前置条件

- macOS 或 Linux；python 3.10+（有 \`uv\` 更好，会自动取 3.11）
- 首次装依赖需要联网；国内网络会自动退到 aliyun 镜像

## 包含什么

\`\`\`
package.json        插件包清单（可直接被 DSH 安装）
plugin/             插件本体：host 路由、Web 面板、python worker、自举模块
models/buffalo_l/   det_10g.onnx + w600k_r50.onnx（约 191MB，已预置）
install.sh          放置模型的小脚本
\`\`\`
EOS

# 5. 打包
cd "$OUT"
rm -f "$NAME.zip"
zip -qr -X "$NAME.zip" "$NAME"
SIZE=$(du -sh "$NAME.zip" | cut -f1)
echo "打包完成：$OUT/$NAME.zip ($SIZE)"
echo "目录版本：$STAGE"
