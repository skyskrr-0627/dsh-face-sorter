# 人脸分组 · dsh-face-sorter

批量图片按「人」分类：扫描一个文件夹，把**每个不同的人**的照片分别归到一组，自动编号 `person_001`、`person_002`……；无人脸的照片和多人合照会单独列出。

- 全部本地离线推理（onnxruntime + InsightFace buffalo_l），图片不上传。
- 默认**不动原文件**：只生成预览、缩略图和报告；需要落盘时点「导出」才会**复制**到目标文件夹。
- 面板里拖动相似度阈值可以即时重新分组（复用已算好的人脸特征，不重新推理）。
- Agent 也能用：工具 `face_group_scan({ folder, threshold })`。

## 面板

左栏「人脸分组」进入。面板提供：照片文件夹选择（走系统原生「选择文件夹」对话框）、相似阈值、进度条、分组卡片（封面 + 张数，点开看全部人脸缩略图与相似度）、无人脸清单、最近若干次分析记录、「复制这一组 / 全部导出」。

## 目录

```
face-sorter/
├── plugin/            # 插件本体（本目录）
│   ├── index.js       # host 侧：任务、重组、缩略图、导出、工具
│   ├── client.js      # client 侧：main 面板 + 侧栏入口
│   ├── python/worker.py  # 检测(SCRFD) + 对齐 + 特征(ArcFace) + 聚类
│   ├── locale/        # 插件管理页文案
│   └── icon.svg
├── .venv/             # python 运行时（setup.sh 创建）
├── models/buffalo_l/  # det_10g.onnx / w600k_r50.onnx
├── runs/<jobId>/      # 每次分析的 progress/result/faces/report/thumbs
└── settings.json      # 面板设置（阈值、路径等）
```

## 首次安装

```bash
cd /Users/skyskrr/Documents/deepseek-harness/default-workspace/face-sorter
./setup.sh
```

`setup.sh` 会：创建 python 3.11 venv → 装 onnxruntime / opencv / numpy / pillow（走国内镜像）→ 下载 buffalo_l 模型（约 280MB）。

## 安装到 DSH

插件以 bundle 形式安装（`plugin/` 就是包目录）：

```
plugin_manager install_bundle target=/Users/skyskrr/Documents/deepseek-harness/default-workspace/face-sorter/plugin
```

安装后刷新 Web GUI（或重启 DSH），左栏会出现「人脸分组」。

## 以图找人

扫描过一次之后，把某个人的照片拖进面板的「以图找人」区域，
插件会在已扫描的图库里把这个人的照片全部筛出来并分组，可单独或批量复制导出。
支持多张查询照片、多人合照（一人一组），RAW/HEIC 也能直接拖。
不想点面板也可以直接让助手做：「在我的图库里找这个人，复制到桌面」。

## 使用要点

- **支持格式**：jpg / jpeg / png / webp / bmp / tif / tiff / heic / heif（子目录递归扫描）。RAW（raf / ARW / NEF）不支持，需要先导出成 jpg。
- **EXIF 方向**：读取时按 EXIF 旋转，竖拍照片不会歪。
- **阈值**：默认 0.42。同一人被拆成两组 → 调低；不同人混进同一组 → 调高。0.35 更宽松，0.50 更严格。
- **多人合照**：一张照片里有几个人，就会被复制进几个人的分组，`files` 里记为多组成员；`noFace` 是完全没有检测到人脸的图片。
- **导出**：目标目录下按分组名建子文件夹（`person_001/…`），同名文件自动加 `_2` 后缀，**不覆盖**已有文件，**不移动**原图。

## 排障

- 面板顶部两个状态点：「python 就绪 / 模型就绪」。任一为红，先跑 `./setup.sh`。
- 失败详情在 `runs/<jobId>/worker.log`，逐张错误在 `runs/<jobId>/result.json` 的 `errors` 与面板底部「读取失败」。
- 模型路径/解释器路径可在 `settings.json` 里改（或用面板所在运行目录的默认值）。
