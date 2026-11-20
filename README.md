# dsh-face-sorter

一个 [DeepSeek Harness](https://github.com/deepseek-ai) 插件：**批量图片按人脸分组筛选**，并支持**拖一张照片就把某个人从整个图库里找出来**。

全程本地离线推理，不联网、不上传任何照片。

## 能力

- **按人分组**：扫描一个文件夹，检测每张图里的人脸、提取 512 维特征、聚类，同一个人归一组（`person_001`、`person_002`…），多人合照与无人脸单独归类。
- **以图找人**：把一个（或几个）人的照片拖进面板，插件在已扫描的图库里把这个人的所有照片筛出来，按身份分组列出，会标注主要来自哪个原有分组（如 `person_001 命中 31/50`）。
- **RAW / HEIC**：索尼 ARW、富士 RAF、佳能 CR2/CR3、尼康 NEF、DNG、ORF、RW2、PEF… 共 29 种相机 RAW，优先读相机内嵌 JPEG 预览（约 0.09s/张），没有预览才走 demosaic；iPhone HEIC 也支持。
- **阈值即时重算**：调相似度滑杆不重跑模型，只在已有向量上重新聚类，秒级出结果。
- **只复制，不动原图**：导出按组复制到目标目录下的子文件夹。
- **给 AI 用的工具**：`face_group_scan` / `face_find_person`，可以直接跟助手说「在我的图库里找这个人，复制到桌面」。

## 结构

```
setup.sh              建 venv、装依赖、下模型（幂等）
plugin/               DSH 插件本体（host 路由 + Web 面板 + python worker）
  index.js            host：任务生命周期、重聚类、缩略图、导出、两个 agent 工具
  client.js           Web 面板（无构建，原生 ESM 客户端模块）
  python/worker.py    推理：SCRFD 检测 + ArcFace 识别 + 聚类（onnxruntime）
  docs/DETAIL.md      面板与工具的详细说明
models/buffalo_l/     det_10g.onnx + w600k_r50.onnx（不入库，由 setup.sh 下载）
```

## 安装

需要 macOS / Linux，python 3.10+，[uv](https://github.com/astral-sh/uv)。

```bash
git clone https://github.com/<you>/dsh-face-sorter.git
cd dsh-face-sorter
./setup.sh          # 约 300MB：onnxruntime + opencv + rawpy + buffalo_l 模型
```

模型从 ModelScope 拉取（GitHub release 在国内经常连不上）：

```
https://modelscope.cn/models/deepghs/insightface/resolve/master/buffalo_l/{det_10g,w600k_r50}.onnx
```

然后在 DSH 里安装插件（`plugin/` 目录，支持软链）：

```bash
# 在 DSH 里：plugin_manager install_bundle，target = 本仓库 plugin/ 的绝对路径
```

重启 DSH，左侧栏出现「人脸分组」。

## 用法

**按人分组**：填（或选）一个照片文件夹 → 「开始分析」。738 张 3300 万像素的 A7M5 素材约 262 秒（CPU），每张约 1.1 秒。

**以图找人**：先扫描过一次建立参照图库 → 把某人的照片拖进「以图找人」卡片 → 得到命中的身份分组。

阈值取相似度余弦值：

| 阈值 | 效果 |
| --- | --- |
| 0.30 | 招回高，容易把长得像的人并在一起 |
| 0.35 | 找人常用起点 |
| 0.42 | 默认，分组较稳 |
| 0.50+ | 严格，只认最高置信度 |

## 实现要点

- **检测**：SCRFD `det_10g.onnx`，640×640 letterbox，三尺度 stride 8/16/32，每位置 2 个 anchor。注意 anchor 必须按 insightface 的 per-position 交错顺序排（`np.stack(..., axis=1)`），否则框会整体漂移。
- **对齐**：5 点相似变换（手写 Umeyama/SVD，不依赖 `cv2.estimateAffinePartial2D`），warp 到 112×112。
- **特征**：ArcFace `w600k_r50.onnx`，512 维，L2 归一化。
- **聚类**：按质量降序贪心，质心为增量均值后归一化；JS 侧复刻同一套逻辑，滑阈值时不重跑模型。
- **找人**：查询人脸与查询者自身先聚类成身份，取质心与图库全部人脸做点积，超过阈值的按相似度排序，去重按图片。
- **质量分**：`sqrt(宽×高) × 检测分数`，同图多人不再同分。

## 隐私

照片和特征只在本机，`data/runs/<id>/` 下保存运行结果与 192px 人脸缩略图，删除该目录即彻底清除。插件不做任何网络请求（只有 `setup.sh` 下载模型时会联网）。

## License

MIT
