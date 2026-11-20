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
package.json          npm 包清单，指向 plugin/（供插件页安装码解析）
setup.sh              手动预建 venv、装依赖、下模型（幂等）
plugin/               DSH 插件本体（host 路由 + Web 面板 + python worker）
  index.js            host：任务生命周期、重聚类、缩略图、导出、两个 agent 工具
  bootstrap.js        运行环境自举：venv / 依赖 / 模型，带进度与取消
  client.js           Web 面板（无构建，原生 ESM 客户端模块）
  python/worker.py    推理：SCRFD 检测 + ArcFace 识别 + 聚类（onnxruntime）
  docs/DETAIL.md      面板与工具的详细说明
market/index.json     插件市场索引（format 1），供 Discover 添加索引源
```

运行时（venv、模型、扫描结果）默认在 `~/.dsh-face-sorter`；历史上放在工作区
`Documents/deepseek-harness/default-workspace/face-sorter` 的环境会被自动沿用。

## 安装

需要 macOS / Linux。python 依赖和模型由插件自己准备，不需要预先配置环境。

**方式一：插件页填安装码（推荐）**

在 DSH 侧栏「插件」页的安装框里填：

```
github:skyskrr-0627/dsh-face-sorter
```

首次安装要 clone 仓库，约 3–5 分钟。装完打开「人脸分组」面板，点一次 **准备运行环境**：它会在 `~/.dsh-face-sorter` 建虚拟环境、装 `onnxruntime`/`opencv`/`rawpy` 等依赖，并从 ModelScope 下载约 190MB 的 buffalo_l 模型（`det_10g.onnx`、`w600k_r50.onnx`）。有本机 python 3.10+ 就直接复用（不额外下载解释器）；没有才让 uv 下载 3.11（约 26MB）。依赖默认走 aliyun 镜像，失败再回官方 PyPI。

**在插件市场里看到它**

「插件」页的 Discover 标签只读取你自己添加的索引源（DSH 没有官方索引）。本仓库自带一份 format-1 索引：

```
https://gh-proxy.com/https://raw.githubusercontent.com/skyskrr-0627/dsh-face-sorter/main/market/index.json
```

CDN 镜像（若上面失效可换）：

```
https://gcore.jsdelivr.net/gh/skyskrr-0627/dsh-face-sorter@main/market/index.json
```

原始的 `raw.githubusercontent.com` 直连在国内多数网络不通，所以上面给的是加速地址。

在 Discover 里「Add index source」填上面任一地址，就能搜到「人脸分组」并一键安装（版本 pinned 到具体 commit）。

**方式三：拷给别人的离线包**

`scripts/make-local-bundle.sh` 会打出 `dist/dsh-face-sorter-<版本>-local.zip`（约 176MB）：里面是插件本体 + 预置好的两个模型 + 一个 `install.sh`。对方解压后：

```bash
cd dsh-face-sorter-0.2.1-local
./install.sh          # 把模型放进 ~/.dsh-face-sorter，省掉 190MB 下载
```

然后把该目录装成插件（插件页填这个目录的绝对路径，或让 agent 调 `install_bundle`），重启 DSH，打开面板点「准备运行环境」——此时只装约 60MB python 依赖。

**方式二：本地检出（改代码时用）**

```bash
git clone https://github.com/skyskrr-0627/dsh-face-sorter.git
cd dsh-face-sorter
./setup.sh          # 可选：提前建好 venv 与模型，等价于面板里的按钮
```

然后把 `plugin/` 目录通过 `plugin_manager install_bundle`（target = 该目录绝对路径）装进 profile。软链形式改动即生效，重启 DSH 后可见。

模型优先从 ModelScope 拉取（GitHub release 在国内经常连不上）：

```
https://modelscope.cn/models/deepghs/insightface/resolve/master/buffalo_l/{det_10g,w600k_r50}.onnx
```

可用环境变量覆盖：`FACE_SORTER_HOME`（运行时目录）、`FACE_SORTER_PIP_INDEX`（pip 索引，默认官方，失败自动退到 aliyun 镜像）、`FACE_SORTER_MODEL_BASE`（模型源）。

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
