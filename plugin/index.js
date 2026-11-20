/**
 * dsh-face-sorter — host side.
 *
 * Detects every face in a folder of photos, embeds it with ArcFace, clusters
 * the embeddings so that "the same person" lands in one group, and exposes the
 * result to the Web panel and to the agent through a tool.
 *
 * All heavy work happens in `python/worker.py` (onnxruntime + buffalo_l models);
 * this side owns job lifecycle, re-clustering, thumbnails and file export.
 */
import { spawn } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { existsSync } from 'node:fs'
import { copyFile, mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  cancelPrepare,
  inspectRuntime,
  modelsReady,
  prepareRuntime,
  prepareStatus,
  resetRuntime,
  runtimePaths,
} from './bootstrap.js'

const PLUGIN_DIR = path.dirname(fileURLToPath(import.meta.url))
// A published bundle must work for any user, so the heavy runtime (venv,
// models, job output) lives in a writable home directory. Existing setups that
// prepared the workspace path keep using it.
const HOME_DIR = path.join(os.homedir(), '.dsh-face-sorter')
const LEGACY_DIR = path.join(os.homedir(), 'Documents', 'deepseek-harness', 'default-workspace', 'face-sorter')
const BASE_DIR = process.env.FACE_SORTER_HOME
  || (existsSync(path.join(LEGACY_DIR, '.venv', 'bin', 'python')) && !existsSync(HOME_DIR) ? LEGACY_DIR : HOME_DIR)
const SCRIPT_PATH = path.join(PLUGIN_DIR, 'python', 'worker.py')
const DATA_DIR = path.join(BASE_DIR, 'data')
const RUNS_DIR = path.join(DATA_DIR, 'runs')
const UPLOAD_DIR = path.join(DATA_DIR, 'uploads')
const SETTINGS_PATH = path.join(DATA_DIR, 'settings.json')
const DEFAULT_PYTHON = process.env.FACE_SORTER_PYTHON || path.join(BASE_DIR, '.venv', 'bin', 'python')
const DEFAULT_MODELS = process.env.FACE_SORTER_MODELS || path.join(BASE_DIR, 'models', 'buffalo_l')

const RAW_EXT = [
  '.raf', '.arw', '.srf', '.sr2', '.cr2', '.cr3', '.crw', '.nef', '.nrw',
  '.dng', '.orf', '.rw2', '.rwl', '.raw', '.pef', '.srw', '.3fr', '.fff',
  '.iiq', '.cap', '.mrw', '.kdc', '.dcr', '.x3f', '.erf', '.mef', '.mos',
  '.pxn', '.r3d',
]
const IMG_EXT = new Set([
  '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.heic', '.heif',
  ...RAW_EXT,
])

const DEFAULT_SETTINGS = {
  pythonPath: DEFAULT_PYTHON,
  modelsDir: DEFAULT_MODELS,
  threshold: 0.42,
  detThreshold: 0.5,
  minFace: 32,
  maxSide: 1600,
  thumbSize: 192,
  lastRoot: '',
}

const jobs = new Map()

// --------------------------------------------------------------------------- //
// small helpers
// --------------------------------------------------------------------------- //
function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' },
  })
}

async function readJson(file, fallback = null) {
  try {
    return JSON.parse(await readFile(file, 'utf8'))
  } catch {
    return fallback
  }
}

async function writeJson(file, data) {
  await mkdir(path.dirname(file), { recursive: true })
  await writeFile(file, JSON.stringify(data), 'utf8')
}

async function readSettings() {
  return { ...DEFAULT_SETTINGS, ...((await readJson(SETTINGS_PATH, {})) || {}) }
}

async function saveSettings(patch) {
  const next = { ...(await readSettings()), ...patch }
  await writeJson(SETTINGS_PATH, next)
  return next
}

function stamp() {
  const d = new Date()
  const p = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
}

/**
 * Fail early with actionable text: a bundle installed from the registry has no
 * venv or weights until the panel's one-click preparation ran.
 */
async function assertRuntime(settings) {
  if (!existsSync(settings.pythonPath)) {
    throw new Error(`运行环境还没准备好（缺少 ${settings.pythonPath}）。打开「人脸分组」面板，点「准备运行环境」会自动安装依赖并下载模型。`)
  }
  if (!(await modelsReady(settings.modelsDir))) {
    throw new Error(`人脸模型还没下载完（${settings.modelsDir}）。打开「人脸分组」面板，点「准备运行环境」。`)
  }
}

// --------------------------------------------------------------------------- //
// clustering (mirrors python/worker.py so the threshold slider is instant)
// --------------------------------------------------------------------------- //
function dot(a, b) {
  let s = 0
  for (let i = 0; i < a.length; i += 1) s += a[i] * b[i]
  return s
}

function normalize(vec) {
  let n = 0
  for (let i = 0; i < vec.length; i += 1) n += vec[i] * vec[i]
  n = Math.sqrt(n)
  if (n === 0) return vec
  const out = new Float64Array(vec.length)
  for (let i = 0; i < vec.length; i += 1) out[i] = vec[i] / n
  return out
}

function clusterFaces(faces, threshold) {
  const order = faces
    .map((face, index) => ({ index, quality: Number(face.quality) || 0 }))
    .sort((a, b) => b.quality - a.quality)
  const centroids = []
  const sums = []
  const counts = []
  const assignment = new Array(faces.length).fill(-1)
  const sims = new Array(faces.length).fill(0)
  for (const { index } of order) {
    const emb = faces[index].emb
    let best = -1
    let bestSim = -1
    for (let c = 0; c < centroids.length; c += 1) {
      const sim = dot(emb, centroids[c])
      if (sim > bestSim) {
        bestSim = sim
        best = c
      }
    }
    if (best >= 0 && bestSim >= threshold) {
      assignment[index] = best
      sims[index] = bestSim
      const sum = sums[best]
      for (let i = 0; i < sum.length; i += 1) sum[i] += emb[i]
      counts[best] += 1
      const mean = new Float64Array(emb.length)
      for (let i = 0; i < mean.length; i += 1) mean[i] = sum[i] / counts[best]
      centroids[best] = normalize(mean)
    } else {
      assignment[index] = centroids.length
      sims[index] = 1
      sums.push(Float64Array.from(emb))
      counts.push(1)
      centroids.push(normalize(emb))
    }
  }
  return faces.map((face, index) => ({ ...face, cluster: assignment[index], clusterSim: Number(sims[index].toFixed(4)) }))
}

function buildResult(base, faces, threshold) {
  const buckets = new Map()
  for (const face of faces) {
    const list = buckets.get(face.cluster) || []
    list.push(face)
    buckets.set(face.cluster, list)
  }
  const ordered = [...buckets.entries()].sort((a, b) => b[1].length - a[1].length || a[0] - b[0])
  const groups = []
  const groupOfFile = new Map()
  ordered.forEach(([, members], index) => {
    const id = `person_${String(index + 1).padStart(3, '0')}`
    const perFile = new Map()
    for (const face of members) {
      const current = perFile.get(face.file)
      if (!current || face.quality > current.quality) perFile.set(face.file, face)
      groupOfFile.set(face.file, id)
    }
    const cover = members.reduce((best, face) => (face.quality > best.quality ? face : best), members[0])
    groups.push({
      id,
      count: perFile.size,
      faces: members.length,
      cover: cover.thumb,
      coverFile: cover.rel,
      files: [...perFile.values()].map((face) => face.rel).sort(),
      members: members
        .slice()
        .sort((a, b) => b.clusterSim - a.clusterSim || b.quality - a.quality)
        .map((face) => ({ rel: face.rel, thumb: face.thumb, sim: face.clusterSim, quality: face.quality })),
    })
  })
  const images = (base.images || []).map((image) => ({ ...image, group: groupOfFile.get(image.file) || null }))
  const noFace = images.filter((image) => !image.faces).map((image) => image.rel).sort()
  return {
    ...base,
    threshold,
    groups,
    images,
    counts: {
      ...(base.counts || {}),
      images: images.length || base.counts?.images || 0,
      withFace: images.filter((image) => image.faces > 0).length,
      noFace: noFace.length,
      faces: faces.length,
      groups: groups.length,
    },
    noFace,
  }
}

// --------------------------------------------------------------------------- //
// job lifecycle
// --------------------------------------------------------------------------- //
async function listImages(root) {
  const out = []
  async function walk(dir) {
    let entries
    try {
      entries = await readdir(dir, { withFileTypes: true })
    } catch {
      return
    }
    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue
      const full = path.join(dir, entry.name)
      if (entry.isDirectory()) await walk(full)
      else if (IMG_EXT.has(path.extname(entry.name).toLowerCase())) out.push(full)
    }
  }
  await walk(root)
  return out.sort()
}

async function startJob({ root, threshold, detThreshold }, logger) {
  const settings = await readSettings()
  const expanded = root.replace(/^~(?=$|\/)/, process.env.HOME || '~')
  const resolved = path.resolve(expanded)
  let info
  try {
    info = await stat(resolved)
  } catch {
    throw new Error(`文件夹不存在：${resolved}`)
  }
  if (!info.isDirectory()) throw new Error(`不是文件夹：${resolved}`)
  await assertRuntime(settings)

  const id = `${stamp()}-${randomUUID().slice(0, 6)}`
  const outDir = path.join(RUNS_DIR, id)
  await mkdir(outDir, { recursive: true })
  const jobFile = path.join(outDir, 'job.json')
  const job = {
    id,
    root: resolved,
    outDir,
    status: 'running',
    startedAt: new Date().toISOString(),
    threshold: Number(threshold ?? settings.threshold),
    detThreshold: Number(detThreshold ?? settings.detThreshold),
    logFile: path.join(outDir, 'worker.log'),
    progress: { stage: 'queued', done: 0, total: 0 },
  }
  await writeJson(jobFile, {
    root: resolved,
    outDir,
    modelsDir: settings.modelsDir,
    threshold: job.threshold,
    detThreshold: job.detThreshold,
    minFace: settings.minFace,
    maxSide: settings.maxSide,
    thumbSize: settings.thumbSize,
  })
  await writeJson(path.join(outDir, 'meta.json'), {
    id,
    root: resolved,
    startedAt: job.startedAt,
    threshold: job.threshold,
  })

  const logs = []
  const child = spawn(settings.pythonPath, ['-u', SCRIPT_PATH, jobFile], {
    cwd: BASE_DIR,
    env: { ...process.env, PYTHONUNBUFFERED: '1', OMP_NUM_THREADS: '4' },
  })
  job.child = child
  child.stdout.on('data', (chunk) => {
    logs.push(String(chunk))
    if (logs.length > 200) logs.shift()
  })
  child.stderr.on('data', (chunk) => {
    logs.push(String(chunk))
    if (logs.length > 200) logs.shift()
  })
  child.on('error', (error) => {
    job.status = 'error'
    job.error = String(error?.message || error)
    logger?.warn?.(`[face-sorter] worker spawn failed: ${job.error}`)
  })
  child.on('close', async (code) => {
    await writeFile(job.logFile, logs.join(''), 'utf8').catch(() => {})
    if (job.status === 'cancelled') return
    if (code === 0) {
      job.status = 'done'
      job.result = await readJson(path.join(outDir, 'result.json'), null)
    } else {
      const result = await readJson(path.join(outDir, 'result.json'), null)
      job.status = 'error'
      job.error = result?.error || `worker exited with code ${code}`
    }
    job.finishedAt = new Date().toISOString()
    logger?.info?.(`[face-sorter] job ${id} ${job.status}`)
  })

  jobs.set(id, job)
  if (jobs.size > 24) {
    const oldest = [...jobs.values()].filter((item) => item.status !== 'running').slice(0, jobs.size - 24)
    for (const item of oldest) jobs.delete(item.id)
  }
  return job
}

async function startMatch({ session, refId, threshold, detThreshold }, logger) {
  const settings = await readSettings()
  await assertRuntime(settings)
  if (!/^[a-zA-Z0-9_-]{6,64}$/.test(String(session || ''))) throw new Error('会话标识无效')
  const queryDir = path.join(UPLOAD_DIR, session)
  let staged = []
  try {
    staged = (await readdir(queryDir)).filter((name) => IMG_EXT.has(path.extname(name).toLowerCase()))
  } catch {
    throw new Error('还没有收到查询图片')
  }
  if (!staged.length) throw new Error('还没有收到查询图片')

  const refJob = await loadJob(String(refId || ''))
  if (!refJob) throw new Error('找不到参照的扫描结果，请先扫描一个文件夹')
  const refFaces = path.join(refJob.outDir, 'faces.json')
  if (!existsSync(refFaces)) throw new Error('这次的扫描没有可复用的人脸特征，请重新扫描')
  const refThumbs = path.join(refJob.outDir, 'thumbs')

  const id = `${stamp()}-${randomUUID().slice(0, 6)}`
  const outDir = path.join(RUNS_DIR, id)
  await mkdir(outDir, { recursive: true })
  const jobFile = path.join(outDir, 'job.json')
  const job = {
    id,
    mode: 'match',
    root: refJob.root,
    outDir,
    status: 'running',
    startedAt: new Date().toISOString(),
    threshold: Number(threshold ?? settings.threshold),
    detThreshold: Number(detThreshold ?? settings.detThreshold),
    refId: refJob.id,
    queries: staged.length,
    logFile: path.join(outDir, 'worker.log'),
    progress: { stage: 'queued', done: 0, total: staged.length },
  }
  await writeJson(jobFile, {
    mode: 'match',
    queryDir,
    refRoot: refJob.root,
    refFaces,
    refThumbs,
    refResult: path.join(refJob.outDir, 'result.json'),
    outDir,
    modelsDir: settings.modelsDir,
    threshold: job.threshold,
    detThreshold: job.detThreshold,
    minFace: settings.minFace,
    maxSide: settings.maxSide,
    thumbSize: settings.thumbSize,
  })
  await writeJson(path.join(outDir, 'meta.json'), {
    id,
    mode: 'match',
    root: refJob.root,
    refId: refJob.id,
    queries: staged.length,
    startedAt: job.startedAt,
    threshold: job.threshold,
  })

  const logs = []
  const child = spawn(settings.pythonPath, ['-u', SCRIPT_PATH, jobFile], {
    cwd: BASE_DIR,
    env: { ...process.env, PYTHONUNBUFFERED: '1', OMP_NUM_THREADS: '4' },
  })
  job.child = child
  const collect = (chunk) => {
    logs.push(String(chunk))
    if (logs.length > 200) logs.shift()
  }
  child.stdout.on('data', collect)
  child.stderr.on('data', collect)
  child.on('error', (error) => {
    job.status = 'error'
    job.error = String(error?.message || error)
    logger?.warn?.(`[face-sorter] match spawn failed: ${job.error}`)
  })
  child.on('close', async (code) => {
    await writeFile(job.logFile, logs.join(''), 'utf8').catch(() => {})
    if (job.status === 'cancelled') return
    const result = await readJson(path.join(outDir, 'result.json'), null)
    if (code === 0 && result?.ok) {
      job.status = 'done'
      job.result = result
    } else {
      job.status = 'error'
      job.error = result?.error || `worker exited with code ${code}`
    }
    job.finishedAt = new Date().toISOString()
    logger?.info?.(`[face-sorter] match ${id} ${job.status}`)
  })

  jobs.set(id, job)
  return job
}

async function jobProgress(job) {
  const live = await readJson(path.join(job.outDir, 'progress.json'), null)
  if (live) job.progress = live
  return {
    id: job.id,
    root: job.root,
    status: job.status,
    error: job.error || null,
    startedAt: job.startedAt,
    finishedAt: job.finishedAt || null,
    progress: job.progress,
    counts: job.result?.counts || null,
  }
}

async function loadJob(id) {
  const inMemory = jobs.get(id)
  if (inMemory) return inMemory
  const outDir = path.join(RUNS_DIR, id)
  const meta = await readJson(path.join(outDir, 'meta.json'), null)
  if (!meta) return null
  const result = await readJson(path.join(outDir, 'result.json'), null)
  const job = {
    id,
    root: meta.root,
    outDir,
    status: result ? (result.ok ? 'done' : 'error') : 'error',
    startedAt: meta.startedAt,
    threshold: meta.threshold,
    error: result && !result.ok ? result.error : null,
    result,
    progress: (await readJson(path.join(outDir, 'progress.json'), null)) || {},
  }
  jobs.set(id, job)
  return job
}

// --------------------------------------------------------------------------- //
// export
// --------------------------------------------------------------------------- //
function uniqueName(taken, name) {
  if (!taken.has(name)) return name
  const ext = path.extname(name)
  const stem = name.slice(0, name.length - ext.length)
  for (let i = 2; i < 10000; i += 1) {
    const candidate = `${stem}_${i}${ext}`
    if (!taken.has(candidate)) return candidate
  }
  return `${stem}_${randomUUID().slice(0, 6)}${ext}`
}

async function exportGroups(job, groupIds, destRoot) {
  const result = job.result
  if (!result?.groups) throw new Error('这次任务还没有结果')
  const selected = result.groups.filter((group) => !groupIds || groupIds.includes(group.id))
  if (!selected.length) throw new Error('没有选中任何分组')
  const dest = path.resolve(
    (destRoot || path.join(job.outDir, 'export')).replace(/^~(?=$|\/)/, process.env.HOME || '~'),
  )
  await mkdir(dest, { recursive: true })
  const taken = new Map()
  const copied = []
  const errors = []
  for (const group of selected) {
    const folder = path.join(dest, group.id)
    await mkdir(folder, { recursive: true })
    if (!taken.has(folder)) taken.set(folder, new Set())
    const names = taken.get(folder)
    for (const rel of group.files) {
      try {
        const name = uniqueName(names, path.basename(rel))
        names.add(name)
        await copyFile(path.join(job.root, rel), path.join(folder, name))
        copied.push({ group: group.id, file: rel, to: path.join(folder, name) })
      } catch (error) {
        errors.push({ group: group.id, file: rel, error: String(error?.message || error) })
      }
    }
  }
  return { dest, copied: copied.length, errors }
}

// --------------------------------------------------------------------------- //
// run lookup (for the find tool)
// --------------------------------------------------------------------------- //
async function listRuns() {
  const entries = await readdir(RUNS_DIR, { withFileTypes: true }).catch(() => [])
  return entries
    .filter((entry) => entry.isDirectory() && !entry.name.startsWith('.'))
    .map((entry) => entry.name)
    .sort()
    .reverse()
}

async function findRun({ refRunId, refFolder } = {}) {
  if (refRunId) {
    const job = await loadJob(String(refRunId))
    if (job) return job
    throw new Error(`找不到运行 ${refRunId}`)
  }
  const wanted = refFolder
    ? path.resolve(String(refFolder).replace(/^~(?=$|\/)/, process.env.HOME || '~'))
    : null
  for (const id of await listRuns()) {
    const job = await loadJob(id)
    if (!job?.result?.ok) continue
    if (!wanted || path.resolve(job.root) === wanted) return job
  }
  if (wanted) throw new Error(`没有扫描过 ${wanted}，先用 face_group_scan 扫描它`)
  throw new Error('还没有任何扫描结果，先用 face_group_scan 扫描一个文件夹')
}

async function copyIntoSession(session, photos) {
  const folder = path.join(UPLOAD_DIR, session)
  await mkdir(folder, { recursive: true })
  const copied = []
  for (const raw of photos) {
    const resolved = path.resolve(String(raw).replace(/^~(?=$|\/)/, process.env.HOME || '~'))
    if (!existsSync(resolved)) throw new Error(`照片不存在：${resolved}`)
    if (!IMG_EXT.has(path.extname(resolved).toLowerCase())) throw new Error(`不支持的图片格式：${resolved}`)
    const name = path.basename(resolved)
    await copyFile(resolved, path.join(folder, name))
    copied.push(name)
  }
  if (!copied.length) throw new Error('至少需要一张查询照片')
  return copied
}

// --------------------------------------------------------------------------- //
// agent tool
// --------------------------------------------------------------------------- //
const TOOL_NAME = 'face_group_scan'

function toolDefinition(ctx) {
  return {
    name: TOOL_NAME,
    description:
      'Group a local folder of photos by face identity. Detects every face (SCRFD), embeds it (ArcFace) and clusters faces so each distinct person gets one group; unnamed people are person_001, person_002, … Also reports the images with no face. Use when the user wants batch photo sorting by person. Returns a compact text summary plus the run id; the visual panel shows thumbnails.',
    parameters: {
      type: 'object',
      properties: {
        folder: { type: 'string', description: 'Absolute path (or ~/…) of the folder to scan. Scanned recursively.' },
        threshold: {
          type: 'number',
          description: 'Cosine similarity threshold for "same person", 0.30–0.60. Default 0.42; raise it to split look-alikes, lower it to merge one person split across groups.',
        },
      },
      required: ['folder'],
    },
    timeoutMs: 20 * 60 * 1000,
    isConcurrencySafe: () => true,
    async execute(args) {
      const folder = typeof args?.folder === 'string' ? args.folder.trim() : ''
      if (!folder) throw new Error(`${TOOL_NAME} needs a "folder" path.`)
      const threshold = typeof args?.threshold === 'number' ? args.threshold : undefined
      const job = await startJob({ root: folder, threshold }, ctx.logger)
      const deadline = Date.now() + 19 * 60 * 1000
      while (job.status === 'running' && Date.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 700))
      }
      if (job.status === 'running') {
        return `扫描仍在进行（运行 ${job.id}）。面板「人脸分组」里可以看进度。`
      }
      if (job.status === 'error') throw new Error(`人脸分组失败：${job.error}`)
      const result = job.result || (await readJson(path.join(job.outDir, 'result.json'), null))
      if (!result?.ok) throw new Error(`人脸分组失败：${result?.error || '未知错误'}`)
      const lines = [
        `运行 ${job.id} · 源目录 ${result.root}`,
        `图片 ${result.counts.images}（含人脸 ${result.counts.withFace}，无人脸 ${result.counts.noFace}）· 人脸 ${result.counts.faces} · 分组 ${result.counts.groups} · 阈值 ${result.threshold} · 用时 ${result.elapsed}s`,
      ]
      for (const group of result.groups.slice(0, 40)) {
        const sample = group.files.slice(0, 3).join(', ')
        lines.push(`- ${group.id}: ${group.count} 张（示例：${sample}${group.files.length > 3 ? ' …' : ''}）`)
      }
      if (result.groups.length > 40) lines.push(`… 其余 ${result.groups.length - 40} 组见面板`)
      if (result.counts.noFace) lines.push(`- 未检测到人脸：${result.counts.noFace} 张`)
      lines.push(`报告：${path.join(job.outDir, 'report.md')}`)
      return lines.join('\n')
    },
  }
}

const FIND_TOOL_NAME = 'face_find_person'

function findToolDefinition(ctx) {
  return {
    name: FIND_TOOL_NAME,
    description:
      'Find one specific person across an already-scanned photo library, using one or more reference photos of that person. Detects the faces in the reference photos, embeds them, and returns every library photo whose face similarity passes the threshold, grouped per identity. Use after face_group_scan, or when the user says "find this person in my photos" and gives a photo path. Optionally copies the hits into a folder.',
    parameters: {
      type: 'object',
      properties: {
        photos: {
          type: 'array',
          items: { type: 'string' },
          description: 'Absolute paths (or ~/…) of one or more reference photos containing the person. Each detected face becomes a query.',
        },
        refFolder: {
          type: 'string',
          description: 'Folder that was scanned before; the most recent run for it is used as the library. Omit to use the latest run.',
        },
        refRunId: { type: 'string', description: 'Explicit scan run id, e.g. 20261120-194129-228bbe. Wins over refFolder.' },
        threshold: {
          type: 'number',
          description: 'Cosine similarity threshold, 0.25–0.6. Default 0.35 for recall; raise toward 0.45 to cut false positives.',
        },
        exportTo: { type: 'string', description: 'Optional destination folder; hits are copied into one subfolder per identity.' },
      },
      required: ['photos'],
    },
    timeoutMs: 10 * 60 * 1000,
    isConcurrencySafe: () => true,
    async execute(args) {
      const photos = Array.isArray(args?.photos) ? args.photos : args?.photo ? [args.photo] : []
      if (!photos.length) throw new Error(`${FIND_TOOL_NAME} needs "photos": paths of reference photos.`)
      const threshold = typeof args?.threshold === 'number' ? args.threshold : 0.35
      const refJob = await findRun({ refRunId: args?.refRunId, refFolder: args?.refFolder })
      const session = `tool${randomUUID().replace(/-/g, '').slice(0, 20)}`
      await copyIntoSession(session, photos)
      const job = await startMatch({ session, refId: refJob.id, threshold }, ctx.logger)
      const deadline = Date.now() + 9 * 60 * 1000
      while (job.status === 'running' && Date.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 500))
      }
      if (job.status === 'running') return `识别仍在进行（运行 ${job.id}），可以稍后在面板里看结果。`
      if (job.status === 'error') throw new Error(`以图找人失败：${job.error}`)
      const result = job.result || (await readJson(path.join(job.outDir, 'result.json'), null))
      if (!result?.ok) throw new Error(`以图找人失败：${result?.error || '未知错误'}`)
      const lines = [
        `运行 ${job.id} · 参照图库 ${result.root}（扫描 ${refJob.id}）· 阈值 ${threshold}`,
        `查询照片 ${result.counts.images} 张 / 查询人脸 ${result.counts.queries} 张 → 身份 ${result.counts.groups} 个，命中图片 ${result.counts.matches} 张，未命中身份 ${result.counts.noMatch} 个`,
      ]
      for (const group of result.groups.slice(0, 20)) {
        const attribution = (group.refGroups || []).map((item) => `${item.id} ${item.hit}/${item.total}`).join('、')
        const sample = group.files.slice(0, 3).join(', ')
        lines.push(
          `- ${group.id}: ${group.count} 张（示例：${sample}${group.files.length > 3 ? ' …' : ''}）` +
            (attribution ? ` · 主要来自 ${attribution}` : '') +
            ` · 查询照 ${group.queryFiles.join('、')}`,
        )
      }
      if (result.groups.length > 20) lines.push(`… 其余 ${result.groups.length - 20} 个身份见面板`)
      if (result.errors?.length) lines.push(`读取失败 ${result.errors.length} 张`)
      if (args?.exportTo && result.groups.length) {
        const exported = await exportGroups(job, null, args.exportTo)
        lines.push(`已复制 ${exported.copied} 个文件到 ${exported.dest}`)
        if (exported.errors.length) lines.push(`复制失败 ${exported.errors.length} 个`)
      }
      lines.push(`面板「人脸分组」可看缩略图；报告：${path.join(job.outDir, 'report.md')}`)
      return lines.join('\n')
    },
  }
}

// --------------------------------------------------------------------------- //
// plugin entry
// --------------------------------------------------------------------------- //
export const inject = ['connection', 'tools']

export function apply(ctx) {
  const logger = ctx.logger

  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/face-sorter/state',
        methods: ['GET'],
        requestBody: 'buffered',
        fetch: async () => {
          const settings = await readSettings()
          let runs = []
          try {
            const entries = await readdir(RUNS_DIR, { withFileTypes: true })
            runs = entries
              .filter((entry) => entry.isDirectory())
              .map((entry) => entry.name)
              .sort()
              .reverse()
              .slice(0, 20)
          } catch {
            runs = []
          }
          const history = []
          for (const id of runs) {
            const meta = await readJson(path.join(RUNS_DIR, id, 'meta.json'), null)
            if (!meta) continue
            const result = await readJson(path.join(RUNS_DIR, id, 'result.json'), null)
            history.push({
              id,
              root: meta.root,
              startedAt: meta.startedAt,
              groups: result?.counts?.groups ?? null,
              images: result?.counts?.images ?? null,
            })
          }
          return json({
            ok: true,
            settings: {
              threshold: settings.threshold,
              detThreshold: settings.detThreshold,
              minFace: settings.minFace,
              maxSide: settings.maxSide,
              pythonPath: settings.pythonPath,
              modelsDir: settings.modelsDir,
              lastRoot: settings.lastRoot,
            },
            ready: {
              python: existsSync(settings.pythonPath),
              models: await modelsReady(settings.modelsDir),
            },
            script: SCRIPT_PATH,
            baseDir: BASE_DIR,
            history,
          })
        },
      }),
    'face-sorter: state route',
  )

  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/face-sorter/runtime',
        methods: ['GET'],
        requestBody: 'buffered',
        fetch: async () => json({ ok: true, ...(await inspectRuntime(BASE_DIR)) }),
      }),
    'face-sorter: runtime route',
  )

  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/face-sorter/prepare',
        methods: ['POST'],
        requestBody: 'buffered',
        fetch: async (request) => {
          let body = {}
          try {
            body = await request.json()
          } catch {
            body = {}
          }
          const action = String(body?.action || 'start')
          try {
            if (action === 'cancel') {
              const cancelled = cancelPrepare()
              return json({ ok: true, cancelled, status: prepareStatus() })
            }
            if (action === 'reset') {
              const runtime = await resetRuntime(BASE_DIR)
              return json({ ok: true, runtime })
            }
            const started = await prepareRuntime(BASE_DIR, {
              modelBase: body?.modelBase,
              pipIndex: body?.pipIndex,
            })
            return json({ ok: true, started: started.started, already: Boolean(started.already), status: prepareStatus() })
          } catch (error) {
            return json({ ok: false, error: String(error?.message || error) }, 400)
          }
        },
      }),
    'face-sorter: prepare route',
  )

  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/face-sorter/stage',
        methods: ['POST'],
        requestBody: 'buffered',
        fetch: async (request) => {
          try {
            const url = new URL(request.url, 'http://localhost')
            const session = String(url.searchParams.get('session') || '')
            const rawName = String(url.searchParams.get('name') || '')
            if (!/^[a-zA-Z0-9_-]{6,64}$/.test(session)) return json({ ok: false, error: '会话标识无效' }, 400)
            const name = path.basename(rawName).replace(/[^\w.\-\u4e00-\u9fa5]+/g, '_')
            if (!name || !IMG_EXT.has(path.extname(name).toLowerCase())) {
              return json({ ok: false, error: '不支持的图片格式' }, 400)
            }
            const buffer = Buffer.from(await request.arrayBuffer())
            if (!buffer.length) return json({ ok: false, error: '空文件' }, 400)
            if (buffer.length > 80 * 1024 * 1024) return json({ ok: false, error: '单张图片超过 80MB' }, 413)
            const folder = path.join(UPLOAD_DIR, session)
            await mkdir(folder, { recursive: true })
            await writeFile(path.join(folder, name), buffer)
            return json({ ok: true, session, name, bytes: buffer.length })
          } catch (error) {
            return json({ ok: false, error: String(error?.message || error) }, 400)
          }
        },
      }),
    'face-sorter: stage route',
  )

  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/face-sorter/match',
        methods: ['POST'],
        requestBody: 'buffered',
        fetch: async (request) => {
          try {
            const body = await request.json().catch(() => ({}))
            const job = await startMatch(
              {
                session: body?.session,
                refId: body?.refId,
                threshold: body?.threshold,
                detThreshold: body?.detThreshold,
              },
              logger,
            )
            return json({ ok: true, jobId: job.id, queries: job.queries })
          } catch (error) {
            return json({ ok: false, error: String(error?.message || error) }, 400)
          }
        },
      }),
    'face-sorter: match route',
  )

  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/face-sorter/scan',
        methods: ['POST'],
        requestBody: 'buffered',
        fetch: async (request) => {
          try {
            const body = await request.json().catch(() => ({}))
            const root = typeof body?.root === 'string' ? body.root : ''
            if (!root) return json({ ok: false, error: '缺少 root' }, 400)
            const job = await startJob(
              { root, threshold: body.threshold, detThreshold: body.detThreshold },
              logger,
            )
            await saveSettings({ lastRoot: job.root, threshold: job.threshold })
            return json({ ok: true, jobId: job.id })
          } catch (error) {
            return json({ ok: false, error: String(error?.message || error) }, 400)
          }
        },
      }),
    'face-sorter: scan route',
  )

  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/face-sorter/job',
        methods: ['GET'],
        requestBody: 'buffered',
        fetch: async (request) => {
          const url = new URL(request.url, 'http://localhost')
          const id = url.searchParams.get('id') || ''
          const job = await loadJob(id)
          if (!job) return json({ ok: false, error: 'not found' }, 404)
          return json({ ok: true, ...(await jobProgress(job)) })
        },
      }),
    'face-sorter: job route',
  )

  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/face-sorter/result',
        methods: ['GET'],
        requestBody: 'buffered',
        fetch: async (request) => {
          const url = new URL(request.url, 'http://localhost')
          const id = url.searchParams.get('id') || ''
          const job = await loadJob(id)
          if (!job) return json({ ok: false, error: 'not found' }, 404)
          const result = job.result || (await readJson(path.join(job.outDir, 'result.json'), null))
          if (!result) return json({ ok: false, error: '还没有结果' }, 409)
          return json({ ok: true, result: { ...result, id: job.id, status: job.status } })
        },
      }),
    'face-sorter: result route',
  )

  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/face-sorter/recluster',
        methods: ['POST'],
        requestBody: 'buffered',
        fetch: async (request) => {
          try {
            const body = await request.json().catch(() => ({}))
            const job = await loadJob(String(body?.id || ''))
            if (!job) return json({ ok: false, error: 'not found' }, 404)
            const threshold = Number(body?.threshold)
            if (!Number.isFinite(threshold) || threshold <= 0 || threshold > 1) {
              return json({ ok: false, error: 'threshold 需在 0–1 之间' }, 400)
            }
            const stored = await readJson(path.join(job.outDir, 'faces.json'), null)
            if (!stored?.faces?.length) return json({ ok: false, error: '这次任务没有可复用的人脸特征' }, 409)
            const baseResult = (job.result || (await readJson(path.join(job.outDir, 'result.json'), null))) || {}
            const base = {
              ...baseResult,
              images: baseResult.images || [],
              counts: baseResult.counts || {},
            }
            const clustered = clusterFaces(stored.faces, threshold)
            const rebuilt = buildResult({ ...base, root: job.root, elapsed: baseResult.elapsed, detThreshold: baseResult.detThreshold }, clustered, threshold)
            job.result = rebuilt
            job.threshold = threshold
            await writeJson(path.join(job.outDir, 'result.json'), rebuilt)
            await saveSettings({ threshold })
            return json({ ok: true, result: { ...rebuilt, id: job.id } })
          } catch (error) {
            return json({ ok: false, error: String(error?.message || error) }, 400)
          }
        },
      }),
    'face-sorter: recluster route',
  )

  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/face-sorter/thumb',
        methods: ['GET'],
        requestBody: 'buffered',
        fetch: async (request) => {
          const url = new URL(request.url, 'http://localhost')
          const id = url.searchParams.get('id') || ''
          const name = url.searchParams.get('name') || ''
          if (!id || !/^[A-Za-z0-9_.-]+$/.test(name)) return new Response('bad request', { status: 400 })
          const file = path.join(RUNS_DIR, id, 'thumbs', name)
          if (!file.startsWith(path.join(RUNS_DIR, id, 'thumbs'))) return new Response('bad request', { status: 400 })
          try {
            const bytes = await readFile(file)
            return new Response(new Uint8Array(bytes), {
              headers: { 'content-type': 'image/jpeg', 'cache-control': 'private, max-age=86400' },
            })
          } catch {
            return new Response('not found', { status: 404 })
          }
        },
      }),
    'face-sorter: thumb route',
  )

  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/face-sorter/export',
        methods: ['POST'],
        requestBody: 'buffered',
        fetch: async (request) => {
          try {
            const body = await request.json().catch(() => ({}))
            const job = await loadJob(String(body?.id || ''))
            if (!job) return json({ ok: false, error: 'not found' }, 404)
            if (job.status === 'running') return json({ ok: false, error: '任务还在运行' }, 409)
            const groups = Array.isArray(body?.groups) && body.groups.length ? body.groups : null
            const result = await exportGroups(job, groups, body?.dest)
            return json({ ok: true, ...result })
          } catch (error) {
            return json({ ok: false, error: String(error?.message || error) }, 400)
          }
        },
      }),
    'face-sorter: export route',
  )

  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/face-sorter/cancel',
        methods: ['POST'],
        requestBody: 'buffered',
        fetch: async (request) => {
          const body = await request.json().catch(() => ({}))
          const job = jobs.get(String(body?.id || ''))
          if (!job) return json({ ok: false, error: 'not found' }, 404)
          if (job.status === 'running' && job.child) {
            job.status = 'cancelled'
            job.child.kill('SIGTERM')
          }
          return json({ ok: true, id: job.id, status: job.status })
        },
      }),
    'face-sorter: cancel route',
  )

  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/face-sorter/pick-folder',
        methods: ['POST'],
        requestBody: 'buffered',
        fetch: async () => {
          const settings = await readSettings()
          const startDir = settings.lastRoot || process.env.HOME || '/'
          return await new Promise((resolve) => {
            const script = `POSIX path of (choose folder with prompt "选择照片文件夹" default location (POSIX file "${startDir.replace(/"/g, '\\"')}"))`
            const child = spawn('osascript', ['-e', script])
            let out = ''
            let err = ''
            child.stdout.on('data', (chunk) => {
              out += chunk
            })
            child.stderr.on('data', (chunk) => {
              err += chunk
            })
            child.on('close', (code) => {
              if (code === 0) resolve(json({ ok: true, path: out.trim().replace(/\/$/, '') }))
              else resolve(json({ ok: false, error: err.trim() || '已取消' }, 200))
            })
            child.on('error', () => resolve(json({ ok: false, error: '无法打开系统选择框' }, 200)))
          })
        },
      }),
    'face-sorter: pick folder route',
  )

  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/face-sorter/settings',
        methods: ['POST'],
        requestBody: 'buffered',
        fetch: async (request) => {
          const body = await request.json().catch(() => ({}))
          const patch = {}
          if (typeof body?.threshold === 'number') patch.threshold = body.threshold
          if (typeof body?.detThreshold === 'number') patch.detThreshold = body.detThreshold
          if (typeof body?.minFace === 'number') patch.minFace = body.minFace
          if (typeof body?.pythonPath === 'string' && body.pythonPath.trim()) patch.pythonPath = body.pythonPath.trim()
          if (typeof body?.modelsDir === 'string' && body.modelsDir.trim()) patch.modelsDir = body.modelsDir.trim()
          const settings = await saveSettings(patch)
          return json({ ok: true, settings })
        },
      }),
    'face-sorter: settings route',
  )

  ctx.effect(() => {
    try {
      ctx.tools.register(toolDefinition(ctx))
  ctx.tools.register(findToolDefinition(ctx))
    } catch (error) {
      logger?.warn?.(`[face-sorter] ${TOOL_NAME} registration skipped: ${error}`)
    }
    return () => {}
  }, 'face-sorter: tool')
}
