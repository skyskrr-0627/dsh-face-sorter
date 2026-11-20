/**
 * dsh-face-sorter — runtime bootstrap.
 *
 * A freshly installed bundle has no python venv and no model weights, so the
 * panel offers one button that prepares both in a writable home directory
 * (`~/.dsh-face-sorter` by default). Everything here is idempotent: an already
 * prepared runtime is reported as ready and nothing is downloaded again.
 */
import { spawn } from 'node:child_process'
import { createWriteStream } from 'node:fs'
import { access, mkdir, readdir, rename, rm, stat, writeFile } from 'node:fs/promises'
import { Readable } from 'node:stream'
import os from 'node:os'
import path from 'node:path'

const MODEL_FILES = ['det_10g.onnx', 'w600k_r50.onnx']
const MODEL_SIZES = { 'det_10g.onnx': 16923893, 'w600k_r50.onnx': 174538686 }
const DEPS = ['onnxruntime', 'numpy', 'opencv-python-headless', 'pillow', 'pillow-heif', 'rawpy']

export const DEFAULT_MODEL_BASE = 'https://modelscope.cn/models/deepghs/insightface/resolve/master/buffalo_l'
export const DEFAULT_PIP_INDEX = 'https://pypi.org/simple'

export function runtimePaths(baseDir) {
  const root = baseDir
  return {
    root,
    venvDir: path.join(root, '.venv'),
    pythonPath: process.env.FACE_SORTER_PYTHON || path.join(root, '.venv', 'bin', 'python'),
    modelsDir: process.env.FACE_SORTER_MODELS || path.join(root, 'models', 'buffalo_l'),
  }
}

async function exists(file) {
  try {
    await access(file)
    return true
  } catch {
    return false
  }
}

async function findOnPath(names) {
  const dirs = String(process.env.PATH || '').split(path.delimiter).filter(Boolean)
  for (const name of names) {
    for (const dir of dirs) {
      const full = path.join(dir, name)
      if (await exists(full)) return full
    }
  }
  return null
}

function run(command, args, { env, onLine, timeoutMs = 0 } = {}) {
  return new Promise((resolve) => {
    const child = spawn(command, args, { env: { ...process.env, ...env } })
    let out = ''
    let err = ''
    const timer = timeoutMs
      ? setTimeout(() => {
        child.kill('SIGKILL')
      }, timeoutMs)
      : null
    const feed = (chunk, sink) => {
      const text = String(chunk)
      if (sink === 'out') out += text
      else err += text
      for (const line of text.split(/\r?\n/)) {
        const trimmed = line.trim()
        if (trimmed) onLine?.(trimmed)
      }
    }
    child.stdout.on('data', (chunk) => feed(chunk, 'out'))
    child.stderr.on('data', (chunk) => feed(chunk, 'err'))
    child.on('error', (error) => {
      if (timer) clearTimeout(timer)
      resolve({ code: -1, out, err: `${err}${error?.message || error}` })
    })
    child.on('close', (code) => {
      if (timer) clearTimeout(timer)
      resolve({ code: code ?? -1, out, err })
    })
  })
}

async function pythonVersion(pythonPath) {
  if (!(await exists(pythonPath))) return null
  const { code, out } = await run(pythonPath, ['-c', 'import sys;print("%d.%d"%sys.version_info[:2])'], { timeoutMs: 15000 })
  if (code !== 0) return null
  return out.trim() || null
}

async function modelBytes(modelsDir, name) {
  try {
    return (await stat(path.join(modelsDir, name))).size
  } catch {
    return 0
  }
}

// --------------------------------------------------------------------------- //
// runtime state
// --------------------------------------------------------------------------- //
const state = {
  phase: 'idle', // idle | running | done | error | cancelled
  step: '',
  message: '',
  percent: 0,
  error: null,
  startedAt: null,
  finishedAt: null,
  log: [],
  modelsBase: process.env.FACE_SORTER_MODEL_BASE || DEFAULT_MODEL_BASE,
  pipIndex: process.env.FACE_SORTER_PIP_INDEX || DEFAULT_PIP_INDEX,
  abort: null,
}

function note(line) {
  state.log.push(line)
  if (state.log.length > 60) state.log.shift()
  state.message = line
}

function setStep(step, percent, message) {
  state.step = step
  state.percent = Math.round(percent)
  if (message) note(message)
}

export async function inspectRuntime(baseDir) {
  const paths = runtimePaths(baseDir)
  const [python, detBytes, recBytes, uv] = await Promise.all([
    pythonVersion(process.env.FACE_SORTER_PYTHON || paths.pythonPath),
    modelBytes(paths.modelsDir, 'det_10g.onnx'),
    modelBytes(paths.modelsDir, 'w600k_r50.onnx'),
    findOnPath(['uv']),
  ])
  const modelsOk = detBytes > 1e6 && recBytes > 1e6
  let depsOk = false
  if (python) {
    const probe = await run(pythonPathOf(paths), ['-c', 'import onnxruntime, cv2, numpy, PIL'], { timeoutMs: 60000 })
    depsOk = probe.code === 0
  }
  return {
    baseDir: paths.root,
    pythonPath: pythonPathOf(paths),
    python,
    depsOk,
    modelsDir: paths.modelsDir,
    models: { det: detBytes, rec: recBytes, ok: modelsOk },
    uv,
    ready: Boolean(python && depsOk && modelsOk),
    needsPrepare: !(python && depsOk && modelsOk),
    modelBase: state.modelsBase,
    phase: state.phase,
    progress: { phase: state.phase, step: state.step, message: state.message, percent: state.percent, log: state.log.slice(-12), error: state.error },
  }
}

function pythonPathOf(paths) {
  return process.env.FACE_SORTER_PYTHON || paths.pythonPath
}

export function prepareStatus() {
  return {
    phase: state.phase,
    step: state.step,
    message: state.message,
    percent: state.percent,
    error: state.error,
    log: state.log.slice(-12),
    startedAt: state.startedAt,
    finishedAt: state.finishedAt,
  }
}

async function download(url, dest, { onProgress, signal }) {
  const response = await fetch(url, { signal })
  if (!response.ok) throw new Error(`下载失败 ${response.status}：${url}`)
  const total = Number(response.headers.get('content-length') || 0)
  const tmp = `${dest}.part`
  await mkdir(path.dirname(dest), { recursive: true })
  const file = createWriteStream(tmp)
  let written = 0
  if (response.body) {
    const stream = Readable.fromWeb(response.body)
    await new Promise((resolve, reject) => {
      stream.on('data', (chunk) => {
        written += chunk.length
        onProgress?.(written, total)
      })
      stream.on('error', reject)
      file.on('error', reject)
      file.on('finish', resolve)
      stream.pipe(file)
    })
    if (total && written !== total) throw new Error(`下载不完整：${written}/${total} 字节`)
  } else {
    const buffer = Buffer.from(await response.arrayBuffer())
    await writeFile(tmp, buffer)
    written = buffer.length
    onProgress?.(written, total || written)
  }
  await rm(dest, { force: true })
  await rename(tmp, dest)
  return written
}

/**
 * Prepare venv, dependencies and models. Returns the final runtime inspection.
 * Concurrent calls share the running preparation.
 */
export async function prepareRuntime(baseDir, options = {}) {
  if (state.phase === 'running') return { started: false, already: true, status: prepareStatus() }
  const paths = runtimePaths(baseDir)
  const modelsBase = options.modelBase || state.modelsBase
  const pipIndex = options.pipIndex || state.pipIndex
  state.phase = 'running'
  state.step = 'start'
  state.percent = 0
  state.error = null
  state.log = []
  state.startedAt = new Date().toISOString()
  state.finishedAt = null
  const abort = new AbortController()
  state.abort = abort

  void (async () => {
    try {
      await mkdir(paths.root, { recursive: true })

      // 1. python + venv ---------------------------------------------------- //
      setStep('venv', 4, '准备 python 运行时')
      const uv = await findOnPath(['uv'])
      const systemPython = await findOnPath(['python3.12', 'python3.11', 'python3.10', 'python3'])
      const existing = await pythonVersion(pythonPathOf(paths))
      if (!existing) {
        if (uv) {
          note('用 uv 创建虚拟环境（python 3.11）')
          const res = await run(uv, ['venv', '--python', '3.11', paths.venvDir], { onLine: note })
          if (res.code !== 0) throw new Error(`uv venv 失败：${res.err.trim().slice(-400) || res.code}`)
        } else if (systemPython) {
          const ver = await pythonVersion(systemPython)
          if (!ver || Number(ver.split('.')[1]) < 10 || Number(ver.split('.')[0]) !== 3) {
            throw new Error(`系统 python 是 ${ver || '未知版本'}，需要 3.10+；请先安装 uv（https://astral.sh/uv）后重试`)
          }
          note(`用 ${systemPython}（${ver}）创建虚拟环境`)
          const res = await run(systemPython, ['-m', 'venv', paths.venvDir], { onLine: note })
          if (res.code !== 0) throw new Error(`python -m venv 失败：${res.err.trim().slice(-400) || res.code}`)
        } else {
          throw new Error('找不到 python3.10+ 或 uv。请先安装 uv：https://astral.sh/uv')
        }
      } else {
        note(`已有虚拟环境（python ${existing}）`)
      }
      const python = pythonPathOf(paths)
      setStep('deps', 20, '安装推理依赖（onnxruntime / opencv / rawpy…）')

      // 2. dependencies ------------------------------------------------------ //
      const probe = await run(python, ['-c', 'import onnxruntime, cv2, numpy, PIL'], { timeoutMs: 60000 })
      if (probe.code !== 0) {
        const indexes = [pipIndex, 'https://mirrors.aliyun.com/pypi/simple']
        let installed = false
        let lastError = ''
        for (const index of [...new Set(indexes)]) {
          note(`从 ${index} 安装依赖`)
          const args = uv
            ? ['pip', 'install', '--python', python, '--index-url', index, ...DEPS]
            : ['-m', 'pip', 'install', '--index-url', index, ...DEPS]
          // A slow or blocked registry must not hang the whole preparation.
          const res = uv
            ? await run(uv, args, { onLine: note, timeoutMs: 300000 })
            : await run(python, args, { onLine: note, timeoutMs: 300000 })
          if (res.code === 0) {
            installed = true
            break
          }
          lastError = (res.err || res.out).trim().slice(-500)
        }
        if (!installed) throw new Error(`依赖安装失败：${lastError || '未知错误'}`)
        const again = await run(python, ['-c', 'import onnxruntime, cv2, numpy, PIL, rawpy'], { timeoutMs: 60000 })
        if (again.code !== 0) throw new Error(`依赖装好了但导入失败：${(again.err || again.out).trim().slice(-400)}`)
      } else {
        note('依赖已就绪')
      }

      // 3. models ------------------------------------------------------------ //
      setStep('models', 45, '下载人脸模型（约 190MB）')
      await mkdir(paths.modelsDir, { recursive: true })
      let downloaded = 0
      const totalBytes = MODEL_FILES.reduce((sum, name) => sum + (MODEL_SIZES[name] || 0), 0)
      for (const name of MODEL_FILES) {
        const dest = path.join(paths.modelsDir, name)
        const have = await modelBytes(paths.modelsDir, name)
        if (have > 1e6) {
          note(`已存在 ${name}（${(have / 1048576).toFixed(1)}MB）`)
          downloaded += have
          continue
        }
        note(`下载 ${name}`)
        const baseShare = downloaded / (totalBytes || 1)
        await download(`${modelsBase}/${name}`, dest, {
          signal: abort.signal,
          onProgress: (written) => {
            const share = (downloaded + written) / (totalBytes || 1)
            state.percent = Math.min(95, Math.round(45 + 50 * Math.max(baseShare, share)))
            state.step = 'models'
            state.message = `下载 ${name}：${(written / 1048576).toFixed(1)}MB`
          },
        })
        downloaded += (await modelBytes(paths.modelsDir, name)) || MODEL_SIZES[name] || 0
      }

      // 4. verify ------------------------------------------------------------ //
      setStep('verify', 97, '校验运行环境')
      const verify = await run(python, [
        '-c',
        'import sys, onnxruntime, cv2, numpy, rawpy\n'
        + 'm=sys.argv[1]\n'
        + 'onnxruntime.InferenceSession(m+"/det_10g.onnx", providers=["CPUExecutionProvider"])\n'
        + 'onnxruntime.InferenceSession(m+"/w600k_r50.onnx", providers=["CPUExecutionProvider"])\n'
        + 'print("ok")',
        paths.modelsDir,
      ], { timeoutMs: 180000 })
      if (verify.code !== 0) throw new Error(`模型校验失败：${(verify.err || verify.out).trim().slice(-400)}`)

      state.phase = 'done'
      state.percent = 100
      setStep('done', 100, '运行环境已就绪')
      state.finishedAt = new Date().toISOString()
    } catch (error) {
      const aborted = abort.signal.aborted
      state.phase = aborted ? 'cancelled' : 'error'
      state.error = aborted ? '已取消' : String(error?.message || error)
      state.message = state.error
      state.finishedAt = new Date().toISOString()
    } finally {
      state.abort = null
    }
  })()

  return { started: true, status: prepareStatus() }
}

export function cancelPrepare() {
  if (state.abort) {
    state.abort.abort()
    return true
  }
  return false
}

/** Models directory is valid for clustering only when both files exist. */
export async function modelsReady(modelsDir) {
  const sizes = await Promise.all(MODEL_FILES.map((name) => modelBytes(modelsDir, name)))
  return sizes.every((size) => size > 1e6)
}

/** Remove a partially prepared runtime so a retry starts clean. */
export async function resetRuntime(baseDir) {
  if (state.phase === 'running') throw new Error('正在准备中，无法重置')
  const paths = runtimePaths(baseDir)
  await rm(paths.venvDir, { recursive: true, force: true })
  for (const name of MODEL_FILES) await rm(path.join(paths.modelsDir, name), { force: true })
  const left = await readdir(paths.modelsDir).catch(() => [])
  if (!left.length) await rm(paths.modelsDir, { recursive: true, force: true })
  state.phase = 'idle'
  state.percent = 0
  state.step = ''
  state.message = ''
  state.error = null
  state.log = []
  return inspectRuntime(baseDir)
}

export const modelFiles = MODEL_FILES
export const diskHint = () => path.join(os.homedir(), '.dsh-face-sorter')
