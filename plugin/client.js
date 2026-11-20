/**
 * dsh-face-sorter — client side (Web panel).
 *
 * Registered slots:
 *   main              — the full page panel (左栏点「人脸分组」进入)
 *   sidebar.panellist — the sidebar entry that opens it
 *
 * No build step: this file is loaded as an ESM module by the DSH web client.
 */
window.__ModuleLoader__.load({
  id: 'dsh-face-sorter',
  factory(require) {
    const React = require('react')
    const h = React.createElement
    const { useCallback, useEffect, useMemo, useRef, useState } = React

    const API = '/api/face-sorter'
    const PANEL_ID = 'face-sorter'
    const STYLE_ID = 'dsh-face-sorter-style'

    const CSS = `
.fs-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: calc(var(--dsh-frame-top-clearance, 48px) + 8px) 24px 48px;
  min-height: 100%;
  background: var(--dsw-alias-bg-base);
  color: var(--dsw-alias-label-primary);
  box-sizing: border-box;
  overflow-y: auto;
}
.fs-head { display: flex; flex-direction: column; gap: 4px; }
.fs-title { font-size: 20px; font-weight: 600; letter-spacing: 0.01em; }
.fs-sub { font-size: 12px; color: var(--dsw-alias-label-tertiary); }
.fs-card {
  background: var(--dsw-alias-bg-layer-1);
  border: 1px solid var(--dsw-alias-border-l2);
  border-radius: 12px;
  padding: 14px 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.fs-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.fs-drop {
  border: 1px dashed var(--dsw-alias-border-l3);
  border-radius: 10px;
  padding: 18px 14px;
  text-align: center;
  font-size: 12px;
  color: var(--dsw-alias-label-secondary);
  background: var(--dsw-alias-bg-layer-2);
  cursor: pointer;
  transition: border-color 120ms ease, background 120ms ease;
}
.fs-drop:hover { border-color: var(--dsw-alias-brand-primary); }
.fs-drop-on {
  border-color: var(--dsw-alias-brand-primary);
  background: var(--dsw-alias-interactive-bg-hover);
  color: var(--dsw-alias-label-primary);
}
.fs-qrow { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.fs-qrow .fs-q { display: flex; flex-direction: column; align-items: center; gap: 2px; font-size: 11px; color: var(--dsw-alias-label-tertiary); }
.fs-qrow img { width: 48px; height: 48px; border-radius: 8px; object-fit: cover; border: 1px solid var(--dsw-alias-border-l2); }
.fs-match { display: flex; flex-direction: column; gap: 10px; }
.fs-match + .fs-match { border-top: 1px solid var(--dsw-alias-border-l1); padding-top: 12px; }
.fs-row-tight { gap: 8px; }
.fs-label { font-size: 12px; color: var(--dsw-alias-label-secondary); min-width: 56px; }
.fs-input {
  flex: 1 1 260px;
  min-width: 180px;
  height: 32px;
  padding: 0 10px;
  border-radius: 8px;
  border: 1px solid var(--dsw-alias-border-l2);
  background: var(--dsw-alias-bg-layer-2);
  color: var(--dsw-alias-label-primary);
  font-size: 13px;
  font-family: inherit;
  outline: none;
}
.fs-input:focus { border-color: var(--dsw-alias-brand-primary); }
.fs-input-sm { flex: 0 1 200px; }
.fs-btn {
  height: 32px;
  padding: 0 14px;
  border-radius: 8px;
  border: 1px solid var(--dsw-alias-border-l2);
  background: var(--dsw-alias-bg-layer-2);
  color: var(--dsw-alias-label-primary);
  font-size: 13px;
  font-family: inherit;
  cursor: pointer;
  transition: background 120ms ease;
  white-space: nowrap;
}
.fs-btn:hover { background: var(--dsw-alias-interactive-bg-hover); }
.fs-btn:active { background: var(--dsw-alias-interactive-bg-active); }
.fs-btn:disabled { opacity: 0.45; cursor: not-allowed; }
.fs-btn-primary {
  background: var(--dsw-alias-brand-primary);
  border-color: transparent;
  color: var(--dsw-alias-label-primary-foreground);
  font-weight: 600;
}
.fs-btn-primary:hover { filter: brightness(1.06); background: var(--dsw-alias-brand-primary); }
.fs-pill {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid var(--dsw-alias-border-l2);
  color: var(--dsw-alias-label-secondary);
}
.fs-pill-ok { color: var(--dsw-alias-state-success-primary); border-color: var(--dsw-alias-state-success-primary); }
.fs-pill-bad { color: var(--dsw-alias-state-error-primary); border-color: var(--dsw-alias-state-error-primary); }
.fs-bar { height: 6px; border-radius: 999px; background: var(--dsw-alias-bg-layer-3); overflow: hidden; }
.fs-bar-fill { height: 100%; background: var(--dsw-alias-brand-primary); transition: width 200ms linear; }
.fs-stats { display: flex; gap: 18px; flex-wrap: wrap; font-size: 12px; color: var(--dsw-alias-label-secondary); }
.fs-stat b { color: var(--dsw-alias-label-primary); font-size: 15px; font-weight: 600; display: block; }
.fs-groups { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 12px; }
.fs-group {
  border: 1px solid var(--dsw-alias-border-l2);
  border-radius: 12px;
  background: var(--dsw-alias-bg-layer-1);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  cursor: pointer;
  transition: border-color 120ms ease;
}
.fs-group:hover { border-color: var(--dsw-alias-border-l3); }
.fs-group-on { border-color: var(--dsw-alias-brand-primary); }
.fs-cover {
  width: 100%;
  aspect-ratio: 1 / 1;
  object-fit: cover;
  display: block;
  background: var(--dsw-alias-bg-layer-3);
}
.fs-group-body { padding: 8px 10px 10px; display: flex; flex-direction: column; gap: 2px; }
.fs-group-id { font-size: 13px; font-weight: 600; }
.fs-group-meta { font-size: 11px; color: var(--dsw-alias-label-tertiary); }
.fs-members { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 10px 10px; }
.fs-member { position: relative; width: 52px; height: 52px; border-radius: 8px; overflow: hidden; }
.fs-member img { width: 100%; height: 100%; object-fit: cover; display: block; }
.fs-member span {
  position: absolute;
  inset: auto 0 0 0;
  font-size: 9px;
  text-align: center;
  color: var(--dsw-alias-label-primary-foreground);
  background: var(--dsw-alias-bg-overlay);
  padding: 1px 0;
}
.fs-detail { display: flex; flex-direction: column; gap: 10px; }
.fs-detail-head { display: flex; align-items: center; gap: 10px; }
.fs-thumbs { display: flex; flex-wrap: wrap; gap: 8px; max-height: 380px; overflow-y: auto; }
.fs-thumb { width: 76px; }
.fs-thumb img { width: 76px; height: 76px; object-fit: cover; border-radius: 8px; display: block; background: var(--dsw-alias-bg-layer-3); }
.fs-thumb div { font-size: 10px; color: var(--dsw-alias-label-tertiary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.fs-empty { font-size: 12px; color: var(--dsw-alias-label-tertiary); padding: 8px 0; }
.fs-list { display: flex; flex-direction: column; gap: 4px; max-height: 220px; overflow-y: auto; font-size: 12px; color: var(--dsw-alias-label-secondary); }
.fs-list code { font-size: 11px; color: var(--dsw-alias-label-tertiary); }
.fs-err { font-size: 12px; color: var(--dsw-alias-state-error-primary); white-space: pre-wrap; }
.fs-note { font-size: 12px; color: var(--dsw-alias-state-success-primary); }
.fs-setup { border-color: var(--dsw-alias-state-warn-primary); }
.fs-setup-title { font-size: 13px; font-weight: 600; color: var(--dsw-alias-state-warn-primary); }
.fs-setup-log {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  line-height: 1.5;
  color: var(--dsw-alias-label-tertiary);
  background: var(--dsw-alias-bg-layer-3);
  border-radius: 6px;
  padding: 6px 8px;
  max-height: 96px;
  overflow: auto;
}
.fs-slider { width: 160px; accent-color: var(--dsw-alias-brand-primary); }
.fs-history { display: flex; flex-direction: column; gap: 6px; font-size: 12px; }
.fs-history button {
  text-align: left;
  background: none;
  border: none;
  color: var(--dsw-alias-link);
  cursor: pointer;
  font-size: 12px;
  padding: 2px 0;
  font-family: inherit;
}
`

    function ensureStyle() {
      if (typeof document === 'undefined') return
      if (document.getElementById(STYLE_ID)) return
      const el = document.createElement('style')
      el.id = STYLE_ID
      el.textContent = CSS
      document.head.appendChild(el)
    }

    async function api(path, body) {
      const init = body === undefined
        ? { method: 'GET' }
        : {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(body),
          }
      const response = await fetch(API + path, { credentials: 'same-origin', ...init })
      const data = await response.json().catch(() => ({}))
      if (!response.ok || data.ok === false) {
        throw new Error(data.error || `请求失败（${response.status}）`)
      }
      return data
    }

    // Thumbnails are served by the host route, which authenticates like any
    // other API call. A plain <img src> does not carry that auth, so the bytes
    // are fetched here and handed to the DOM as blob: URLs.
    const thumbCache = new Map()

    function releaseThumb(key) {
      const entry = thumbCache.get(key)
      thumbCache.delete(key)
      if (entry && typeof entry.then === 'function') {
        entry.then((url) => url && URL.revokeObjectURL(url)).catch(() => {})
      } else if (entry) {
        URL.revokeObjectURL(entry)
      }
    }

    function loadThumb(jobId, name) {
      const key = jobId + '/' + name
      const hit = thumbCache.get(key)
      if (hit) return hit
      if (thumbCache.size > 1200) {
        for (const stale of Array.from(thumbCache.keys()).slice(0, 400)) releaseThumb(stale)
      }
      const promise = fetch(`${API}/thumb?id=${encodeURIComponent(jobId)}&name=${encodeURIComponent(name)}`, {
        credentials: 'same-origin',
      })
        .then((response) => (response.ok ? response.blob() : Promise.reject(new Error(String(response.status)))))
        .then((blob) => URL.createObjectURL(blob))
        .catch(() => null)
      thumbCache.set(key, promise)
      return promise
    }

    function Thumb({ jobId, name, className, alt }) {
      const [url, setUrl] = useState(null)
      useEffect(() => {
        if (!jobId || !name) return undefined
        let alive = true
        Promise.resolve(loadThumb(jobId, name)).then((value) => {
          if (alive && value) setUrl(value)
        })
        return () => {
          alive = false
        }
      }, [jobId, name])
      return h('img', { className, src: url || undefined, alt: alt || '', loading: 'lazy' })
    }

    function GroupCard({ group, jobId, active, onToggle }) {
      return h(
        'div',
        {
          className: 'fs-group' + (active ? ' fs-group-on' : ''),
          onClick: () => onToggle(group.id),
          title: group.coverFile,
        },
        h(Thumb, { className: 'fs-cover', jobId, name: group.cover, alt: group.id }),
        h(
          'div',
          { className: 'fs-group-body' },
          h('div', { className: 'fs-group-id' }, group.id),
          h('div', { className: 'fs-group-meta' }, `${group.count} 张图片 · ${group.faces} 张人脸`),
        ),
      )
    }

    function GroupDetail({ group, jobId, onExport, busy }) {
      const [expanded, setExpanded] = useState(false)
      const shown = expanded ? group.members : group.members.slice(0, 12)
      return h(
        'div',
        { className: 'fs-card fs-detail' },
        h(
          'div',
          { className: 'fs-detail-head' },
          h('div', { className: 'fs-group-id' }, group.id),
          h('div', { className: 'fs-group-meta' }, `${group.count} 张图片 · ${group.faces} 张人脸`),
          h('div', { style: { flex: 1 } }),
          h(
            'button',
            { className: 'fs-btn', disabled: busy, onClick: () => onExport([group.id]) },
            '复制这一组',
          ),
        ),
        h(
          'div',
          { className: 'fs-thumbs' },
          shown.map((member) =>
            h(
              'div',
              { className: 'fs-thumb', key: member.thumb, title: `${member.rel} · 相似度 ${member.sim}` },
              h(Thumb, { jobId, name: member.thumb, alt: member.rel }),
              h('div', null, member.rel.split('/').pop()),
            ),
          ),
        ),
        group.members.length > 12
          ? h(
              'button',
              { className: 'fs-btn', onClick: () => setExpanded(!expanded) },
              expanded ? '收起' : `展开全部 ${group.members.length} 张`,
            )
          : null,
      )
    }

    const RAW_ACCEPT = '.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.heic,.heif,.raf,.arw,.srf,.sr2,.cr2,.cr3,.crw,.nef,.nrw,.dng,.orf,.rw2,.rwl,.raw,.pef,.srw,.3fr,.fff,.iiq,.cap,.mrw,.kdc,.dcr,.x3f,.erf,.mef,.mos,.pxn,.r3d'

    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

    function newSession() {
      const raw = typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID().replace(/-/g, '')
        : String(Date.now()) + Math.random().toString(16).slice(2)
      return 'q' + raw.slice(0, 24)
    }

    async function stageFile(session, file) {
      // Raw image bytes, one request per photo: a 40MB RAW would otherwise have
      // to be base64'd inside JSON and double in size.
      const url = `${API}/stage?session=${encodeURIComponent(session)}&name=${encodeURIComponent(file.name)}`
      const response = await fetch(url, { method: 'POST', credentials: 'same-origin', body: file })
      const data = await response.json().catch(() => ({}))
      if (!response.ok || data.ok === false) {
        throw new Error(data.error || `上传失败（${response.status}）`)
      }
      return data
    }

    function MatchFinder({ refJobId, threshold, onNotice }) {
      const [dragging, setDragging] = useState(false)
      const [busy, setBusy] = useState(false)
      const [stage, setStage] = useState('')
      const [error, setError] = useState('')
      const [result, setResult] = useState(null)
      const [matchJobId, setMatchJobId] = useState(null)
      const [openGroup, setOpenGroup] = useState(null)
      const inputRef = useRef(null)

      const run = useCallback(
        async (fileList) => {
          const files = Array.from(fileList || []).filter((file) => file && file.size > 0)
          if (!files.length) return
          if (!refJobId) {
            setError('先扫描一个文件夹，建立参照图库，再放图进来找。')
            return
          }
          setBusy(true)
          setError('')
          setResult(null)
          setMatchJobId(null)
          try {
            const session = newSession()
            for (let index = 0; index < files.length; index += 1) {
              setStage(`上传 ${index + 1}/${files.length}：${files[index].name}`)
              await stageFile(session, files[index])
            }
            setStage('识别中…')
            const started = await api('/match', { session, refId: refJobId, threshold })
            const id = started.jobId
            for (let tick = 0; tick < 600; tick += 1) {
              await sleep(700)
              const info = await api(`/job?id=${encodeURIComponent(id)}`)
              const progress = info.progress || {}
              setStage(
                progress.stage === 'matching'
                  ? '和参照图库比对中…'
                  : `识别 ${progress.done || 0}/${progress.total || files.length}`,
              )
              if (info.status !== 'running') {
                if (info.status === 'error') throw new Error(info.error || '识别失败')
                break
              }
            }
            const data = await api(`/result?id=${encodeURIComponent(id)}`)
            setResult(data.result)
            setMatchJobId(id)
            setStage('')
          } catch (err) {
            setError(String(err.message || err))
            setStage('')
          } finally {
            setBusy(false)
          }
        },
        [refJobId, threshold],
      )

      const exportMatch = useCallback(
        async (groups) => {
          if (!matchJobId) return
          try {
            const data = await api('/export', { id: matchJobId, groups })
            onNotice?.(`已复制 ${data.copied} 个文件到 ${data.dest}`)
          } catch (err) {
            setError(String(err.message || err))
          }
        },
        [matchJobId, onNotice],
      )

      const groups = result?.groups || []
      const counts = result?.counts || null

      return h(
        'div',
        { className: 'fs-card' },
        h('div', { className: 'fs-group-id' }, '以图找人'),
        h(
          'div',
          { className: 'fs-group-meta' },
          '把某个人的照片拖进来（可多张，也支持 RAW/HEIC），插件会认出照片里的人脸，在参照图库里把同一个人的图片筛出来。',
        ),
        h(
          'div',
          {
            className: 'fs-drop' + (dragging ? ' fs-drop-on' : ''),
            onClick: () => !busy && inputRef.current && inputRef.current.click(),
            onDragOver: (event) => {
              event.preventDefault()
              setDragging(true)
            },
            onDragLeave: () => setDragging(false),
            onDrop: (event) => {
              event.preventDefault()
              setDragging(false)
              if (!busy) run(event.dataTransfer?.files)
            },
          },
          busy ? stage || '处理中…' : '把图片拖到这里，或点击选择文件',
        ),
        h('input', {
          ref: inputRef,
          type: 'file',
          accept: RAW_ACCEPT,
          multiple: true,
          style: { display: 'none' },
          onChange: (event) => {
            run(event.target.files)
            event.target.value = ''
          },
        }),
        error ? h('div', { className: 'fs-err' }, error) : null,
        counts
          ? h(
              'div',
              { className: 'fs-row fs-row-tight' },
              h('span', { className: 'fs-pill' }, `${counts.queries} 张查询人脸`),
              h('span', { className: 'fs-pill' }, `${counts.groups} 个身份`),
              h('span', { className: 'fs-pill' }, `命中 ${counts.matches} 张图片`),
              counts.noMatch ? h('span', { className: 'fs-pill fs-pill-bad' }, `${counts.noMatch} 个没找到`) : null,
              h('div', { style: { flex: 1 } }),
              groups.length
                ? h(
                    'button',
                    { className: 'fs-btn fs-btn-primary', onClick: () => exportMatch(groups.map((g) => g.id)) },
                    '导出全部命中',
                  )
                : null,
            )
          : null,
        groups.length
          ? h(
              'div',
              { className: 'fs-match' },
              groups.map((group) => {
                const expanded = openGroup === group.id
                const shown = expanded ? group.members : group.members.slice(0, 12)
                const attribution = (group.refGroups || [])
                  .map((item) => `${item.id} 命中 ${item.hit}/${item.total}`)
                  .join(' · ')
                return h(
                  'div',
                  { className: 'fs-match', key: group.id },
                  h(
                    'div',
                    { className: 'fs-row fs-row-tight' },
                    h(
                      'div',
                      { className: 'fs-qrow' },
                      (group.queryThumbs || []).map((thumb, index) =>
                        h(
                          'div',
                          { className: 'fs-q', key: thumb + index },
                          h(Thumb, { jobId: matchJobId, name: thumb, alt: '查询人脸' }),
                          h('span', null, '查询' + (index + 1)),
                        ),
                      ),
                    ),
                    h('div', { style: { flex: 1 } }),
                    h('div', { className: 'fs-group-meta' }, `${group.count} 张图片 · ${group.faces} 张人脸`),
                    attribution ? h('span', { className: 'fs-pill' }, attribution) : null,
                    h(
                      'button',
                      { className: 'fs-btn', onClick: () => exportMatch([group.id]) },
                      '导出这一组',
                    ),
                  ),
                  h(
                    'div',
                    { className: 'fs-thumbs' },
                    shown.map((member) =>
                      h(
                        'div',
                        { className: 'fs-thumb', key: member.thumb, title: `${member.rel} · 相似度 ${member.sim}` },
                        h(Thumb, { jobId: matchJobId, name: member.thumb, alt: member.rel }),
                        h('div', null, member.rel.split('/').pop()),
                      ),
                    ),
                  ),
                  group.members.length > 12
                    ? h(
                        'button',
                        {
                          className: 'fs-btn',
                          onClick: () => setOpenGroup(expanded ? null : group.id),
                        },
                        expanded ? '收起' : `展开全部 ${group.members.length} 张`,
                      )
                    : null,
                )
              }),
            )
          : null,
      )
    }

    /** One-click first-run setup: venv + python deps + buffalo_l weights. */
    function RuntimeSetup({ onReady }) {
      const [runtime, setRuntime] = useState(null)
      const [error, setError] = useState('')
      const [busy, setBusy] = useState(false)
      const timerRef = useRef(null)

      const stopPolling = useCallback(() => {
        if (timerRef.current) {
          clearInterval(timerRef.current)
          timerRef.current = null
        }
      }, [])

      const load = useCallback(async () => {
        try {
          const data = await api('/runtime')
          setRuntime(data)
          if (data.ready) {
            stopPolling()
            setBusy(false)
            onReady?.()
          }
          if (data.phase && data.phase !== 'running') setBusy(false)
          return data
        } catch (err) {
          setError(String(err.message || err))
          stopPolling()
          setBusy(false)
          return null
        }
      }, [onReady, stopPolling])

      const startPolling = useCallback(() => {
        stopPolling()
        timerRef.current = setInterval(load, 900)
      }, [load, stopPolling])

      useEffect(() => {
        load()
        return stopPolling
      }, [load, stopPolling])

      const prepare = useCallback(async () => {
        setError('')
        setBusy(true)
        try {
          await api('/prepare', { action: 'start' })
          await load()
          startPolling()
        } catch (err) {
          setBusy(false)
          setError(String(err.message || err))
        }
      }, [load, startPolling])

      const cancel = useCallback(async () => {
        try {
          await api('/prepare', { action: 'cancel' })
        } catch (err) {
          setError(String(err.message || err))
        }
      }, [])

      if (!runtime || runtime.ready) return null
      const progress = runtime.progress || {}
      const running = progress.phase === 'running' || busy
      const percent = progress.percent || 0
      const lines = progress.log || []

      return h(
        'div',
        { className: 'fs-card fs-setup' },
        h('div', { className: 'fs-setup-title' }, '运行环境还没准备好'),
        h(
          'div',
          { className: 'fs-sub' },
          `首次使用需要安装 python 依赖并下载约 190MB 人脸模型，全自动，预计 1–3 分钟。安装位置：${runtime.baseDir}`,
        ),
        h(
          'div',
          { className: 'fs-row' },
          h(
            'button',
            { className: 'fs-btn fs-btn-primary', disabled: running, onClick: prepare },
            running ? '准备中…' : '准备运行环境',
          ),
          running ? h('button', { className: 'fs-btn', onClick: cancel }, '取消') : null,
          h(
            'span',
            { className: 'fs-pill ' + (runtime.python ? 'fs-pill-ok' : 'fs-pill-bad') },
            runtime.python ? `python ${runtime.python}` : 'python 缺失',
          ),
          h(
            'span',
            { className: 'fs-pill ' + (runtime.models?.ok ? 'fs-pill-ok' : 'fs-pill-bad') },
            runtime.models?.ok ? '模型就绪' : '模型未下载',
          ),
        ),
        running || percent
          ? h(
            'div',
            { style: { display: 'flex', flexDirection: 'column', gap: 6 } },
            h('div', { className: 'fs-bar' }, h('div', { className: 'fs-bar-fill', style: { width: percent + '%' } })),
            h('div', { className: 'fs-group-meta' }, `${percent}% · ${progress.message || '准备中'}`),
            lines.length
              ? h(
                'div',
                { className: 'fs-setup-log' },
                ...lines.slice(-6).map((line, index) => h('div', { key: index }, line)),
              )
              : null,
          )
          : null,
        progress.error ? h('div', { className: 'fs-err' }, progress.error) : null,
        error ? h('div', { className: 'fs-err' }, error) : null,
      )
    }

    function Panel() {
      const [state, setState] = useState(null)
      const [root, setRoot] = useState('')
      const [threshold, setThreshold] = useState(0.42)
      const [dest, setDest] = useState('')
      const [jobId, setJobId] = useState(null)
      const [job, setJob] = useState(null)
      const [result, setResult] = useState(null)
      const [activeGroup, setActiveGroup] = useState(null)
      const [error, setError] = useState('')
      const [notice, setNotice] = useState('')
      const [busy, setBusy] = useState(false)
      const [pickBusy, setPickBusy] = useState(false)
      const pollRef = useRef(null)

      const refreshState = useCallback(async () => {
        try {
          const data = await api('/state')
          setState(data)
          setRoot((current) => current || data.settings.lastRoot || '')
          setThreshold((current) => (current === 0.42 ? data.settings.threshold : current))
        } catch (err) {
          setError(String(err.message || err))
        }
      }, [])

      useEffect(() => {
        ensureStyle()
        refreshState()
        return () => {
          if (pollRef.current) clearInterval(pollRef.current)
        }
      }, [refreshState])

      const loadResult = useCallback(async (id) => {
        try {
          const data = await api(`/result?id=${encodeURIComponent(id)}`)
          setResult(data.result)
          if (data.result?.threshold !== undefined) setThreshold(data.result.threshold)
        } catch (err) {
          setError(String(err.message || err))
        }
      }, [])

      const startPolling = useCallback(
        (id) => {
          if (pollRef.current) clearInterval(pollRef.current)
          pollRef.current = setInterval(async () => {
            try {
              const data = await api(`/job?id=${encodeURIComponent(id)}`)
              setJob(data)
              if (data.status !== 'running') {
                clearInterval(pollRef.current)
                pollRef.current = null
                setBusy(false)
                if (data.status === 'done') await loadResult(id)
                else if (data.status === 'error') setError(data.error || '分析失败')
              }
            } catch (err) {
              clearInterval(pollRef.current)
              pollRef.current = null
              setBusy(false)
              setError(String(err.message || err))
            }
          }, 800)
        },
        [loadResult],
      )

      const scan = useCallback(async () => {
        setError('')
        setNotice('')
        setResult(null)
        setJob(null)
        setActiveGroup(null)
        if (!root.trim()) {
          setError('先填一个照片文件夹路径，或点「选择文件夹」。')
          return
        }
        setBusy(true)
        try {
          const data = await api('/scan', { root: root.trim(), threshold })
          setJobId(data.jobId)
          startPolling(data.jobId)
        } catch (err) {
          setBusy(false)
          setError(String(err.message || err))
        }
      }, [root, threshold, startPolling])

      const cancel = useCallback(async () => {
        if (!jobId) return
        try {
          await api('/cancel', { id: jobId })
        } catch (err) {
          setError(String(err.message || err))
        }
      }, [jobId])

      const pickFolder = useCallback(async () => {
        setPickBusy(true)
        setError('')
        try {
          const data = await api('/pick-folder', {})
          if (data.path) setRoot(data.path)
          else if (data.error) setError(data.error)
        } catch (err) {
          setError(String(err.message || err))
        } finally {
          setPickBusy(false)
        }
      }, [])

      const recluster = useCallback(
        async (value) => {
          if (!jobId || !result) return
          setBusy(true)
          setError('')
          try {
            const data = await api('/recluster', { id: jobId, threshold: value })
            setResult(data.result)
            setNotice(`已按阈值 ${value.toFixed(2)} 重新分组`)
          } catch (err) {
            setError(String(err.message || err))
          } finally {
            setBusy(false)
          }
        },
        [jobId, result],
      )

      const doExport = useCallback(
        async (groups) => {
          if (!jobId) return
          setBusy(true)
          setError('')
          setNotice('')
          try {
            const data = await api('/export', { id: jobId, groups, dest: dest.trim() || undefined })
            setNotice(`已复制 ${data.copied} 个文件到 ${data.dest}${data.errors?.length ? `（${data.errors.length} 个失败）` : ''}`)
          } catch (err) {
            setError(String(err.message || err))
          } finally {
            setBusy(false)
          }
        },
        [jobId, dest],
      )

      const progress = job?.progress || {}
      const total = progress.total || 0
      const done = progress.done || 0
      const percent = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0
      const counts = result?.counts || job?.counts || null
      const groups = result?.groups || []
      const active = useMemo(() => groups.find((group) => group.id === activeGroup) || null, [groups, activeGroup])
      const ready = state?.ready

      return h(
        'div',
        { className: 'fs-panel' },
        h(
          'div',
          { className: 'fs-head' },
          h('div', { className: 'fs-title' }, '人脸分组'),
          h(
            'div',
            { className: 'fs-sub' },
            '扫描一个文件夹，把同一个人的照片归到一组；不搬动原文件，导出只是复制。',
          ),
          h(
            'div',
            { className: 'fs-row fs-row-tight', style: { marginTop: 6 } },
            h('span', { className: 'fs-pill ' + (ready?.python ? 'fs-pill-ok' : 'fs-pill-bad') },
              ready?.python ? 'python 就绪' : 'python 缺失'),
            h('span', { className: 'fs-pill ' + (ready?.models ? 'fs-pill-ok' : 'fs-pill-bad') },
              ready?.models ? '模型就绪' : '模型缺失'),
            state?.baseDir ? h('span', { className: 'fs-pill' }, state.baseDir) : null,
          ),
        ),

        h(RuntimeSetup, { onReady: refreshState }),

        h(
          'div',
          { className: 'fs-card' },
          h(
            'div',
            { className: 'fs-row' },
            h('span', { className: 'fs-label' }, '照片文件夹'),
            h('input', {
              className: 'fs-input',
              value: root,
              placeholder: '/Users/你/Pictures/某次活动',
              spellCheck: false,
              onChange: (event) => setRoot(event.target.value),
              onKeyDown: (event) => {
                if (event.key === 'Enter' && !busy) scan()
              },
            }),
            h('button', { className: 'fs-btn', disabled: pickBusy, onClick: pickFolder }, pickBusy ? '选择中…' : '选择文件夹'),
            h('button', { className: 'fs-btn fs-btn-primary', disabled: busy, onClick: scan }, busy ? '分析中…' : '开始分析'),
            busy ? h('button', { className: 'fs-btn', onClick: cancel }, '停止') : null,
          ),
          h(
            'div',
            { className: 'fs-row' },
            h('span', { className: 'fs-label' }, '相似阈值'),
            h('input', {
              className: 'fs-slider',
              type: 'range',
              min: 0.25,
              max: 0.7,
              step: 0.01,
              value: threshold,
              onChange: (event) => setThreshold(Number(event.target.value)),
              onPointerUp: () => result && recluster(threshold),
            }),
            h('span', { className: 'fs-pill' }, threshold.toFixed(2)),
            h('span', { className: 'fs-group-meta' }, '调高更严格（同一人可能被拆开），调低更宽松（不同人可能被合并）'),
          ),
          busy && job
            ? h(
                'div',
                { style: { display: 'flex', flexDirection: 'column', gap: 6 } },
                h('div', { className: 'fs-bar' }, h('div', { className: 'fs-bar-fill', style: { width: percent + '%' } })),
                h(
                  'div',
                  { className: 'fs-group-meta' },
                  `${progress.stage || '准备中'} · ${done}/${total} · ${percent}% · 已用 ${progress.elapsed ?? 0}s`,
                ),
              )
            : null,
          error ? h('div', { className: 'fs-err' }, error) : null,
          notice ? h('div', { className: 'fs-note' }, notice) : null,
        ),

        h(MatchFinder, { refJobId: jobId, threshold, onNotice: setNotice }),

        counts
          ? h(
              'div',
              { className: 'fs-card' },
              h(
                'div',
                { className: 'fs-stats' },
                h('div', null, h('b', null, String(counts.images ?? 0)), '图片'),
                h('div', null, h('b', null, String(counts.withFace ?? 0)), '含人脸'),
                h('div', null, h('b', null, String(counts.faces ?? 0)), '人脸'),
                h('div', null, h('b', null, String(counts.groups ?? 0)), '分组'),
                h('div', null, h('b', null, String(counts.noFace ?? 0)), '无人脸'),
                result?.elapsed ? h('div', null, h('b', null, `${result.elapsed}s`), '用时') : null,
                result?.threshold ? h('div', null, h('b', null, result.threshold.toFixed(2)), '阈值') : null,
              ),
              groups.length
                ? h(
                    'div',
                    { className: 'fs-row' },
                    h('span', { className: 'fs-label' }, '导出目录'),
                    h('input', {
                      className: 'fs-input fs-input-sm',
                      value: dest,
                      placeholder: '留空 = 本轮运行目录下的 export',
                      spellCheck: false,
                      onChange: (event) => setDest(event.target.value),
                    }),
                    h(
                      'button',
                      { className: 'fs-btn', disabled: busy, onClick: () => doExport(null) },
                      `全部导出（${groups.length} 组）`,
                    ),
                    jobId ? h('span', { className: 'fs-pill' }, `运行 ${jobId}`) : null,
                  )
                : null,
            )
          : null,

        groups.length
          ? h(
              'div',
              { className: 'fs-groups' },
              groups.map((group) =>
                h(GroupCard, {
                  key: group.id,
                  group,
                  jobId,
                  active: group.id === activeGroup,
                  onToggle: (id) => setActiveGroup(id === activeGroup ? null : id),
                }),
              ),
            )
          : null,

        active ? h(GroupDetail, { key: active.id + threshold, group: active, jobId, onExport: doExport, busy }) : null,

        result && result.noFace && result.noFace.length
          ? h(
              'div',
              { className: 'fs-card' },
              h('div', { className: 'fs-group-id' }, `未检测到人脸（${result.noFace.length}）`),
              h(
                'div',
                { className: 'fs-list' },
                result.noFace.map((rel) => h('div', { key: rel }, h('code', null, rel))),
              ),
            )
          : null,

        result && result.errors && result.errors.length
          ? h(
              'div',
              { className: 'fs-card' },
              h('div', { className: 'fs-group-id' }, `读取失败（${result.errors.length}）`),
              h(
                'div',
                { className: 'fs-list' },
                result.errors.map((item, index) => h('div', { key: index }, `${item.file} — ${item.error}`)),
              ),
            )
          : null,

        state && state.history && state.history.length > 1
          ? h(
              'div',
              { className: 'fs-card' },
              h('div', { className: 'fs-group-id' }, '最近的分析'),
              h(
                'div',
                { className: 'fs-history' },
                state.history.map((item) =>
                  h(
                    'button',
                    {
                      key: item.id,
                      onClick: async () => {
                        setJobId(item.id)
                        setError('')
                        setNotice('')
                        await loadResult(item.id)
                      },
                    },
                    `${item.id} · ${item.root} · ${item.groups ?? '?'} 组`,
                  ),
                ),
              ),
            )
          : null,
      )
    }

    function SidebarIcon() {
      return h(
        'svg',
        { width: 20, height: 20, viewBox: '0 0 20 20', fill: 'none', xmlns: 'http://www.w3.org/2000/svg', 'aria-hidden': 'true' },
        h('rect', { x: 2.4, y: 2.4, width: 6, height: 6, rx: 2, stroke: 'currentColor', strokeWidth: 1.4 }),
        h('rect', { x: 11.6, y: 2.4, width: 6, height: 6, rx: 2, stroke: 'currentColor', strokeWidth: 1.4 }),
        h('rect', { x: 2.4, y: 11.6, width: 6, height: 6, rx: 2, stroke: 'currentColor', strokeWidth: 1.4 }),
        h('circle', { cx: 14.6, cy: 14.6, r: 3, stroke: 'currentColor', strokeWidth: 1.4 }),
      )
    }

    return {
      inject: ['slots', 'layout'],
      apply(ctx) {
        ensureStyle()
        ctx.effect(() => {
          const disposer = ctx.slots.register(
            { name: 'main', key: PANEL_ID, order: 60 },
            Panel,
          )
          return () => {
            if (typeof disposer === 'function') disposer()
          }
        }, 'face-sorter: main panel')

        ctx.effect(() => {
          const disposer = ctx.slots.register(
            {
              name: 'sidebar.panellist',
              id: PANEL_ID,
              order: 60,
              label: () => '人脸分组',
            },
            SidebarIcon,
          )
          return () => {
            if (typeof disposer === 'function') disposer()
          }
        }, 'face-sorter: sidebar entry')
      },
    }
  },
})
