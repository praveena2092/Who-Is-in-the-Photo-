import { useCallback, useEffect, useRef, useState } from 'react'
import * as api from './api.js'
import { chunk, filesFromDrop, folderName, isImage } from './files.js'
import { downloadLabelledPhoto } from './download.js'

const BATCH = 6

export default function App() {
  const [status, setStatus] = useState(null)
  const [gallery, setGallery] = useState({ folder: null, people: [], reports: [], progress: null, error: null })
  const [photo, setPhoto] = useState({ file: null, result: null, loading: false, error: null })
  const [threshold, setThreshold] = useState(0.35)
  const [toast, setToast] = useState(null)

  // Poll until the models have loaded.
  useEffect(() => {
    let stop = false
    async function poll() {
      try {
        const s = await api.getStatus()
        if (stop) return
        setStatus(s)
        if (s.loading) setTimeout(poll, 800)
        else if (s.default_threshold) setThreshold(s.default_threshold)
      } catch (e) {
        if (!stop) { setStatus({ ready: false, error: e.message }); setTimeout(poll, 2000) }
      }
    }
    poll()
    return () => { stop = true }
  }, [])

  const notify = useCallback((msg) => {
    setToast(msg)
    clearTimeout(notify.t)
    notify.t = setTimeout(() => setToast(null), 2600)
  }, [])

  // ---- step 1: reference folder
  async function loadFolder(files, name) {
    const images = files.filter(isImage)
    if (!images.length) {
      setGallery((g) => ({ ...g, error: 'That folder has no photos in it (JPG, PNG, WEBP or HEIC).' }))
      return
    }
    setGallery({ folder: name, people: [], reports: [], progress: { done: 0, total: images.length }, error: null })
    try {
      await api.clearGallery()
      const reports = []
      let done = 0
      for (const batch of chunk(images, BATCH)) {
        const r = await api.addGalleryPhotos(batch)
        reports.push(...r.reports)
        done += batch.length
        setGallery((g) => ({ ...g, reports: [...reports], progress: { done, total: images.length } }))
      }
      const { people } = await api.listGallery()
      setGallery({ folder: name, people, reports, progress: null, error: null })
      if (photo.result) refreshMatch(threshold, photo.result.photo_id)
    } catch (e) {
      setGallery((g) => ({ ...g, progress: null, error: e.message }))
    }
  }

  // ---- step 2: group photo
  async function loadPhoto(file) {
    if (!file) return
    if (!isImage(file)) { setPhoto((p) => ({ ...p, error: 'Please choose a photo (JPG, PNG, WEBP or HEIC).' })); return }
    setPhoto({ file, result: null, loading: true, error: null })
    try {
      const result = await api.recognize(file, threshold)
      setPhoto({ file, result, loading: false, error: null })
      document.getElementById('results')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    } catch (e) {
      setPhoto({ file, result: null, loading: false, error: e.message })
    }
  }

  const matchTimer = useRef()
  function refreshMatch(t, photoId) {
    clearTimeout(matchTimer.current)
    matchTimer.current = setTimeout(async () => {
      try {
        const result = await api.rematch(photoId, t)
        setPhoto((p) => ({ ...p, result }))
      } catch (e) {
        setPhoto((p) => ({ ...p, error: e.message }))
      }
    }, 120)
  }

  function onThreshold(t) {
    setThreshold(t)
    if (photo.result) refreshMatch(t, photo.result.photo_id)
  }

  const ready = status?.ready
  const galleryReady = gallery.people.length > 0 && !gallery.progress

  return (
    <div className="app">
      <Header status={status} />
      <main>
        <section className="intro">
          <h1>Who's in the photo?</h1>
          <p>Pick the folder of everyone's photos, then add a group photo. We'll put a name on every face and mark anyone we don't know.</p>
        </section>

        <section className="steps">
          <FolderStep gallery={gallery} disabled={!ready} onFiles={loadFolder} />
          <PhotoStep photo={photo} disabled={!ready} needsGallery={!galleryReady} onFile={loadPhoto} />
        </section>

        {photo.result && (
          <Results
            key={photo.result.photo_id}
            result={photo.result}
            threshold={threshold}
            onThreshold={onThreshold}
            onNewPhoto={() => document.getElementById('photo-input')?.click()}
            notify={notify}
          />
        )}
      </main>
      <footer>Runs entirely on this computer. No photos leave your machine.</footer>
      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
  )
}

/* ------------------------------------------------------------------ header */

function Header({ status }) {
  let cls = 'pill pill-wait', text = 'Starting…'
  if (status?.ready) { cls = 'pill pill-ok'; text = 'Ready · offline' }
  else if (status?.error) { cls = 'pill pill-err'; text = 'Problem' }
  else if (status?.loading) { text = 'Loading face models…' }
  return (
    <header className="topbar">
      <div className="brand"><span className="logo" aria-hidden>◉</span> Face Finder</div>
      <span className={cls} title={status?.error || ''}>{text}</span>
      {status?.error && <div className="banner-err">{status.error}</div>}
    </header>
  )
}

/* -------------------------------------------------------------- drop zone */

function DropZone({ disabled, onDropFiles, children, className = '' }) {
  const [over, setOver] = useState(false)
  return (
    <div
      className={`drop ${over ? 'over' : ''} ${disabled ? 'disabled' : ''} ${className}`}
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setOver(true) }}
      onDragLeave={() => setOver(false)}
      onDrop={async (e) => {
        e.preventDefault(); setOver(false)
        if (disabled) return
        onDropFiles(await filesFromDrop(e.dataTransfer), e)
      }}
    >
      {children}
    </div>
  )
}

/* ---------------------------------------------------------- step 1: folder */

function FolderStep({ gallery, disabled, onFiles }) {
  const input = useRef()
  const [open, setOpen] = useState(false)
  const { people, reports, progress, error, folder } = gallery
  const problems = reports.filter((r) => r.status === 'no_face' || r.status === 'unreadable')
  const warnings = reports.filter((r) => r.status === 'multiple_faces')
  const done = people.length > 0 && !progress

  return (
    <div className={`card step ${done ? 'step-done' : ''}`}>
      <div className="step-head">
        <span className="step-num">{done ? '✓' : '1'}</span>
        <div>
          <h2>People folder</h2>
          <p className="muted">One photo per person. The file name is their name, e.g. <code>Priya Sharma.jpg</code></p>
        </div>
      </div>

      <input
        ref={input} type="file" hidden webkitdirectory="" directory="" multiple
        onChange={(e) => { const f = [...e.target.files]; e.target.value = ''; onFiles(f, folderName(f)) }}
      />

      {progress ? (
        <div className="progress-box">
          <div className="progress-label">Reading faces… {progress.done} of {progress.total}</div>
          <div className="bar"><div style={{ width: `${(progress.done / progress.total) * 100}%` }} /></div>
        </div>
      ) : done ? (
        <div className="gallery-summary">
          <div className="summary-line">
            <strong>{people.length} people</strong> from <span className="folder">📁 {folder}</span>
          </div>
          <div className="avatars" onClick={() => setOpen(true)} title="See everyone">
            {people.slice(0, 9).map((p) => <img key={p.id} src={p.thumb_url} alt={p.name} />)}
            {people.length > 9 && <span className="more">+{people.length - 9}</span>}
          </div>
          {problems.length > 0 && (
            <button className="link warn" onClick={() => setOpen(true)}>
              ⚠ {problems.length} photo{problems.length > 1 ? 's' : ''} had no usable face
            </button>
          )}
          <div className="row">
            <button className="btn ghost" onClick={() => setOpen(true)}>See everyone</button>
            <button className="btn ghost" onClick={() => input.current.click()}>Change folder</button>
          </div>
        </div>
      ) : (
        <DropZone disabled={disabled} onDropFiles={(files) => onFiles(files, 'Dropped folder')}>
          <div className="drop-icon">📁</div>
          <button className="btn primary big" disabled={disabled} onClick={() => input.current.click()}>Choose folder</button>
          <div className="muted small">or drag the folder here</div>
        </DropZone>
      )}
      {error && <div className="error">{error}</div>}
      {open && <PeopleModal people={people} problems={problems} warnings={warnings} onClose={() => setOpen(false)} />}
    </div>
  )
}

function PeopleModal({ people, problems, warnings, onClose }) {
  const [q, setQ] = useState('')
  useEffect(() => {
    const k = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [onClose])
  const shown = people.filter((p) => p.name.toLowerCase().includes(q.toLowerCase()))
  return (
    <div className="modal-back" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="People in folder">
        <div className="modal-head">
          <h3>{people.length} people in the folder</h3>
          <input className="search" placeholder="Search names…" value={q} onChange={(e) => setQ(e.target.value)} autoFocus />
          <button className="icon-btn" onClick={onClose} aria-label="Close">✕</button>
        </div>
        {(problems.length > 0 || warnings.length > 0) && (
          <div className="issues">
            {problems.map((r) => <div key={r.file} className="issue bad"><b>{r.file}</b> — {r.message} This person can't be recognised.</div>)}
            {warnings.map((r) => <div key={r.file} className="issue"><b>{r.file}</b> — {r.message}</div>)}
          </div>
        )}
        <div className="people-grid">
          {shown.map((p) => (
            <figure key={p.id}>
              <img src={p.thumb_url} alt="" />
              <figcaption>{p.name}</figcaption>
            </figure>
          ))}
        </div>
      </div>
    </div>
  )
}

/* ----------------------------------------------------------- step 2: photo */

function PhotoStep({ photo, disabled, needsGallery, onFile }) {
  const input = useRef()
  const [preview, setPreview] = useState(null)
  useEffect(() => {
    if (!photo.file) return setPreview(null)
    const u = URL.createObjectURL(photo.file)
    setPreview(u)
    return () => URL.revokeObjectURL(u)
  }, [photo.file])
  const off = disabled || needsGallery

  return (
    <div className={`card step ${photo.result ? 'step-done' : ''} ${needsGallery ? 'step-locked' : ''}`}>
      <div className="step-head">
        <span className="step-num">{photo.result ? '✓' : '2'}</span>
        <div>
          <h2>Group photo</h2>
          <p className="muted">A photo of up to about 10 people. Faces work best when they're clearly visible.</p>
        </div>
      </div>
      <input id="photo-input" ref={input} type="file" accept="image/*,.heic,.heif" hidden
        onChange={(e) => { onFile(e.target.files[0]); e.target.value = '' }} />

      {photo.loading ? (
        <div className="scanning">
          {preview && <img src={preview} alt="" />}
          <div className="scan-line" />
          <div className="scan-label">Finding faces…</div>
        </div>
      ) : (
        <DropZone disabled={off} onDropFiles={(files) => onFile(files.find(isImage) || files[0])}>
          <div className="drop-icon">🖼️</div>
          <button className="btn primary big" disabled={off} onClick={() => input.current.click()}>
            {photo.result ? 'Try another photo' : 'Upload group photo'}
          </button>
          <div className="muted small">{needsGallery ? 'Choose the people folder first' : 'or drag a photo here'}</div>
        </DropZone>
      )}
      {photo.error && <div className="error">{photo.error}</div>}
    </div>
  )
}

/* ----------------------------------------------------------------- results */

function Results({ result, threshold, onThreshold, onNewPhoto, notify }) {
  const [active, setActive] = useState(null)
  const [labels, setLabels] = useState(true)
  const { faces, summary } = result
  const listRefs = useRef({})

  function focusFace(i, fromPhoto) {
    setActive(i)
    if (fromPhoto) listRefs.current[i]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }

  async function copyList() {
    const text = faces.map((f) => `${f.number}. ${f.name}`).join('\n')
    try { await navigator.clipboard.writeText(text); notify('List copied') }
    catch { notify('Could not copy — your browser blocked it') }
  }

  function downloadCsv() {
    const rows = [['number', 'name', 'status', 'score'], ...faces.map((f) => [f.number, f.name, f.status, f.score])]
    const csv = rows.map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(',')).join('\n')
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
    a.download = 'who-is-in-the-photo.csv'
    a.click()
  }

  return (
    <section id="results" className="results">
      <div className="results-head">
        <div>
          <h2>Results</h2>
          <div className="chips">
            <span className="chip">{summary.total} {summary.total === 1 ? 'face' : 'faces'} found</span>
            <span className="chip chip-known">{summary.known} recognised</span>
            <span className="chip chip-unknown">{summary.unknown} unknown</span>
          </div>
        </div>
        <div className="tools">
          <Strictness value={threshold} onChange={onThreshold} />
          <label className="toggle"><input type="checkbox" checked={labels} onChange={(e) => setLabels(e.target.checked)} /> Names on photo</label>
          <button className="btn ghost" onClick={() => downloadLabelledPhoto(result).catch(() => notify('Download failed'))}>⬇ Labelled photo</button>
          <button className="btn ghost" onClick={onNewPhoto}>New photo</button>
        </div>
      </div>

      <div className="results-body">
        <div className="card photo-card">
          <PhotoView result={result} active={active} labels={labels} onHover={focusFace} />
        </div>

        <aside className="card list-card">
          <div className="list-head">
            <h3>Everyone in the photo</h3>
            <div className="row">
              <button className="btn tiny" onClick={copyList}>Copy</button>
              <button className="btn tiny" onClick={downloadCsv}>CSV</button>
            </div>
          </div>
          {faces.length === 0 ? (
            <div className="empty">No faces found in this photo. Try a sharper photo where faces are larger and facing the camera.</div>
          ) : (
            <ol className="face-list">
              {faces.map((f) => (
                <li
                  key={f.index}
                  ref={(el) => (listRefs.current[f.index] = el)}
                  className={`face-row ${f.status} ${active === f.index ? 'active' : ''}`}
                  onMouseEnter={() => setActive(f.index)}
                  onMouseLeave={() => setActive(null)}
                  onClick={() => setActive(f.index)}
                >
                  <span className="num">{f.number}</span>
                  <img src={f.thumb_url} alt="" />
                  <div className="who">
                    <div className="name">{f.name}</div>
                    {f.status === 'known' ? (
                      <div className={`conf conf-${f.confidence}`} title={`Similarity score ${f.score}`}>
                        {f.confidence === 'high' ? 'Strong match' : 'Likely match'}
                      </div>
                    ) : (
                      <div className="conf conf-unknown" title={f.candidates[0] ? `Closest: ${f.candidates[0].name} (${f.candidates[0].score})` : ''}>
                        Not in the folder{f.candidates[0] ? ` · closest: ${f.candidates[0].name}` : ''}
                      </div>
                    )}
                  </div>
                  <ScoreBar score={f.score} threshold={threshold} known={f.status === 'known'} />
                </li>
              ))}
            </ol>
          )}
        </aside>
      </div>
    </section>
  )
}

function ScoreBar({ score, threshold, known }) {
  // 0 .. 0.8 mapped to the bar; marker shows the current cut-off.
  const pct = (v) => `${Math.max(0, Math.min(1, v / 0.8)) * 100}%`
  return (
    <div className="scorebar" title={`Similarity ${score.toFixed(2)} (needs ${threshold.toFixed(2)})`}>
      <div className={`fill ${known ? 'k' : 'u'}`} style={{ width: pct(score) }} />
      <div className="cut" style={{ left: pct(threshold) }} />
    </div>
  )
}

function Strictness({ value, onChange }) {
  return (
    <label className="strict" title="How sure we must be before we put a name on a face">
      <span>Match strictness</span>
      <span className="small muted">Relaxed</span>
      <input type="range" min="0.2" max="0.6" step="0.01" value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))} />
      <span className="small muted">Strict</span>
    </label>
  )
}

function PhotoView({ result, active, labels, onHover }) {
  return (
    <div
      className="photo-wrap"
      // Keep the box overlay exactly on the image: size the wrapper to the photo's own aspect ratio.
      style={{ aspectRatio: `${result.width} / ${result.height}`, width: `min(100%, calc(78vh * ${result.width / result.height}))` }}
    >
      <img src={result.image_url} alt="Group photo" draggable={false} />
      {result.faces.map((f) => {
        const { x, y, w, h } = f.box
        const below = y < 0.07
        return (
          <div
            key={f.index}
            className={`box ${f.status} ${active === f.index ? 'active' : ''} ${active != null && active !== f.index ? 'dim' : ''}`}
            style={{ left: `${x * 100}%`, top: `${y * 100}%`, width: `${w * 100}%`, height: `${h * 100}%` }}
            onMouseEnter={() => onHover(f.index, true)}
            onMouseLeave={() => onHover(null)}
          >
            <span className="badge">{f.number}</span>
            {(labels || active === f.index) && (
              <span className={`tag ${below ? 'below' : ''}`}>{f.name}</span>
            )}
          </div>
        )
      })}
    </div>
  )
}
