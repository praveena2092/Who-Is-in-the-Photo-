const KNOWN = '#16a34a'
const UNKNOWN = '#f59e0b'

/** Draw boxes + names onto the photo and save it as a JPEG. */
export async function downloadLabelledPhoto(result) {
  const img = await new Promise((res, rej) => {
    const i = new Image()
    i.onload = () => res(i)
    i.onerror = rej
    i.src = result.image_url
  })
  const W = img.naturalWidth, H = img.naturalHeight
  const c = document.createElement('canvas')
  c.width = W; c.height = H
  const g = c.getContext('2d')
  g.drawImage(img, 0, 0)

  const unit = Math.max(W, H) / 1000
  const font = Math.max(12, Math.round(16 * unit))
  g.font = `600 ${font}px system-ui, -apple-system, Segoe UI, Roboto, sans-serif`
  g.textBaseline = 'middle'

  for (const f of result.faces) {
    const x = f.box.x * W, y = f.box.y * H, w = f.box.w * W, h = f.box.h * H
    const color = f.status === 'known' ? KNOWN : UNKNOWN
    g.lineWidth = Math.max(2, 3 * unit)
    g.strokeStyle = color
    if (f.status !== 'known') g.setLineDash([8 * unit, 5 * unit])
    g.strokeRect(x, y, w, h)
    g.setLineDash([])

    const text = `${f.number}  ${f.name}`
    const pad = font * 0.45
    const tw = g.measureText(text).width + pad * 2
    const th = font + pad * 1.2
    let ty = y - th - 2 * unit
    if (ty < 0) ty = y + h + 2 * unit
    const tx = Math.min(Math.max(0, x + w / 2 - tw / 2), W - tw)
    g.fillStyle = color
    roundRect(g, tx, ty, tw, th, th / 3)
    g.fill()
    g.fillStyle = f.status === 'known' ? '#fff' : '#1f1300'
    g.fillText(text, tx + pad, ty + th / 2)
  }

  const blob = await new Promise((res) => c.toBlob(res, 'image/jpeg', 0.92))
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = 'labelled-group-photo.jpg'
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 5000)
}

function roundRect(g, x, y, w, h, r) {
  g.beginPath()
  g.moveTo(x + r, y)
  g.arcTo(x + w, y, x + w, y + h, r)
  g.arcTo(x + w, y + h, x, y + h, r)
  g.arcTo(x, y + h, x, y, r)
  g.arcTo(x, y, x + w, y, r)
  g.closePath()
}
