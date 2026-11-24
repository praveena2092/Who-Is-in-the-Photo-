async function request(url, options = {}) {
  let res
  try {
    res = await fetch(url, options)
  } catch {
    throw new Error('Cannot reach the app server. Is it still running?')
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch { /* not JSON */ }
    const err = new Error(detail)
    err.status = res.status
    throw err
  }
  return res.json()
}

export const getStatus = () => request('/api/status')
export const clearGallery = () => request('/api/gallery', { method: 'DELETE' })
export const listGallery = () => request('/api/gallery')

export function addGalleryPhotos(files) {
  const body = new FormData()
  // Send only the file name, not the folder path: the name is the person.
  files.forEach((f) => body.append('files', f, f.name))
  return request('/api/gallery/photos', { method: 'POST', body })
}

export function recognize(file, threshold) {
  const body = new FormData()
  body.append('file', file, file.name)
  return request(`/api/photos?threshold=${threshold}`, { method: 'POST', body })
}

export function rematch(photoId, threshold) {
  return request(`/api/photos/${photoId}/match`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ threshold }),
  })
}
