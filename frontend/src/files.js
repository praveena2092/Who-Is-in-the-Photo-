const IMAGE_EXT = /\.(jpe?g|png|webp|bmp|tiff?|heic|heif)$/i

export const isImage = (file) => IMAGE_EXT.test(file.name) && !file.name.startsWith('.')

/** Files from a drag-and-drop, walking into dropped folders. */
export async function filesFromDrop(dataTransfer) {
  const items = [...(dataTransfer.items || [])]
  const entries = items.map((i) => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean)
  if (!entries.length) return [...(dataTransfer.files || [])]
  const out = []
  async function walk(entry) {
    if (entry.isFile) {
      out.push(await new Promise((res, rej) => entry.file(res, rej)))
    } else if (entry.isDirectory) {
      const reader = entry.createReader()
      // readEntries returns at most ~100 entries per call
      for (;;) {
        const batch = await new Promise((res, rej) => reader.readEntries(res, rej))
        if (!batch.length) break
        for (const e of batch) await walk(e)
      }
    }
  }
  for (const e of entries) await walk(e)
  return out
}

/** Folder name to show the user, from the first file's relative path. */
export function folderName(files, fallback = 'Selected folder') {
  const p = files.find((f) => f.webkitRelativePath)?.webkitRelativePath
  return p ? p.split('/')[0] : fallback
}

export function chunk(arr, n) {
  const out = []
  for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n))
  return out
}
