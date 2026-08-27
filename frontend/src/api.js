const BASE_URL = "http://localhost:8000"

export async function getDocuments() {
  const res = await fetch(`${BASE_URL}/api/documents`)
  if (!res.ok) throw new Error("Failed to load documents")
  return res.json()
}

export async function getAvailableModels() {
  try {
    const res = await fetch(`${BASE_URL}/api/models`)
    if (!res.ok) throw new Error("Failed to load models")
    return res.json()
  } catch (err) {
    console.warn("Could not fetch models from backend:", err)
    return { models: ["llama3.2:3b"], default: "llama3.2:3b" }
  }
}

export async function uploadFiles(files) {
  const formData = new FormData()
  for (const f of files) formData.append("files", f)
  const res = await fetch(`${BASE_URL}/api/upload`, { method: "POST", body: formData })
  if (!res.ok) throw new Error("Upload failed")
  return res.json()
}

export async function runIndexing(chunkSize, chunkOverlap) {
  const res = await fetch(
    `${BASE_URL}/api/index?chunk_size=${chunkSize}&chunk_overlap=${chunkOverlap}`,
    { method: "POST" }
  )
  if (!res.ok) throw new Error("Indexing failed")
  return res.json()
}

export async function deleteDocument(docId) {
  const res = await fetch(`${BASE_URL}/api/documents/${encodeURIComponent(docId)}`, {
    method: "DELETE",
  })
  if (!res.ok) throw new Error("Failed to delete document")
  return res.json()
}

/**
 * Streams an answer via ND-JSON (newline-delimited JSON), calling
 * the provided callbacks as each event type arrives.
 */
export async function askStream(params, { onSources, onToken, onDone, onError }) {
  try {
    const res = await fetch(`${BASE_URL}/api/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    })
    if (!res.ok || !res.body) {
      const errorText = await res.text().catch(() => "")
      throw new Error(`Request failed (${res.status}): ${errorText || res.statusText}`)
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ""

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      const lines = buffer.split("\n")
      buffer = lines.pop() // keep incomplete last line for next chunk

      for (const line of lines) {
        if (!line.trim()) continue
        try {
          const event = JSON.parse(line)
          if (event.type === "sources") onSources(event.chunks)
          else if (event.type === "token") onToken(event.text)
          else if (event.type === "done") onDone(event.citation_report)
          else if (event.type === "error") {
            onError(new Error(event.error))
            return
          }
        } catch (parseErr) {
          console.warn("Failed to parse event line:", line, parseErr)
        }
      }
    }
  } catch (err) {
    onError(err)
  }
}
