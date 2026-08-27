import { useState, useRef } from 'react'
import { uploadFiles, runIndexing } from '../api.js'

export default function TopNav({
  settings,
  onOpenSettings,
  onReindexed,
  onToast,
  busy,
  selectedDocFilter = [],
  onToggleDocFilter,
  onClearDocFilter,
}) {
  const [uploading, setUploading] = useState(false)
  const [indexing, setIndexing] = useState(false)
  const fileInputRef = useRef(null)

  const selectedList = Array.isArray(selectedDocFilter)
    ? selectedDocFilter
    : (selectedDocFilter ? [selectedDocFilter] : [])
  const selectedCount = selectedList.length

  const handleUploadClick = () => {
    if (uploading || indexing) return
    fileInputRef.current?.click()
  }

  const handleFileChange = async (e) => {
    const selectedFiles = Array.from(e.target.files || [])
    if (selectedFiles.length === 0) return

    setUploading(true)
    onToast({ type: 'info', message: `Uploading ${selectedFiles.length} file(s)...` })

    try {
      const res = await uploadFiles(selectedFiles)
      const count = res.saved?.length || selectedFiles.length
      onToast({
        type: 'success',
        message: `Saved ${count} file(s). Click 'Index' to index them into vectorstore.`,
      })
    } catch (err) {
      onToast({ type: 'error', message: `Upload failed: ${err.message}` })
    } finally {
      setUploading(false)
      // reset file input so the same files can be re-selected if needed
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleIndexClick = async () => {
    if (indexing || uploading || busy) return

    setIndexing(true)
    onToast({ type: 'info', message: 'Ingesting & indexing documents into vectorstore...' })

    try {
      const result = await runIndexing(settings.chunkSize, settings.chunkOverlap)
      onToast({
        type: 'success',
        message: `Indexed ${result.chunks_indexed} chunks from ${result.pages_processed} pages.`,
      })
      onReindexed()
    } catch (err) {
      onToast({ type: 'error', message: `Indexing failed: ${err.message}` })
    } finally {
      setIndexing(false)
    }
  }

  return (
    <header className="top-nav">
      <div className="top-nav-left">
        <div className="status-pill" title="Current Engine Configuration">
          <span className="status-dot"></span>
          <span className="pill-model">{settings.model}</span>
          <span className="pill-divider">·</span>
          <span className="pill-strategy">{settings.strategy} (k={settings.topK})</span>
        </div>

        {selectedCount === 1 && (
          <div className="top-doc-filter-badge" title={`Querying only within ${selectedList[0]}`}>
            <span className="filter-badge-icon">🎯</span>
            <span className="filter-badge-name">{selectedList[0]}</span>
            <button
              type="button"
              className="filter-badge-close"
              onClick={onClearDocFilter}
              title="Clear document filter (search all documents)"
            >
              ✕
            </button>
          </div>
        )}

        {selectedCount === 2 && (
          <div className="top-doc-filter-group">
            {selectedList.map((docName) => (
              <div key={docName} className="top-doc-filter-badge" title={`Filtering queries to ${docName}`}>
                <span className="filter-badge-icon">🎯</span>
                <span className="filter-badge-name">{docName}</span>
                <button
                  type="button"
                  className="filter-badge-close"
                  onClick={() => (onToggleDocFilter ? onToggleDocFilter(docName) : onClearDocFilter())}
                  title={`Remove ${docName} from filter`}
                >
                  ✕
                </button>
              </div>
            ))}
            <button
              type="button"
              className="top-doc-filter-clear-all"
              onClick={onClearDocFilter}
              title="Clear all document filters (search all documents)"
            >
              ✕ Clear All
            </button>
          </div>
        )}

        {selectedCount > 2 && (
          <div
            className="top-doc-filter-badge"
            title={`Querying within ${selectedCount} selected documents:\n${selectedList.join('\n')}`}
          >
            <span className="filter-badge-icon">🎯</span>
            <span className="filter-badge-name">{selectedCount} documents selected</span>
            <button
              type="button"
              className="filter-badge-close"
              onClick={onClearDocFilter}
              title="Clear document filter (search all documents)"
            >
              ✕
            </button>
          </div>
        )}
      </div>

      <div className="top-nav-actions">
        {/* Hidden file input */}
        <input
          type="file"
          ref={fileInputRef}
          multiple
          accept=".pdf,.txt,.md"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />

        {/* Upload button */}
        <button
          className="nav-btn nav-btn-upload"
          onClick={handleUploadClick}
          disabled={uploading || indexing}
          title="Upload PDF, TXT, or MD documents"
        >
          {uploading ? (
            <>
              <span className="nav-spinner"></span>
              <span>Uploading...</span>
            </>
          ) : (
            <>
              <span className="nav-btn-icon">＋</span>
              <span>Upload</span>
            </>
          )}
        </button>

        {/* Index button */}
        <button
          className="nav-btn nav-btn-index"
          onClick={handleIndexClick}
          disabled={indexing || uploading || busy}
          title="Index uploaded documents with embeddings"
        >
          {indexing ? (
            <>
              <span className="nav-spinner"></span>
              <span>Indexing...</span>
            </>
          ) : (
            <>
              <span className="nav-btn-icon">⚡</span>
              <span>Index</span>
            </>
          )}
        </button>

        {/* Settings button on the top right */}
        <button
          className="nav-btn nav-btn-settings"
          onClick={onOpenSettings}
          title="Engine & Retrieval Settings"
        >
          <span className="nav-btn-icon">⚙️</span>
          <span>Settings</span>
        </button>
      </div>
    </header>
  )
}
