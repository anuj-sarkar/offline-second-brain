import { useState } from 'react'
import { deleteDocument } from '../api.js'

export default function Sidebar({
  documents,
  onReindexed,
  onClearChat,
  onToast,
  selectedDocFilter = [],
  onToggleDocFilter,
  onSelectAllDocs,
  onClearDocFilter,
}) {
  const [deletingDoc, setDeletingDoc] = useState(null)
  const [filterQuery, setFilterQuery] = useState('')

  const docEntries = Object.entries(documents || {})
  const docCount = docEntries.length
  const totalChunks = docEntries.reduce((sum, [, count]) => sum + (Number(count) || 0), 0)

  const filteredDocs = docEntries.filter(([name]) =>
    name.toLowerCase().includes(filterQuery.toLowerCase().trim())
  )

  const selectedList = Array.isArray(selectedDocFilter)
    ? selectedDocFilter
    : (selectedDocFilter ? [selectedDocFilter] : [])
  const selectedCount = selectedList.length

  const isDocSelected = (name) => selectedList.includes(name)

  const handleDelete = async (name, e) => {
    e.stopPropagation()
    if (!window.confirm(`Delete "${name}" from indexed library?`)) return

    setDeletingDoc(name)
    try {
      await deleteDocument(name)
      if (onToast) onToast({ type: 'info', message: `Deleted "${name}"` })
      if (isDocSelected(name) && onToggleDocFilter) {
        onToggleDocFilter(name)
      }
      onReindexed()
    } catch (err) {
      if (onToast) onToast({ type: 'error', message: `Failed to delete: ${err.message}` })
    } finally {
      setDeletingDoc(null)
    }
  }

  const handleSelectAll = () => {
    if (!onSelectAllDocs) return
    const allNames = filteredDocs.map(([name]) => name)
    onSelectAllDocs(allNames)
  }

  const getDocIcon = (filename) => {
    const ext = filename.split('.').pop()?.toLowerCase()
    if (ext === 'pdf') return '📄'
    if (ext === 'md') return '📑'
    return '📝'
  }

  const renderHighlightedName = (name, query) => {
    if (!query.trim()) return name
    const q = query.trim().toLowerCase()
    const index = name.toLowerCase().indexOf(q)
    if (index === -1) return name

    const before = name.substring(0, index)
    const match = name.substring(index, index + q.length)
    const after = name.substring(index + q.length)

    return (
      <>
        {before}
        <mark className="search-highlight">{match}</mark>
        {after}
      </>
    )
  }

  const allFilteredSelected =
    filteredDocs.length > 0 && filteredDocs.every(([name]) => isDocSelected(name))

  return (
    <aside className="sidebar">
      {/* Brand Header */}
      <div className="sidebar-brand-group">
        <div className="brand-row">
          <div className="brand-logo-icon">🧠</div>
          <div>
            <div className="brand">Second Brain</div>
            <div className="brand-tag">Local · Private · Offline</div>
          </div>
        </div>
      </div>

      {/* Library Section */}
      <div className="sidebar-section sidebar-library-section">
        <div className="sidebar-section-header">
          <span className="sidebar-section-title">
            Library {docCount > 0 && <span className="count-pill">{docCount}</span>}
          </span>
          <span className="sidebar-chunk-total">{totalChunks} chunks</span>
        </div>

        {/* Active Document Filter Notification if set */}
        {selectedCount > 0 && (
          <div className="active-filter-banner">
            <div className="filter-banner-text">
              <span>🎯 Focus ({selectedCount}):</span>
              <strong title={selectedList.join(', ')}>
                {selectedCount === 1
                  ? selectedList[0]
                  : `${selectedCount} documents selected`}
              </strong>
            </div>
            <button
              type="button"
              className="clear-doc-filter-btn"
              onClick={onClearDocFilter}
              title="Clear selection and search across all documents"
            >
              ✕ All Docs
            </button>
          </div>
        )}

        {/* Document Search / Filter Bar */}
        {docCount > 0 && (
          <div className="sidebar-filter-wrapper">
            <div className="sidebar-filter-box">
              <span className="filter-icon">🔍</span>
              <input
                type="text"
                className="sidebar-filter-input"
                placeholder={`Search ${docCount} document${docCount > 1 ? 's' : ''}...`}
                value={filterQuery}
                onChange={(e) => setFilterQuery(e.target.value)}
              />
              {filterQuery && (
                <button
                  type="button"
                  className="filter-clear-btn"
                  onClick={() => setFilterQuery('')}
                  title="Clear filter"
                >
                  ✕
                </button>
              )}
            </div>
            <div className="filter-meta-row">
              {filterQuery && (
                <span className="filter-result-count">
                  Showing {filteredDocs.length} of {docCount}
                </span>
              )}
              <div className="batch-selection-actions">
                <button
                  type="button"
                  className="batch-action-link"
                  onClick={allFilteredSelected ? onClearDocFilter : handleSelectAll}
                  title={allFilteredSelected ? "Deselect all visible documents" : "Select all visible documents"}
                >
                  {allFilteredSelected ? "Deselect All" : "Select All"}
                </button>
                {selectedCount > 0 && !allFilteredSelected && (
                  <>
                    <span className="batch-action-sep">·</span>
                    <button
                      type="button"
                      className="batch-action-link"
                      onClick={onClearDocFilter}
                      title="Clear selection"
                    >
                      Clear
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        <div className="doc-list">
          {docCount === 0 ? (
            <div className="hint empty-library-hint">
              <p>No documents indexed yet.</p>
              <p className="hint-sub">
                Use the <strong>＋ Upload</strong> &amp; <strong>⚡ Index</strong> buttons in the top right to add documents.
              </p>
            </div>
          ) : filteredDocs.length === 0 ? (
            <div className="no-matches-box">
              <p>No documents match "{filterQuery}"</p>
              <button
                type="button"
                className="btn-ghost-small"
                onClick={() => setFilterQuery('')}
              >
                Clear Search
              </button>
            </div>
          ) : (
            filteredDocs.map(([name, count]) => {
              const selected = isDocSelected(name)
              return (
                <div
                  className={`doc-row ${selected ? 'doc-row-selected' : ''}`}
                  key={name}
                  onClick={() => onToggleDocFilter && onToggleDocFilter(name)}
                  title={
                    selected
                      ? `Selected for queries (click to unselect ${name})`
                      : `Click to include ${name} in query focus`
                  }
                >
                  <span className={`doc-checkbox ${selected ? 'checked' : ''}`}>
                    {selected ? '✓' : ''}
                  </span>
                  <span className="doc-icon">{getDocIcon(name)}</span>
                  <span className="doc-name">
                    {renderHighlightedName(name, filterQuery)}
                  </span>
                  <span className="doc-count">{count}c</span>
                  <button
                    className="doc-delete-btn"
                    title={`Delete ${name}`}
                    disabled={deletingDoc === name}
                    onClick={(e) => handleDelete(name, e)}
                  >
                    {deletingDoc === name ? '…' : '×'}
                  </button>
                </div>
              )
            })
          )}
        </div>
      </div>

      {/* Sidebar Footer */}
      <div className="sidebar-footer">
        {onClearChat && (
          <button className="sidebar-action-btn" onClick={onClearChat} title="Clear conversation history">
            🧹 Clear Chat
          </button>
        )}
        <div className="offline-badge">
          <span className="offline-dot"></span>
          <span>100% Offline &amp; Private</span>
        </div>
      </div>
    </aside>
  )
}