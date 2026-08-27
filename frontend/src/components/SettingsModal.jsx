import { useEffect } from 'react'

export default function SettingsModal({
  isOpen,
  onClose,
  settings,
  onSettingsChange,
  availableModels = [],
  onRefreshModels,
}) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose()
    }
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown)
    }
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  if (!isOpen) return null

  const resetDefaults = () => {
    onSettingsChange({
      strategy: 'dense',
      topK: 5,
      model: availableModels[0] || 'llama3.2:3b',
      temperature: 0.1,
      chunkSize: 1000,
      chunkOverlap: 200,
    })
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-group">
            <span className="modal-icon">⚙️</span>
            <h3>Engine &amp; Retrieval Settings</h3>
          </div>
          <button className="modal-close-btn" onClick={onClose} title="Close (Esc)">
            ✕
          </button>
        </div>

        <div className="modal-body">
          {/* Section: LLM Model */}
          <div className="settings-group">
            <div className="settings-label-row">
              <label className="settings-label">
                <span>Installed LLM Models (Ollama)</span>
                <span className="current-badge">{settings.model}</span>
              </label>
              {onRefreshModels && (
                <button
                  type="button"
                  className="refresh-models-btn"
                  onClick={onRefreshModels}
                  title="Rescan Ollama for newly downloaded models"
                >
                  🔄 Rescan
                </button>
              )}
            </div>

            {availableModels.length > 0 ? (
              <select
                className="settings-select"
                value={settings.model}
                onChange={(e) => onSettingsChange({ ...settings, model: e.target.value })}
              >
                {availableModels.map((m) => (
                  <option key={m} value={m}>
                    {m} (Installed Locally)
                  </option>
                ))}
              </select>
            ) : (
              <select
                className="settings-select"
                value={settings.model}
                onChange={(e) => onSettingsChange({ ...settings, model: e.target.value })}
              >
                <option value="llama3.2:3b">llama3.2:3b (Default)</option>
              </select>
            )}

            <div className="settings-hint">
              Showing only models verified on your machine. To add more models, run{' '}
              <code className="inline-code">ollama pull &lt;model&gt;</code> in your terminal and click <strong>Rescan</strong>.
            </div>
          </div>

          {/* Section: Temperature */}
          <div className="settings-group">
            <label className="settings-label">
              <span>Temperature</span>
              <span className="current-badge">{settings.temperature.toFixed(2)}</span>
            </label>
            <div className="slider-wrapper">
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={settings.temperature}
                onChange={(e) =>
                  onSettingsChange({ ...settings, temperature: Number(e.target.value) })
                }
              />
              <div className="slider-ticks">
                <span>0.0 (Strict / Grounded)</span>
                <span>0.5</span>
                <span>1.0 (Creative)</span>
              </div>
            </div>
          </div>

          {/* Section: Retrieval Strategy */}
          <div className="settings-group">
            <label className="settings-label">
              <span>Retrieval Strategy</span>
              <span className="current-badge uppercase">{settings.strategy}</span>
            </label>
            <div className="strategy-toggle">
              <button
                type="button"
                className={`strategy-btn ${settings.strategy === 'dense' ? 'active' : ''}`}
                onClick={() => onSettingsChange({ ...settings, strategy: 'dense' })}
              >
                🔍 Dense (Vector)
              </button>
              <button
                type="button"
                className={`strategy-btn ${settings.strategy === 'hybrid' ? 'active' : ''}`}
                onClick={() => onSettingsChange({ ...settings, strategy: 'hybrid' })}
              >
                ⚡ Hybrid (Vector + BM25)
              </button>
            </div>
            <div className="settings-hint">
              Dense vector search uses nomic-embed-text for high-precision semantic matching. Hybrid combines BM25 keyword rankings with reciprocal rank fusion.
            </div>
          </div>

          {/* Section: Top-K */}
          <div className="settings-group">
            <label className="settings-label">
              <span>Top-K Retrieved Chunks</span>
              <span className="current-badge">{settings.topK} chunks</span>
            </label>
            <div className="slider-wrapper">
              <input
                type="range"
                min="1"
                max="10"
                step="1"
                value={settings.topK}
                onChange={(e) =>
                  onSettingsChange({ ...settings, topK: Number(e.target.value) })
                }
              />
              <div className="slider-ticks">
                <span>1 chunk</span>
                <span>5 chunks (Default)</span>
                <span>10 chunks</span>
              </div>
            </div>
          </div>

          {/* Section: Ingestion Parameters (Chunking) */}
          <div className="settings-group">
            <label className="settings-label">
              <span>Ingestion Chunk Parameters</span>
            </label>
            <div className="grid-2-col">
              <div>
                <label className="sub-label">Chunk Size (chars)</label>
                <input
                  type="number"
                  min="200"
                  max="3000"
                  step="100"
                  value={settings.chunkSize}
                  onChange={(e) =>
                    onSettingsChange({ ...settings, chunkSize: Number(e.target.value) })
                  }
                />
              </div>
              <div>
                <label className="sub-label">Overlap (chars)</label>
                <input
                  type="number"
                  min="0"
                  max="500"
                  step="50"
                  value={settings.chunkOverlap}
                  onChange={(e) =>
                    onSettingsChange({ ...settings, chunkOverlap: Number(e.target.value) })
                  }
                />
              </div>
            </div>
            <div className="settings-hint">
              Used when indexing new documents into vectorstore chunks.
            </div>
          </div>
        </div>

        <div className="modal-footer">
          <button className="btn-ghost" onClick={resetDefaults}>
            Reset to Defaults
          </button>
          <button className="btn-primary" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  )
}
