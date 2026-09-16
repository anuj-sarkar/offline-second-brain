import { useState, useEffect, useRef } from 'react'
import Sidebar from './components/Sidebar.jsx'
import TopNav from './components/TopNav.jsx'
import SettingsModal from './components/SettingsModal.jsx'
import MessageBlock from './components/MessageBlock.jsx'
import { getDocuments, getAvailableModels, askStream } from './api.js'

export default function App() {
  const [documents, setDocuments] = useState({})
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const [busy, setBusy] = useState(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [toasts, setToasts] = useState([])
  const [availableModels, setAvailableModels] = useState([])
  const [selectedDocFilter, setSelectedDocFilter] = useState([])
  const [settings, setSettings] = useState({
    strategy: "hybrid_no_rerank",
    topK: 5,
    model: "llama3.2:3b",
    temperature: 0.1,
    chunkSize: 1000,
    chunkOverlap: 200,
  })
  const scrollRef = useRef(null)

  const showToast = ({ type = 'info', message }) => {
    const id = Date.now() + Math.random()
    setToasts((prev) => [...prev, { id, type, message }])
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id))
    }, 4500)
  }

  const refreshDocuments = () => {
    getDocuments()
      .then(setDocuments)
      .catch(() => setDocuments({}))
  }

  const refreshModels = async () => {
    try {
      const data = await getAvailableModels()
      if (data && data.models && data.models.length > 0) {
        setAvailableModels(data.models)
        // If current model is not in detected models, update it
        if (!data.models.includes(settings.model)) {
          setSettings((prev) => ({ ...prev, model: data.models[0] }))
        }
      }
    } catch (e) {
      console.warn("Could not refresh models:", e)
    }
  }

  useEffect(() => {
    refreshDocuments()
    refreshModels()
  }, [])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })
  }, [messages])

  const handleToggleDocFilter = (name) => {
    setSelectedDocFilter((prev) => {
      const current = Array.isArray(prev) ? prev : (prev ? [prev] : [])
      return current.includes(name) ? current.filter((n) => n !== name) : [...current, name]
    })
  }

  const handleSelectAllDocs = (allNames) => {
    setSelectedDocFilter(allNames || [])
  }

  const handleClearDocFilter = () => {
    setSelectedDocFilter([])
  }

  const handleAsk = async () => {
    const question = input.trim()
    if (!question || busy) return
    setInput("")
    setBusy(true)

    const index = messages.length
    setMessages((prev) => [
      ...prev,
      { question, answer: "", chunks: [], report: null, streaming: true },
    ])

    const activeFilter =
      Array.isArray(selectedDocFilter) && selectedDocFilter.length > 0
        ? selectedDocFilter
        : null

    await askStream(
      {
        question,
        top_k: settings.topK,
        retrieval_strategy: settings.strategy,
        model_name: settings.model,
        temperature: settings.temperature,
        filename_filter: activeFilter,
      },
      {
        onSources: (chunks) => {
          setMessages((prev) => {
            const next = [...prev]
            next[index] = { ...next[index], chunks }
            return next
          })
        },
        onToken: (text) => {
          setMessages((prev) => {
            const next = [...prev]
            next[index] = { ...next[index], answer: next[index].answer + text }
            return next
          })
        },
        onDone: (report) => {
          setMessages((prev) => {
            const next = [...prev]
            next[index] = { ...next[index], report, streaming: false }
            return next
          })
          setBusy(false)
        },
        onError: (err) => {
          setMessages((prev) => {
            const next = [...prev]
            next[index] = {
              ...next[index],
              answer: `⚠️ ${err.message || 'Generation failed'}`,
              streaming: false,
            }
            return next
          })
          setBusy(false)
          showToast({ type: 'error', message: err.message || 'Generation failed' })
        },
      }
    )
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleAsk()
    }
  }

  const handleClearChat = () => {
    if (messages.length > 0 && window.confirm("Clear current conversation?")) {
      setMessages([])
    }
  }

  const selectedCount = Array.isArray(selectedDocFilter) ? selectedDocFilter.length : 0

  const getInputPlaceholder = () => {
    if (selectedCount === 0) return "Ask a question about your documents..."
    if (selectedCount === 1) return `Ask a question about ${selectedDocFilter[0]}...`
    return `Ask a question about the ${selectedCount} selected documents...`
  }

  return (
    <div className="app-shell">
      <Sidebar
        documents={documents}
        onReindexed={refreshDocuments}
        onClearChat={handleClearChat}
        onToast={showToast}
        selectedDocFilter={selectedDocFilter}
        onToggleDocFilter={handleToggleDocFilter}
        onSelectAllDocs={handleSelectAllDocs}
        onClearDocFilter={handleClearDocFilter}
      />

      <div className="main">
        <TopNav
          settings={settings}
          onOpenSettings={() => {
            refreshModels()
            setIsSettingsOpen(true)
          }}
          onReindexed={refreshDocuments}
          onToast={showToast}
          busy={busy}
          selectedDocFilter={selectedDocFilter}
          onToggleDocFilter={handleToggleDocFilter}
          onClearDocFilter={handleClearDocFilter}
        />

        <div className="chat-scroll" ref={scrollRef}>
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">🧠</div>
              <h2>Ask your library anything</h2>
              <p>Answers are grounded in your indexed documents, with verified citations.</p>
              {selectedCount > 0 ? (
                <div className="empty-doc-filter-indicator">
                  <span>🎯 Focusing answers specifically on:</span>
                  <strong>
                    {selectedCount === 1
                      ? selectedDocFilter[0]
                      : `${selectedCount} documents (${selectedDocFilter.join(', ')})`}
                  </strong>
                  <button onClick={handleClearDocFilter}>Search All Docs</button>
                </div>
              ) : (
                <div className="empty-shortcuts">
                  <span className="shortcut-tag">⚡ Fast local embeddings</span>
                  <span className="shortcut-tag">🔒 Zero data leakage</span>
                </div>
              )}
            </div>
          ) : (
            messages.map((entry, i) => <MessageBlock key={i} entry={entry} />)
          )}
        </div>

        <div className="input-bar">
          <div className="input-row">
            <input
              type="text"
              placeholder={getInputPlaceholder()}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={busy}
            />
            <button className="send-btn" onClick={handleAsk} disabled={busy || !input.trim()}>
              {busy ? "…" : "Ask"}
            </button>
          </div>
        </div>
      </div>

      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onSettingsChange={setSettings}
        availableModels={availableModels}
        onRefreshModels={refreshModels}
      />

      {/* Toast Notifications */}
      {toasts.length > 0 && (
        <div className="toast-container">
          {toasts.map((toast) => (
            <div key={toast.id} className={`toast toast-${toast.type}`}>
              <span className="toast-icon">
                {toast.type === 'success' ? '✓' : toast.type === 'error' ? '⚠' : 'ℹ'}
              </span>
              <span className="toast-text">{toast.message}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
