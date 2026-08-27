import { useState } from 'react'
import SourceCard from './SourceCard.jsx'

export default function MessageBlock({ entry }) {
  const [showSources, setShowSources] = useState(false)
  const { question, answer, chunks, report, streaming } = entry

  return (
    <div className="message-block">
      <div className="msg-question">{question}</div>
      <div className="msg-answer">
        {answer}
        {streaming && <span className="cursor-blink">▍</span>}
      </div>

      {!streaming && report && report.has_any_citations && (
        <div className={`citation-badge ${report.all_verified ? "verified" : "warning"}`}>
          {report.all_verified
            ? "✓ All citations verified"
            : `⚠ ${report.fabricated_count} unverified citation(s)`}
        </div>
      )}

      {chunks && chunks.length > 0 && (
        <>
          <button className="sources-toggle" onClick={() => setShowSources(!showSources)}>
            {showSources ? "Hide" : "Show"} {chunks.length} source{chunks.length !== 1 ? "s" : ""}
          </button>
          {showSources && (
            <div className="source-cards">
              {chunks.map((c, i) => <SourceCard key={i} chunk={c} />)}
            </div>
          )}
        </>
      )}
    </div>
  )
}
