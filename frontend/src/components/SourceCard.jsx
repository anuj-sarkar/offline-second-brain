export default function SourceCard({ chunk }) {
  const score = chunk.similarity ?? chunk.fused_score ?? 0
  return (
    <div className="source-card">
      <div className="source-card-head">
        <span className="source-card-file">{chunk.filename} · p.{chunk.page_number}</span>
        <span className="source-card-score">{score.toFixed(3)}</span>
      </div>
      <div className="source-card-text">{chunk.text.slice(0, 320)}{chunk.text.length > 320 ? "…" : ""}</div>
    </div>
  )
}
