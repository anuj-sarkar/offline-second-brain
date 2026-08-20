# Phase 12: RAG Failure Analysis

This document catalogs every real failure discovered while building
Offline Second Brain, in the format: **Reproduce → Why it happens →
Component responsible → Solution → Measured impact.**

Every case below was found through actual testing on real documents,
not invented speculatively - several were found by accident while
testing something else entirely, which is itself a useful lesson:
systematic testing surfaces failures that careful design alone
doesn't anticipate.

---

## Case 1: BM25 tokenizer failing on punctuation-adjacent query terms

**Reproduce:** Query `"What is NSGA-II?"` through hybrid search.
BM25's naive `text.lower().split()` tokenizer turned the last query
word into `"nsga-ii?"` (trailing `?` glued on), which never matched
`"nsga-ii"` as it appears in document text.

**Why it happens:** Whitespace-only tokenization doesn't strip
punctuation. A single unmatched token silently removed the one query
term that mattered, leaving only common stopwords (`"what"`, `"is"`)
to drive BM25 scoring.

**Component responsible:** `app/retrieval/hybrid_search.py`, `_tokenize()`.

**Solution:** Replaced whitespace splitting with a regex tokenizer
(`[a-z0-9]+(?:-[a-z0-9]+)*`) that strips surrounding punctuation
while preserving internal hyphens (so `"nsga-ii"` stays one token).

**Measured impact:** Confirmed via `test_tokenize_strips_trailing_punctuation`
- direct before/after unit test. Did not fully fix retrieval quality
on its own (see Case 5), but was a necessary prerequisite.

---

## Case 2: Structural low-content pages inflating BM25 scores

**Reproduce:** After fixing Case 1, hybrid search for `"What is NSGA-II?"`
still surfaced completely unrelated attention-visualization word-list
pages (`attention_is_all_you_need.pdf`, pages 14-15) at ranks 3 and 5.

**Why it happens:** These pages (812-818 characters) passed our
original `LOW_CONTENT_CHAR_THRESHOLD = 100` check, so they weren't
filtered from indexing. Their one-word-per-line format made them
very short "documents" by BM25's standards, and BM25's document-length
normalization inflated their score from a single stopword match ("is").

**Component responsible:** `app/ingestion/loader.py`, low-content
detection logic (only checked raw character count).

**Solution:** Added a structural heuristic (`_is_low_content`) checking
average characters per line - pages with 30+ lines averaging under
12 chars/line are flagged regardless of total character count.

**Measured impact:** Chunk count dropped 213 → 210 (3 pages correctly
excluded). Garbled pages confirmed gone from all subsequent hybrid
search results (Phase 9 re-test).

---

## Case 3: Boilerplate/footnote text diluting the abstract chunk

**Reproduce:** Even after Cases 1-2, the actual NSGA-II definitional
abstract (page 1) never appeared in dense-only top-5 results for
`"What is NSGA-II?"` across every test through Phase 9.

**Why it happens:** Page 1's extracted text interleaved the abstract
with unrelated boilerplate - IEEE running headers, "Manuscript
received..." author-affiliation footnotes, and copyright lines -
diluting the chunk's embedding with irrelevant content. A second,
different paper (`attention_is_all_you_need.pdf`) showed the same
underlying problem in a different boilerplate format (NeurIPS-style
`*Equal contribution` footnotes), confirming this generalizes across
publishers, not a one-document fluke.

**Component responsible:** `app/ingestion/loader.py` (no cleaning
step existed before Phase 10).

**Solution:** New `app/ingestion/cleaning.py` module, pattern-based
regex stripping of confirmed boilerplate patterns, applied before
chunking. Required TWO separate pattern sets (NeurIPS-style asterisk
footnotes vs. IEEE journal manuscript footnotes) - different venues
format boilerplate differently, so pattern-based cleaning doesn't
generalize across publishers without venue-specific patterns.

**Measured impact:** NSGA-II page 1 char count reduced 6061 → 5535
(~526 chars of noise removed). **Directly measurable retrieval
improvement:** page 1 went from absent-in-top-5 (every prior attempt)
to rank 4 in dense-only top-5 (similarity 0.6094), retrieving genuinely
definitional text for the first time. This is the one case where we
have a clean before/after showing a real fix.

---

## Case 4: Unicode character variants breaking pattern matching

**Reproduce:** The footnote-stripping regex (Case 3) initially used
a plain ASCII `*` in its character class and silently failed to match
real footnote markers.

**Why it happens:** PDF-extracted text used `∗` (Unicode asterisk
operator, U+2217) instead of the plain ASCII asterisk `*` for footnote
markers - visually near-identical, different code point. This is the
SAME underlying category of bug as the `/40 /51/41` superscript
garbling found back in Phase 2 (Big-O notation) - PDF fonts frequently
substitute visually-similar Unicode characters for typeset symbols,
and naive ASCII-only pattern matching misses them.

**Component responsible:** `app/ingestion/cleaning.py`,
`_FOOTNOTE_MARKER_PATTERN`.

**Solution:** Widened the character class to `[*∗†‡]`, explicitly
including the Unicode variant.

**Measured impact:** Confirmed via `test_strips_footnote_marker_blocks`
- failed before the fix, passed after. A recurring lesson worth
flagging for Phase 13+: any future text-pattern work on PDF-extracted
content should expect Unicode variants of common symbols.

---

## Case 5: Hybrid search underperforming dense-only on definitional queries (systematic, not anecdotal)

**Reproduce:** Phase 11's formal evaluation (5 questions, Recall@5/
Precision@5/MRR) showed dense-only outperforming hybrid on 2 of 3
metrics in aggregate:

| Metric | Dense | Hybrid |
|---|---|---|
| Recall@5 | 0.900 | 0.700 |
| Precision@5 | 0.280 | 0.320 |
| MRR | 0.717 | 0.600 |

The single biggest driver: `"What is NSGA-II?"` scored a complete
miss under hybrid (Recall@5=0.00) despite scoring perfectly under
dense-only (Recall@5=1.00) on the identical, cleaned corpus.

**Why it happens:** BM25 rewards raw term frequency. Experimental/
results-discussion pages that repeat "NSGA-II" many times score
higher under BM25 than the single, concise, one-time-mention
definitional abstract - the opposite of what a definitional query
actually needs. RRF fusion, working exactly as designed, faithfully
combines a strong dense signal with a misleading BM25 signal, and
the misleading signal wins.

**Component responsible:** Not a bug in `hybrid_search.py`'s
implementation - this is an inherent property of BM25 applied to
this query type, correctly exposed by the fusion working as intended.

**Solution (Implemented in Redesign):** Added local cross-encoder reranker (`flashrank` with `ms-marco-MiniLM-L-12-v2`) on top of the candidate pool from dense and BM25 searches. The cross-encoder jointly assesses query-passage relevance with full token cross-attention, bypassing term-frequency bias.

**Measured impact:** Formal evaluation showed `Hybrid + FlashRank Reranker` achieved **Recall@5 = 1.000 (100%)**, **MRR = 1.000 (100%)**, and **Precision@5 = 0.520**, resolving the "What is NSGA-II?" failure completely with rank #1 retrieval.

---

## Case 6: Terminology mismatch between query and document phrasing (Phase 12 live test)

**Reproduce:** Compared two phrasings of the same underlying question
against `attention_is_all_you_need.pdf`:

- Exact terminology: `"What is encoder-decoder attention?"` →
  page 5 (the correct section) ranked **#1**, similarity 0.8045.
- Paraphrased, avoiding the paper's vocabulary entirely:
  `"How does information from the input side get combined with what
  the output side is generating?"` → page 5 still appeared, but
  dropped to **rank 4** (similarity 0.6004).

**Why it happens:** Semantic embeddings alone can suffer from representation drift when domain-specific jargon is swapped with paraphrases.

**Solution (Implemented in Redesign):** 
1. Added asymmetric task prefixes (`search_document:` and `search_query:`) to `nomic-embed-text`.
2. Applied FlashRank cross-encoder reranker over hybrid candidate pool.
3. Multi-turn conversational query condensation prompt to rephrase contextual queries.

**Measured impact:** The correct encoder-decoder attention chunk on page 5 was boosted back to rank #1 with FlashRank cross-encoder scoring.

---

## Summary table

| # | Failure | Status | Evidence |
|---|---|---|---|
| 1 | BM25 tokenizer punctuation bug | Fixed | Unit test |
| 2 | Structural low-content pages inflating BM25 | Fixed | Chunk count + re-test |
| 3 | Boilerplate diluting definitional chunks | Fixed | Char count + retrieval rank improvement |
| 4 | Unicode footnote marker variants | Fixed | Unit test |
| 5 | Hybrid underperforming dense on definitional queries | Open, documented | Formal Phase 11 evaluation |
| 6 | Terminology mismatch degrades (not breaks) dense retrieval | Open, documented | Live comparison test |

Four of six cases were found, fixed, and verified with before/after
evidence. Two remain open by design - they're genuine, non-trivial
limitations rather than implementation bugs, and are better candidates
for future architectural work (reranking, query routing) than for a
quick patch.
