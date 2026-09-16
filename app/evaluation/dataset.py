"""
app/evaluation/dataset.py

Evaluation dataset for the Offline Second Brain retrieval system.

Ground-truth page assignments are built from document-structure knowledge
for the four indexed PDFs, cross-referenced with content confirmed during
Phases 5-12:

  - attention_is_all_you_need.pdf               (15 pages, 39 chunks)
  - a_fast_and_elitist_..._NSGA-II.pdf         (16 pages, 83 chunks)
  - a_multiobjective_...decomposition.pdf       (20 pages, 83 chunks)
  - krvea.pdf                                   (19 pages, 98 chunks)

Each EvalQuestion.relevant_pages is a list of (filename, page_number) pairs.
A retrieved chunk is counted as relevant if its page matches any listed pair.
Page-level (not chunk-level): a page can be split across multiple chunks, any
of which answering the question counts.

Query-type taxonomy (6 types, 35 questions total):
  definitional  - "What is X?" — tests abstract / intro retrieval
  comparative   - "How does X differ from Y?" — multi-doc discrimination
  factual       - Numbers, tables, algorithm parameters
  conceptual    - Mechanism explanations (mid-paper content, pages 3+)
  edge_case     - Formulas, adversarial jargon-free paraphrases
  multi_hop     - Reasoning that requires multiple sections or documents
"""

from dataclasses import dataclass

_ATT   = "attention_is_all_you_need.pdf"
_NSGA  = "a_fast_and_elitist_multiobjective_genetic_algorithm_NSGA-II.pdf"
_MEAD  = "a_multiobjective_evolutionary_algorithm_based_on_decomposition.pdf"
_KRVEA = "krvea.pdf"


@dataclass
class EvalQuestion:
    question: str
    relevant_pages: list[tuple[str, int]]
    query_type: str = ""
    notes: str = ""


EVAL_DATASET: list[EvalQuestion] = [

    # ==================================================================
    # DEFINITIONAL (6)
    # "What is X?" -- tests cold-start / abstract retrieval.
    # ==================================================================

    EvalQuestion(
        question="What is the main innovation of the Transformer architecture?",
        relevant_pages=[(_ATT, 1), (_ATT, 2)],
        query_type="definitional",
        notes="Phase 7 confirmed. Page 1 abstract, page 2 intro describe self-attention-only arch.",
    ),
    EvalQuestion(
        question="What is NSGA-II?",
        relevant_pages=[(_NSGA, 1)],
        query_type="definitional",
        notes="Canonical hard case Phases 9-11. Page 1 abstract. Historically retrieved poorly.",
    ),
    EvalQuestion(
        question="What is the MOEA/D algorithm?",
        relevant_pages=[(_MEAD, 1), (_MEAD, 2)],
        query_type="definitional",
        notes="Page 1 abstract defines decomposition-based MOO; page 2 introduces neighbourhood.",
    ),
    EvalQuestion(
        question="What is the KR-VEA algorithm and what problem does it solve?",
        relevant_pages=[(_KRVEA, 1), (_KRVEA, 2)],
        query_type="definitional",
        notes="Indexed Phase 11, never queried before. Page 1 abstract + page 2 motivation.",
    ),
    EvalQuestion(
        question="What is multi-head attention and why is it used instead of a single attention function?",
        relevant_pages=[(_ATT, 3), (_ATT, 4)],
        query_type="definitional",
        notes="Section 3.2 Multi-Head Attention pages 3-4. Motivation: multiple representation subspaces.",
    ),
    EvalQuestion(
        question="What is non-dominated sorting and what role does it play in NSGA-II?",
        relevant_pages=[(_NSGA, 2), (_NSGA, 3)],
        query_type="definitional",
        notes="Pages 2-3: fast non-dominated sort O(MN^2) replaces NSGA sharing function.",
    ),

    # ==================================================================
    # COMPARATIVE (6)
    # "How does X differ from Y?" -- tests multi-doc discrimination.
    # ==================================================================

    EvalQuestion(
        question="How does the Transformer encoder differ from the decoder in its use of attention?",
        relevant_pages=[(_ATT, 4), (_ATT, 5)],
        query_type="comparative",
        notes="Sections 3.2.2 (encoder self-attention) and 3.2.3 (enc-dec, masked decoder) pages 4-5.",
    ),
    EvalQuestion(
        question="What are the key differences between MOEA/D and NSGA-II in handling the Pareto front?",
        relevant_pages=[(_MEAD, 1), (_MEAD, 2)],
        query_type="comparative",
        notes="MOEA/D intro pages 1-2 explicitly contrasts decomposition vs dominance-based NSGA-II.",
    ),
    EvalQuestion(
        question="How does KR-VEA improve upon standard reference-vector-based evolutionary algorithms?",
        relevant_pages=[(_KRVEA, 1), (_KRVEA, 2)],
        query_type="comparative",
        notes="Pages 1-2: fixed reference vectors fail on irregular fronts -- KR-VEA motivation.",
    ),
    EvalQuestion(
        question="How does multi-head attention compare to a single attention function at full dimensionality?",
        relevant_pages=[(_ATT, 3)],
        query_type="comparative",
        notes="Page 3 Section 3.2: h heads of d_k/h dims vs single head at d_model explicitly compared.",
    ),
    EvalQuestion(
        question="How does NSGA-II selection differ from the original NSGA?",
        relevant_pages=[(_NSGA, 1), (_NSGA, 2)],
        query_type="comparative",
        notes="Pages 1-2: NSGA O(N^3) no elitism vs NSGA-II O(MN^2) elitist archive.",
    ),
    EvalQuestion(
        question="How does the Transformer compare to RNN and CNN architectures for sequence modeling?",
        relevant_pages=[(_ATT, 1), (_ATT, 2)],
        query_type="comparative",
        notes="Pages 1-2 contrast against RNN sequential computation and CNN limited receptive field.",
    ),

    # ==================================================================
    # FACTUAL (6)
    # Numbers, tables, parameters. Tests BM25 exact-match value.
    # ==================================================================

    EvalQuestion(
        question="What BLEU score did the Transformer achieve on English-to-German translation?",
        relevant_pages=[(_ATT, 8)],
        query_type="factual",
        notes="Table 2 page 8: base model 27.3 BLEU, big model 28.4 BLEU.",
    ),
    EvalQuestion(
        question="What were the training details and hyperparameters used for the base Transformer model?",
        relevant_pages=[(_ATT, 7)],
        query_type="factual",
        notes="Table 3 page 7: d_model=512, h=8, d_ff=2048, dropout, warmup_steps.",
    ),
    EvalQuestion(
        question="How many attention heads and what model dimension does the base Transformer use?",
        relevant_pages=[(_ATT, 7)],
        query_type="factual",
        notes="Table 3 page 7: h=8, d_model=512. Numeric late-page lookup vs heavy early pages.",
    ),
    EvalQuestion(
        question="What is the time complexity of the fast non-dominated sorting algorithm in NSGA-II?",
        relevant_pages=[(_NSGA, 3)],
        query_type="factual",
        notes="Algorithm 1 + O(MN^2) complexity proof on page 3.",
    ),
    EvalQuestion(
        question="How does NSGA-II perform on the ZDT benchmark test problems?",
        relevant_pages=[(_NSGA, 8), (_NSGA, 9)],
        query_type="factual",
        notes="ZDT1-ZDT6 GD and spread metrics on pages 8-9. Deep results section.",
    ),
    EvalQuestion(
        question="What scalarization approaches does MOEA/D support?",
        relevant_pages=[(_MEAD, 3)],
        query_type="factual",
        notes="Page 3 Section II: weighted sum, Tchebycheff, boundary intersection.",
    ),

    # ==================================================================
    # CONCEPTUAL (6)
    # Mechanism explanations. Relevant content is mid-paper (pages 3-7).
    # ==================================================================

    EvalQuestion(
        question="How does self-attention work?",
        relevant_pages=[(_ATT, 2)],
        query_type="conceptual",
        notes="Phase 4/9 confirmed. Page 2: query/key/value mapping definition.",
    ),
    EvalQuestion(
        question="What are the applications of attention in the Transformer model?",
        relevant_pages=[(_ATT, 5)],
        query_type="conceptual",
        notes="Section 3.2.3 Applications of Attention on page 5. Phase 4 confirmed.",
    ),
    EvalQuestion(
        question="How does positional encoding in the Transformer allow it to use word order without recurrence?",
        relevant_pages=[(_ATT, 6)],
        query_type="conceptual",
        notes="Section 3.5 Positional Encoding page 6: sine/cosine encoding + extrapolation rationale.",
    ),
    EvalQuestion(
        question="What is the role of the neighbourhood structure in MOEA/D and how does it speed up optimization?",
        relevant_pages=[(_MEAD, 2), (_MEAD, 3)],
        query_type="conceptual",
        notes="Pages 2-3: T closest weight vectors restrict mating, reducing inter-subproblem communication.",
    ),
    EvalQuestion(
        question="How does NSGA-II use the crowding distance operator during environmental selection?",
        relevant_pages=[(_NSGA, 5), (_NSGA, 6)],
        query_type="conceptual",
        notes="Algorithm 2 (crowding-distance) page 5; Algorithm 3 uses it as secondary sort criterion page 6.",
    ),
    EvalQuestion(
        question="How does KR-VEA adapt its reference vectors during optimization?",
        relevant_pages=[(_KRVEA, 4), (_KRVEA, 5)],
        query_type="conceptual",
        notes="Reference vector adaptation mechanism pages 4-5. Late-page AND new-document.",
    ),

    # ==================================================================
    # EDGE CASE (6)
    # Math formulas and adversarial jargon-free paraphrases.
    # ==================================================================

    EvalQuestion(
        question="What is the exact equation for Scaled Dot-Product Attention?",
        relevant_pages=[(_ATT, 3)],
        query_type="edge_case",
        notes="Equation 1 softmax(QK^T/sqrt(d_k))V on page 3. Case 7 math-extraction failure mode.",
    ),
    EvalQuestion(
        question="What is the crowding distance formula used in NSGA-II?",
        relevant_pages=[(_NSGA, 5)],
        query_type="edge_case",
        notes="Algorithm 2 page 5. Case 7 failure -- formula notation may be garbled by PDF extraction.",
    ),
    EvalQuestion(
        question="What does the ablation study reveal about reducing the number of attention heads?",
        relevant_pages=[(_ATT, 9)],
        query_type="edge_case",
        notes="Table 3 ablation Section 6.3 page 9: quality drops when h is reduced.",
    ),
    EvalQuestion(
        question="What are the limitations of self-attention for very long sequences according to the complexity table?",
        relevant_pages=[(_ATT, 4)],
        query_type="edge_case",
        notes="Table 1 'Why Self-Attention' page 4: O(n^2 d) quadratic cost flagged as bottleneck.",
    ),
    EvalQuestion(
        question=(
            "How does information from the input side get combined with "
            "what the output side is generating?"
        ),
        relevant_pages=[(_ATT, 5)],
        query_type="edge_case",
        notes=(
            "Jargon-free paraphrase of encoder-decoder attention. "
            "Phase 12 Case 6: dropped page 5 from rank 1 to rank 4 under dense-only."
        ),
    ),
    EvalQuestion(
        question=(
            "How do you pick the best solutions when optimizing multiple "
            "conflicting goals at the same time?"
        ),
        relevant_pages=[(_NSGA, 1), (_NSGA, 2)],
        query_type="edge_case",
        notes="Jargon-free paraphrase of Pareto-optimal / non-dominated sorting.",
    ),

    # ==================================================================
    # MULTI-HOP (5)
    # Synthesis across sections or documents. Top-5 must span multiple
    # relevant locations for full credit.
    # ==================================================================

    EvalQuestion(
        question=(
            "How do evolutionary algorithms and neural network architectures "
            "each approach the idea of parallelism in their design?"
        ),
        relevant_pages=[(_ATT, 2), (_NSGA, 2)],
        query_type="multi_hop",
        notes="Cross-doc. ATT p2: parallelism vs RNNs. NSGA p2: O(MN^2) sort. Top-5 must span both.",
    ),
    EvalQuestion(
        question="What are the computational complexity differences between attention mechanisms and genetic sorting?",
        relevant_pages=[(_ATT, 4), (_NSGA, 3)],
        query_type="multi_hop",
        notes="Cross-doc. ATT p4: Table 1. NSGA p3: O(MN^2) fast-sort.",
    ),
    EvalQuestion(
        question="Why does NSGA-II elitism prevent the loss of good solutions found in previous generations?",
        relevant_pages=[(_NSGA, 1), (_NSGA, 2)],
        query_type="multi_hop",
        notes="Requires connecting motivation (p1 intro) with mechanism (combined R_t population, p2).",
    ),
    EvalQuestion(
        question="What evidence do the NSGA-II authors present that their fast non-dominated sort improves on the original NSGA?",
        relevant_pages=[(_NSGA, 2), (_NSGA, 3)],
        query_type="multi_hop",
        notes="O(N^3) vs O(MN^2) argument spans related-work (p2) and algorithm proof (p3).",
    ),
    EvalQuestion(
        question=(
            "How does MOEA/D use weight vectors to generate Pareto-optimal solutions "
            "and why does this differ from NSGA-II?"
        ),
        relevant_pages=[(_MEAD, 3), (_MEAD, 4), (_NSGA, 1)],
        query_type="multi_hop",
        notes="Cross-doc. MOEA/D weight vector scalarization (pp 3-4) vs NSGA-II dominance ranking (p1).",
    ),
]
