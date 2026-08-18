"""
app/evaluation/dataset.py

Phase 11 deliverable: a small evaluation dataset of questions whose
relevant source pages are KNOWN - built from real content we already
confirmed exists in these documents across earlier phases (Phase 5,
7, 9, 10 outputs), not invented speculatively.

Each question lists relevant_pages as (filename, page_number) pairs.
A retrieved chunk counts as relevant if it matches one of these -
deliberately page-level, not chunk-level, since a page's content can
be split across multiple chunks and any of them answering the
question should count.
"""

from dataclasses import dataclass


@dataclass
class EvalQuestion:
    question: str
    relevant_pages: list[tuple[str, int]]  # (filename, page_number)
    notes: str = ""  # why we believe this ground truth, for traceability


EVAL_DATASET: list[EvalQuestion] = [
    EvalQuestion(
        question="What is the main innovation of the Transformer architecture?",
        relevant_pages=[
            ("attention_is_all_you_need.pdf", 1),
            ("attention_is_all_you_need.pdf", 2),
        ],
        notes="Confirmed in Phase 7 - model correctly answered using page 2 content.",
    ),
    EvalQuestion(
        question="How does self-attention work?",
        relevant_pages=[("attention_is_all_you_need.pdf", 2)],
        notes="Confirmed in Phase 4/9 - page 2 contains the self-attention definition.",
    ),
    EvalQuestion(
        question="What is NSGA-II?",
        relevant_pages=[
            ("a_fast_and_elitist_multiobjective_genetic_algorithm_NSGA-II.pdf", 1),
        ],
        notes=("The known hard case from Phase 9/10 - abstract on page 1 defines it, "
               "but historically ranked poorly. This is the query we've been using "
               "to diagnose retrieval quality throughout this project."),
    ),
    EvalQuestion(
        question="What are the applications of attention in the Transformer model?",
        relevant_pages=[("attention_is_all_you_need.pdf", 5)],
        notes="Confirmed in Phase 4 - page 5, section 2.3, directly titled this.",
    ),
    EvalQuestion(
        question="How does NSGA-II handle constrained optimization problems?",
        relevant_pages=[
            ("a_fast_and_elitist_multiobjective_genetic_algorithm_NSGA-II.pdf", 1),
            ("a_fast_and_elitist_multiobjective_genetic_algorithm_NSGA-II.pdf", 11),
        ],
        notes="Page 1 abstract mentions constrained NSGA-II; page 11 has the constrained-domination definition (seen in Phase 10 hybrid results).",
    ),
]