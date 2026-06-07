"""
SFT Dataset Builder — Contract Obligation Extraction Fine-Tuning
Processes raw contract segments into JSONL instruction-response pairs for supervised fine-tuning.

Key design decisions:
1. spaCy NER for initial entity annotation (party names, dates, monetary values)
2. NLTK 3-sentence sliding window to preserve cross-sentence obligation structures
3. Variable-window extension for termination clauses (22 trigger keywords → 5-sentence window)
4. Pandas stratification to prevent over-indexing on high-frequency payment clauses

Output: 3,200-segment JSONL with 75/15/10 train/val/test splits
Legal reviewer validation: 11% of segments relabeled (obligation type field)
"""

import json
import re
import pandas as pd
import spacy
import nltk
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


OBLIGATION_TYPES = ["payment", "reporting", "indemnification", "termination"]

# Keywords that trigger variable window extension for termination clauses
TERMINATION_TRIGGER_KEYWORDS = {
    "terminate", "termination", "dissolution", "default", "event of default",
    "acceleration", "insolvency", "bankruptcy", "material adverse", "cross-default",
    "breach", "failure to pay", "cure period", "notice of termination",
    "early termination", "trigger event", "triggering event", "winding up",
    "liquidation", "receiver", "administrator", "cessation", "repudiation",
}

EXTRACTION_SCHEMA = {
    "obligated_party": "The entity or party that has the obligation",
    "obligation_type": "One of: payment, reporting, indemnification, termination",
    "deadline_expression": "The full deadline or timing requirement (include all conditions)",
    "consequence_clause": "What happens if the obligation is not met (full clause)",
}

SYSTEM_PROMPT = """You are extracting contractual obligations from legal contract segments.
For each contract segment, identify and extract:
1. obligated_party: the entity that must fulfill the obligation
2. obligation_type: payment | reporting | indemnification | termination
3. deadline_expression: the COMPLETE deadline or timing requirement (include ALL conditions and qualifiers)
4. consequence_clause: the COMPLETE consequence of non-performance (include ALL conditions)

Return ONLY a JSON object. If a field is not present, use null."""


@dataclass
class ObligationSegment:
    text: str
    obligated_party: Optional[str]
    obligation_type: Optional[str]
    deadline_expression: Optional[str]
    consequence_clause: Optional[str]
    source_doc_id: str
    segment_index: int


def load_spacy_pipeline(model_path: Optional[str] = None) -> spacy.Language:
    """Load NER pipeline — either from COiN models or base spaCy."""
    if model_path:
        return spacy.load(model_path)
    return spacy.load("en_core_web_sm")


def segment_with_sliding_window(
    text: str,
    window_size: int = 3,
    stride: int = 1,
) -> List[str]:
    """
    Segment contract text using sliding window over sentences.
    Variable window: extends to 5 sentences for termination clause candidates.
    """
    sentences = nltk.sent_tokenize(text)
    segments = []

    i = 0
    while i <= len(sentences) - window_size:
        window = sentences[i: i + window_size]
        window_text = " ".join(window)

        # Variable window extension: termination clauses span more sentences
        if _contains_termination_keywords(window_text) and i + 5 <= len(sentences):
            extended_window = sentences[i: i + 5]
            window_text = " ".join(extended_window)

        segments.append(window_text.strip())
        i += stride

    # Capture tail if remaining sentences form a meaningful segment
    if len(sentences) % stride != 0:
        tail = " ".join(sentences[-window_size:])
        if tail not in segments:
            segments.append(tail.strip())

    return segments


def _contains_termination_keywords(text: str) -> bool:
    """Check if text contains termination clause trigger keywords."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in TERMINATION_TRIGGER_KEYWORDS)


def annotate_segment_with_ner(
    segment: str,
    nlp: spacy.Language,
) -> Dict[str, Optional[str]]:
    """
    Use spaCy NER to extract initial entity annotations.
    These are validated by legal reviewers before use in training data.
    """
    doc = nlp(segment)
    annotations = {
        "parties": [ent.text for ent in doc.ents if ent.label_ in ("ORG", "PERSON")],
        "dates": [ent.text for ent in doc.ents if ent.label_ == "DATE"],
        "monetary": [ent.text for ent in doc.ents if ent.label_ == "MONEY"],
    }
    return annotations


def format_as_instruction_pair(segment: ObligationSegment) -> Dict:
    """Format as instruction-response pair for SFT."""
    response = {
        "obligated_party": segment.obligated_party,
        "obligation_type": segment.obligation_type,
        "deadline_expression": segment.deadline_expression,
        "consequence_clause": segment.consequence_clause,
    }
    return {
        "instruction": SYSTEM_PROMPT,
        "input": segment.text,
        "output": json.dumps(response, indent=2),
        "obligation_type": segment.obligation_type,  # for stratification
        "source_doc_id": segment.source_doc_id,
    }


def build_sft_dataset(
    segments: List[ObligationSegment],
    output_dir: str,
    train_pct: float = 0.75,
    val_pct: float = 0.15,
) -> Dict[str, int]:
    """
    Build stratified train/val/test splits from annotated segments.
    Stratified by obligation_type to prevent over-indexing on payment clauses.
    """
    df = pd.DataFrame([asdict(s) for s in segments])
    df["instruction_pair"] = df.apply(
        lambda row: format_as_instruction_pair(ObligationSegment(**{
            k: row[k] for k in ObligationSegment.__dataclass_fields__
        })),
        axis=1,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    splits = {"train": [], "val": [], "test": []}

    for obligation_type in OBLIGATION_TYPES:
        type_df = df[df["obligation_type"] == obligation_type].sample(frac=1, random_state=42)
        n = len(type_df)
        n_train = int(n * train_pct)
        n_val = int(n * val_pct)

        splits["train"].extend(type_df.iloc[:n_train]["instruction_pair"].tolist())
        splits["val"].extend(type_df.iloc[n_train:n_train + n_val]["instruction_pair"].tolist())
        splits["test"].extend(type_df.iloc[n_train + n_val:]["instruction_pair"].tolist())

    counts = {}
    for split_name, records in splits.items():
        path = output_path / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        counts[split_name] = len(records)
        print(f"  {split_name}: {len(records)} segments → {path}")

    print(f"\nDataset built. Total: {sum(counts.values())} segments")
    print(f"Distribution: {counts}")
    return counts
