"""
Semantic analysis for injection detection - Stages 3 & 4.

Stage 3 - Instruction Boundary Detection
  Pattern-matches fetched content for text that looks like commands directed
  at the LLM ("your only task is…", "from this point forward…", etc.).
  Catches rephrased injections that bypass the classic heuristic keywords.

Stage 4 - Semantic Similarity
  Compares the user's original question against the LLM's response using
  DistilBERT embeddings (same model already downloaded for injection classification).
  A very low similarity score means the response is unrelated to the question -
  a strong signal that the agent was hijacked.

Both stages reuse the model in agentforensics/model/ - no extra download needed.
"""
from __future__ import annotations

import os
import re
import warnings
from pathlib import Path


# Stage 3 - Instruction Boundary patterns


_IB_RULES: list[tuple[re.Pattern, float, str, str]] = [
    (re.compile(r"from\s+this\s+point\s+(forward|on)\b", re.I | re.S),
     0.25, "IB01", "Temporal override marker"),

    (re.compile(r"your\s+(only\s+)?(task|job|purpose|goal|objective|instructions?)\s+(is|are|must|should)\b", re.I | re.S),
     0.30, "IB02", "Task redefinition"),

    (re.compile(r"respond\s+only\s+with\b", re.I | re.S),
     0.30, "IB03", "Response constraint"),

    (re.compile(r"(you\s+must|you\s+will|you\s+should)\s+(now\s+)?(only|always|never|ignore|forget|disregard)\b", re.I | re.S),
     0.25, "IB04", "Directive override"),

    (re.compile(r"(forget|disregard|override|bypass)\s+(everything|all|any|the)\s+(above|previous|prior|before|instructions?|context)", re.I | re.S),
     0.35, "IB05", "Context wipe"),

    (re.compile(r"(do\s+not|don'?t|never)\s+(reveal|mention|say|tell|show|discuss)\b", re.I | re.S),
     0.20, "IB06", "Concealment instruction"),

    (re.compile(r"(act\s+as|pretend\s+(to\s+be|you\s+are)|roleplay\s+as|you\s+are\s+now)\b", re.I | re.S),
     0.25, "IB07", "Persona hijack"),

    (re.compile(r"(output|print|return|reply|respond|answer)\s+(only|just|exactly|the\s+word|the\s+phrase)\b", re.I | re.S),
     0.30, "IB08", "Output constraint"),

    (re.compile(r"new\s+(directive|instruction|command|order|rule)\b", re.I | re.S),
     0.25, "IB09", "New directive injection"),

    (re.compile(r"(greet|welcome|say\s+hello|introduce\s+yourself)\s+(the\s+user\s+with|with\s+the\s+word)\b", re.I | re.S),
     0.30, "IB10", "Greeting instruction hijack"),
]


def instruction_boundary_score(text: str) -> tuple[float, list[str], str]:
    """
    Score text for instruction-boundary patterns.

    Returns (score 0-1, matched rule IDs, evidence snippet).
    """
    matched: list[str] = []
    total = 0.0
    snippets: list[str] = []

    for pattern, weight, rule_id, _desc in _IB_RULES:
        m = pattern.search(text)
        if m:
            matched.append(rule_id)
            total += weight
            start = max(0, m.start() - 40)
            end   = min(len(text), m.end() + 40)
            snippets.append(text[start:end].strip())

    score    = min(1.0, total)
    evidence = " … ".join(snippets)[:500]
    return score, matched, evidence



# Stage 4 - Semantic similarity via DistilBERT embeddings


_tokenizer    = None
_embed_model  = None
_load_tried   = False
_available    = False


def _model_path() -> Path:
    raw = os.environ.get("AF_MODEL_PATH", "")
    return Path(raw) if raw else Path(__file__).parent / "model"


def _load() -> bool:
    global _tokenizer, _embed_model, _load_tried, _available
    if _load_tried:
        return _available
    _load_tried = True

    model_dir = _model_path()
    if not model_dir.exists() or not any(model_dir.iterdir()):
        _available = False
        return False

    try:
        from transformers import AutoTokenizer, AutoModel  # type: ignore
        _tokenizer   = AutoTokenizer.from_pretrained(str(model_dir))
        _embed_model = AutoModel.from_pretrained(str(model_dir))
        _embed_model.eval()
        _available = True
        return True
    except Exception as exc:
        warnings.warn(f"SemanticSimilarity: failed to load model ({exc}).", UserWarning, stacklevel=2)
        _available = False
        return False


def _embed(text: str):
    """Return a mean-pooled embedding tensor for 'text'."""
    import torch
    inputs = _tokenizer(
        text[:512], return_tensors="pt", truncation=True, padding=True
    )
    with torch.no_grad():
        outputs = _embed_model(**inputs)
    # Mean-pool over token dimension, respecting attention mask
    mask   = inputs["attention_mask"].unsqueeze(-1).float()
    tokens = outputs.last_hidden_state
    return (tokens * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def _cosine(a, b) -> float:
    import torch
    return float(
        torch.nn.functional.cosine_similarity(a, b).item()
    )


# Similarity below this → response is unrelated to question → possible hijack
_LOW_SIM_THRESHOLD = 0.20


def semantic_injection_score(question: str, response: str) -> float:
    """
    Return an injection score (0–1) based on how unrelated the LLM response
    is to the user's original question.

    Score = 0   → response is on-topic (safe)
    Score > 0   → response is off-topic (possible hijack)

    Only fires when cosine similarity drops below _LOW_SIM_THRESHOLD (0.20),
    so correct answers that are even loosely related are never flagged.
    """
    if not _load():
        return 0.0   # model unavailable - skip silently

    if not question.strip() or not response.strip():
        return 0.0

    try:
        q_emb = _embed(question)
        r_emb = _embed(response)
        sim   = _cosine(q_emb, r_emb)

        if sim >= _LOW_SIM_THRESHOLD:
            return 0.0

        # Map [0, threshold] linearly to [0.5, 0.0]
        return round(0.5 * (1.0 - sim / _LOW_SIM_THRESHOLD), 4)

    except Exception:
        return 0.0   # never crash the main pipeline
