from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "BAAI/bge-m3"

_tokenizer = None
_model = None

DEVICE = "cpu"


def get_model():
    global _tokenizer, _model

    if _model is None:
        print(f"Loading semantic evidence model: {MODEL_NAME}")
        print(f"Semantic retrieval device: {DEVICE}")

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        _model.eval()

    return _tokenizer, _model


def _encode(texts: list[str]) -> np.ndarray:
    tokenizer, model = get_model()

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

        attention_mask = inputs["attention_mask"].unsqueeze(-1)

        embeddings = (
            outputs.last_hidden_state * attention_mask
        ).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)

        embeddings = torch.nn.functional.normalize(
            embeddings,
            p=2,
            dim=1,
        )

    return embeddings.cpu().numpy()


def retrieve_semantic_evidence(
    requirement: str,
    evidence_chunks: list[str],
    top_k: int = 5,
) -> list[dict]:

    if not evidence_chunks:
        return []

    query_embedding = _encode([requirement])[0]

    chunk_embeddings = _encode(evidence_chunks)

    scores = np.dot(
        chunk_embeddings,
        query_embedding,
    )

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "chunk": evidence_chunks[int(idx)],
            "score": float(scores[int(idx)]),
        }
        for idx in ranked_indices
    ]