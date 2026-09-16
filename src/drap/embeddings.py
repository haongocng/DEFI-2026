from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence


class Embedder(Protocol):
    def encode(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Dependency-free lexical embedder for tests and local smoke runs."""

    def __init__(self, dimensions: int = 384):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[\w'-]+", text.casefold())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self.dimensions
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        local_files_only: bool = False,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install the semantic extra: pip install -e '.[semantic]'"
            ) from exc
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.model = SentenceTransformer(
            model_name,
            local_files_only=local_files_only,
        )
        dimension_reader = getattr(self.model, "get_embedding_dimension", None)
        if dimension_reader is None:
            dimension_reader = self.model.get_sentence_embedding_dimension
        self.dimensions = int(dimension_reader())

    def encode(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
