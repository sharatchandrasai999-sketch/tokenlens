"""Semantic cache - skip paying for a request you've effectively answered before.

Flow: embed the incoming prompt -> find the nearest stored prompt -> if similarity
clears a threshold, return the cached answer (a HIT, you pay nothing); otherwise
call the model, store the answer, return it (a MISS).

Embedders are swappable:
  - SentenceTransformerEmbedder : real semantic embeddings (optional install).
  - HashingEmbedder             : offline, no downloads. Character n-gram hashing
                                  into a fixed vector. Catches near-identical and
                                  lightly-varied prompts well; it is NOT deep
                                  semantic understanding. Honest default so the
                                  project runs anywhere and is testable.

The dangerous failure is a FALSE HIT: two prompts look similar but need different
answers ("capital of Austria" vs "capital of Australia"). So this module ships an
eval that measures hit rate AND false-hit rate, which is how you pick a threshold
with data instead of hope.
"""

from __future__ import annotations
import math
import hashlib


# --------------------------------------------------------------------------
# Embedders
# --------------------------------------------------------------------------
class HashingEmbedder:
    """Offline embedder: hash character 3-grams into a fixed-length vector."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        text = " ".join(text.lower().split())
        vec = [0.0] * self.dim
        grams = [text[i:i + 3] for i in range(max(len(text) - 2, 1))]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        return _normalize(vec)


class SentenceTransformerEmbedder:
    """Real semantic embeddings (needs `pip install sentence-transformers`)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return _normalize(self._model.encode(text).tolist())


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
class SemanticCache:
    def __init__(self, embedder=None, threshold: float = 0.92,
                 store_path: str | None = None):
        self.embedder = embedder or HashingEmbedder()
        self.threshold = threshold
        self.store_path = store_path
        self._entries: list[dict] = []  # {embedding, prompt, response}
        self.hits = 0
        self.misses = 0
        if store_path:
            self._load()

    def _load(self):
        import json
        try:
            with open(self.store_path) as fh:
                self._entries = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            self._entries = []

    def _save(self):
        if not self.store_path:
            return
        import json
        with open(self.store_path, "w") as fh:
            json.dump(self._entries, fh)

    def _nearest(self, emb):
        best, best_sim = None, -1.0
        for e in self._entries:
            sim = cosine(emb, e["embedding"])
            if sim > best_sim:
                best, best_sim = e, sim
        return best, best_sim

    def get(self, prompt: str):
        """Return (hit, response_or_None, similarity)."""
        if not self._entries:
            self.misses += 1
            return False, None, 0.0
        emb = self.embedder.embed(prompt)
        entry, sim = self._nearest(emb)
        if sim >= self.threshold:
            self.hits += 1
            return True, entry["response"], sim
        self.misses += 1
        return False, None, sim

    def put(self, prompt: str, response: str):
        self._entries.append({"embedding": self.embedder.embed(prompt),
                              "prompt": prompt, "response": response})
        self._save()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"entries": len(self._entries), "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0}


def evaluate_threshold(pairs: list[tuple[str, str, bool]], embedder=None,
                       threshold: float = 0.92) -> dict:
    """Measure a threshold on labeled data.

    pairs: (stored_prompt, query_prompt, should_match). Returns hit_rate and the
    critical false_hit_rate (matched when it should NOT have).
    """
    emb = embedder or HashingEmbedder()
    tp = fp = tn = fn = 0
    for stored, query, should in pairs:
        sim = cosine(emb.embed(stored), emb.embed(query))
        matched = sim >= threshold
        if should and matched:
            tp += 1
        elif should and not matched:
            fn += 1
        elif not should and matched:
            fp += 1
        else:
            tn += 1
    pos = tp + fn
    neg = fp + tn
    return {
        "threshold": threshold,
        "hit_rate": round(tp / pos, 3) if pos else 0.0,        # caught real matches
        "false_hit_rate": round(fp / neg, 3) if neg else 0.0,  # served wrong answer
        "true_pos": tp, "false_pos": fp, "true_neg": tn, "false_neg": fn,
    }
