import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer


def _read_json(path: str) -> List[Dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_corpus(menu: List[Dict], faqs: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """
    Returns:
      docs: list of dicts with {type, id, title, text, meta}
      texts: list of strings aligned to docs for vectorizing
    """
    docs: List[Dict] = []

    for item in menu:
        title = item["name"]
        tags = " ".join(item.get("tags", []))
        text = f"{title}. {item['description']} Tags: {tags} Price: {item.get('price','')}"
        docs.append({"type": "menu", "id": item["id"], "title": title, "text": text, "meta": item})

    for f in faqs:
        title = f["question"]
        text = f"Q: {f['question']} A: {f['answer']}"
        docs.append({"type": "faq", "id": f["id"], "title": title, "text": text, "meta": f})

    texts = [d["text"] for d in docs]
    return docs, texts


class FaissTfidfSearch:
    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None
        self.index: faiss.Index | None = None
        self.docs: List[Dict] = []

    def fit(self, menu_path="data/menu.json", faqs_path="data/faqs.json") -> None:
        menu = _read_json(menu_path)
        faqs = _read_json(faqs_path)
        self.docs, texts = build_corpus(menu, faqs)

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)
        X = self.vectorizer.fit_transform(texts)  # sparse (N, D)

        # Convert to dense float32 for FAISS
        dense = X.toarray().astype("float32")

        # Normalize for cosine-like similarity using inner product
        faiss.normalize_L2(dense)

        d = dense.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(dense)

    def search(self, query: str, k: int = 5, filter_type: str | None = None) -> List[Dict]:
        if not self.vectorizer or not self.index:
            raise RuntimeError("Search engine not initialized. Call fit() first.")

        q = self.vectorizer.transform([query]).toarray().astype("float32")
        faiss.normalize_L2(q)

        scores, idxs = self.index.search(q, k)

        results: List[Dict] = []
        for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
            if i == -1:
                continue
            doc = self.docs[i]
            if filter_type and doc["type"] != filter_type:
                continue
            results.append({
                "type": doc["type"],
                "id": doc["id"],
                "title": doc["title"],
                "score": float(score),
                "meta": doc["meta"]
            })
        return results

