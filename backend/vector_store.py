# Copyright F5, Inc. 2026
# Licensed under the MIT License. See LICENSE.

# backend/vector_store.py
import os
import json
import glob
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


class VectorStore:
    """
    Chroma-backed vector store with:
      - JSON-file loaders (one file = one collection)
      - Full metadata preservation (name, title, salary, address, etc.)
      - Query top-k and get-all helpers returning (docs, metas, ids)
    """

    def __init__(
        self,
        db_path: str = os.getenv("CHROMA_DIR", "./chroma_db"),
        data_dir: str = os.getenv("DATA_DIR", "./data"),
        batch_size: int = 256,
    ):
        self.db_path = db_path
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Use Chroma's built-in OpenAI adapter (implements embed_query, embed_documents, name)
        embed_api_key = os.getenv("OPENAI_EMBED_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not embed_api_key:
            raise ValueError("Missing OPENAI_EMBED_API_KEY (or fallback OPENAI_API_KEY) for embeddings.")
        self.embedding_fn = OpenAIEmbeddingFunction(
            api_key=embed_api_key,
            model_name=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        )

        self.client = chromadb.PersistentClient(path=db_path)
        self.collections: Dict[str, Any] = {}

    # -------- Public API --------

    def load_all_collections(self) -> Dict[str, int]:
        """
        Scan data_dir for *.json files. Each file name becomes the collection name.
        Seed a collection if it's empty; otherwise no-op.
        Returns {collection_name: count}.
        """
        counts = {}
        for path in glob.glob(os.path.join(self.data_dir, "*.json")):
            name = os.path.splitext(os.path.basename(path))[0]
            col = self._get_or_create_collection(name)
            if col.count() == 0:
                items = self._read_items(path)
                if items:
                    self._upsert_items(col, items)
                    self._persist_safe()
            counts[name] = col.count()
        return counts

    def reload_collection(self, name: str) -> int:
        """
        Force re-load from file into the collection (drop + re-add).
        """
        path = os.path.join(self.data_dir, f"{name}.json")
        col = self._get_or_create_collection(name)
        # delete all existing
        if col.count() > 0:
            ids = col.get(include=[]).get("ids", [])
            if ids:
                col.delete(ids=ids)

        items = self._read_items(path)
        if items:
            self._upsert_items(col, items)
        self._persist_safe()
        return col.count()

    def query_top_k(
        self, collection: str, query: str, k: int, include_meta: bool = True
    ) -> Tuple[List[str], List[Dict[str, Any]], List[Optional[str]]]:
        """
        Returns (docs, metas, ids) for the top-k results.
        """
        col = self._get_or_create_collection(collection)
        n = max(1, min(k, max(col.count(), 1)))
        res = col.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"] if include_meta else ["documents"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        ids = (res.get("ids") or [[]])[0] or [None] * len(docs)
        return docs, metas, ids

    def get_all_docs(
        self, collection: str, include_meta: bool = True
    ) -> Tuple[List[str], List[Dict[str, Any]], List[Optional[str]]]:
        """
        Returns (docs, metas, ids) for the full collection.
        """
        col = self._get_or_create_collection(collection)
        got = col.get(include=["documents", "metadatas"] if include_meta else ["documents"])
        docs = got.get("documents", [])
        metas = got.get("metadatas", [])
        ids = got.get("ids", []) or [None] * len(docs)
        return docs, metas, ids

    # -------- Internals --------

    def _get_or_create_collection(self, name: str):
        if name in self.collections:
            return self.collections[name]
        col = self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self.collections[name] = col
        return col

    def _read_items(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON array of objects")

        items: List[Dict[str, Any]] = []
        for i, obj in enumerate(data):
            if "text" not in obj:
                raise ValueError(f"{path} item {i} missing 'text' field")

            # Ensure id exists (fallback to filename:index)
            _id = str(obj.get("id", f"{os.path.basename(path)}:{i}"))

            # 'text' goes to documents; all other keys are metadata (name/title/salary/address/etc.)
            meta = {k: v for k, v in obj.items() if k != "text"}
            meta["id"] = _id  # also keep id in metadata for convenience

            items.append({"id": _id, "text": str(obj["text"]), "meta": meta})
        return items

    def _upsert_items(self, col, items: List[Dict[str, Any]]):
        # Batch to avoid very large single calls
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            ids = [it["id"] for it in batch]
            docs = [it["text"] for it in batch]
            metas = [it["meta"] for it in batch]
            col.upsert(ids=ids, documents=docs, metadatas=metas)

    def _persist_safe(self):
        try:
            self.client.persist()
        except Exception:
            # Some backends auto-persist; ignore
            pass
