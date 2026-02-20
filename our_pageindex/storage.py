"""Tree storage backends for the pageindex module. Save/load trees by doc_id (e.g. Supabase)."""
import json
import os
from typing import Any, Dict, List, Optional

# Table name for trees (doc_id -> tree JSON + optional metadata)
TABLE_PAGEINDEX_TREES = "pageindex_trees"


class TreeStorage:
    """Abstract storage for document trees. Implementations: InMemoryTreeStorage, SupabaseTreeStorage."""

    def get_tree(self, doc_id: str) -> Optional[List[Dict[str, Any]]]:
        """Load tree for doc_id. Returns None if not found."""
        raise NotImplementedError

    def save_tree(
        self,
        doc_id: str,
        tree: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Persist tree for doc_id. Returns True on success."""
        raise NotImplementedError


class InMemoryTreeStorage(TreeStorage):
    """In-memory storage for trees (useful for tests or single-run usage)."""

    def __init__(self) -> None:
        self._store: Dict[str, tuple] = {}

    def get_tree(self, doc_id: str) -> Optional[List[Dict[str, Any]]]:
        if doc_id not in self._store:
            return None
        return self._store[doc_id][0]

    def save_tree(
        self,
        doc_id: str,
        tree: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        self._store[doc_id] = (tree, metadata or {})
        return True


class SupabaseTreeStorage(TreeStorage):
    """Save/load trees in Supabase table `pageindex_trees` (doc_id primary key, tree_json JSONB, metadata JSONB, created_at)."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        table: str = TABLE_PAGEINDEX_TREES,
    ) -> None:
        self._url = (supabase_url or os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or "").strip()
        self._key = (supabase_key or os.getenv("SUPABASE_ANON_KEY") or "").strip()
        self._table = table
        self._client: Any = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self._url or not self._key:
            raise ValueError(
                "Supabase credentials required. Set SUPABASE_URL and SUPABASE_ANON_KEY (or pass supabase_url/supabase_key)."
            )
        from supabase import create_client
        self._client = create_client(self._url, self._key)
        return self._client

    def get_tree(self, doc_id: str) -> Optional[List[Dict[str, Any]]]:
        if not (doc_id or "").strip():
            return None
        try:
            client = self._get_client()
            response = (
                client.table(self._table)
                .select("tree_json")
                .eq("doc_id", doc_id.strip())
                .limit(1)
                .execute()
            )
            if response.data and len(response.data) > 0:
                raw = response.data[0].get("tree_json")
                if raw is None:
                    return None
                if isinstance(raw, list):
                    return raw
                if isinstance(raw, str):
                    return json.loads(raw)
                if isinstance(raw, dict):
                    return [raw]
            return None
        except Exception:
            return None

    def save_tree(
        self,
        doc_id: str,
        tree: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not (doc_id or "").strip():
            return False
        try:
            client = self._get_client()
            row = {
                "doc_id": doc_id.strip(),
                "tree_json": tree,
                "metadata": metadata or {},
            }
            client.table(self._table).upsert(row, on_conflict="doc_id").execute()
            return True
        except Exception:
            return False
