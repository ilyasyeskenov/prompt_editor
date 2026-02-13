"""PageIndex API client: document upload, status, tree, legacy retrieval."""
import time
import requests
from typing import Optional, List, Union, Dict, Any


class PageIndexError(Exception):
    """Raised when a PageIndex API call fails."""
    pass


def _retrieved_nodes_to_chunks(retrieved_nodes: List[Dict[str, Any]], file_name: str = "PageIndex") -> List[Dict[str, Any]]:
    """Convert PageIndex retrieved_nodes to chunk list expected by omission/contradiction agents."""
    chunks = []
    for node in retrieved_nodes or []:
        title = node.get("title") or file_name
        for rc in node.get("relevant_contents") or []:
            chunks.append({
                "content": rc.get("relevant_content", ""),
                "file_name": title,
                "page_number": rc.get("page_index"),
            })
    return chunks


class PageIndexClient:
    """Client for PageIndex Legacy Retrieval API and document status."""

    def __init__(self, api_key: str, chat_url: str = "https://api.pageindex.ai/chat/completions"):
        self.api_key = api_key
        self.chat_url = chat_url.rstrip("/")
        self.doc_base_url = "https://api.pageindex.ai/doc"
        self.retrieval_base_url = "https://api.pageindex.ai/retrieval"

    def _headers(self) -> Dict[str, str]:
        return {"api_key": self.api_key, "Content-Type": "application/json"}

    def upload_doc(
        self,
        file_bytes: bytes,
        filename: str = "document.pdf",
    ) -> str:
        """
        Upload a PDF to PageIndex for processing (tree generation). Returns doc_id.
        """
        if not self.api_key:
            raise PageIndexError("PAGEINDEX_API_KEY is not set")
        resp = requests.post(
            f"{self.doc_base_url}/",
            headers={"api_key": self.api_key},
            files={"file": (filename, file_bytes, "application/pdf")},
            timeout=120,
        )
        if resp.status_code == 401:
            raise PageIndexError("PageIndex API key invalid or missing")
        if resp.status_code != 200:
            raise PageIndexError(
                f"PageIndex upload error: {resp.status_code} - {resp.text[:500]}"
            )
        data = resp.json()
        doc_id = data.get("doc_id")
        if not doc_id:
            raise PageIndexError("PageIndex upload returned no doc_id")
        return doc_id

    def is_retrieval_ready(self, doc_id: str) -> bool:
        """
        Check if the document is ready for retrieval (tree available).
        Matches PageIndex SDK / API: check retrieval_ready or status=completed.
        """
        data = self.get_doc_status(doc_id)
        if data.get("retrieval_ready"):
            return True
        if (data.get("status") or "").strip().lower() == "completed":
            return True
        return False

    def poll_until_ready(
        self,
        doc_id: str,
        poll_interval: float = 3.0,
        max_wait: float = 300.0,
    ) -> bool:
        """
        Poll GET doc/{doc_id}/?type=tree until the document is retrieval_ready.
        Uses retrieval_ready or status=completed (per PageIndex API / SDK).
        """
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            data = self.get_doc_status(doc_id)
            if data.get("retrieval_ready") or (data.get("status") or "").strip().lower() == "completed":
                return True
            status = (data.get("status") or "").strip().lower()
            if status and status not in ("processing", "pending", "queued", ""):
                raise PageIndexError(f"PageIndex doc processing failed: status={status}")
            time.sleep(poll_interval)
        raise PageIndexError(f"PageIndex doc not retrieval_ready after {max_wait}s")

    def submit_retrieval(self, doc_id: str, query: str, thinking: bool = False) -> str:
        """POST /retrieval/ → returns retrieval_id."""
        if not self.api_key:
            raise PageIndexError("PAGEINDEX_API_KEY is not set")
        payload = {"doc_id": doc_id, "query": query, "thinking": thinking}
        resp = requests.post(
            f"{self.retrieval_base_url}/",
            headers=self._headers(),
            json=payload,
            timeout=60,
        )
        if resp.status_code == 401:
            raise PageIndexError("PageIndex API key invalid or missing")
        if resp.status_code != 200:
            raise PageIndexError(f"PageIndex retrieval submit error: {resp.status_code} - {resp.text[:500]}")
        data = resp.json()
        rid = data.get("retrieval_id")
        if not rid:
            raise PageIndexError("PageIndex retrieval returned no retrieval_id")
        return rid

    def get_retrieval(self, retrieval_id: str) -> Dict[str, Any]:
        """GET /retrieval/{retrieval_id}/ → status and (when completed) retrieved_nodes."""
        if not self.api_key:
            raise PageIndexError("PAGEINDEX_API_KEY is not set")
        resp = requests.get(
            f"{self.retrieval_base_url}/{retrieval_id}/",
            headers={"api_key": self.api_key},
            timeout=30,
        )
        if resp.status_code == 401:
            raise PageIndexError("PageIndex API key invalid or missing")
        if resp.status_code != 200:
            raise PageIndexError(f"PageIndex retrieval status error: {resp.status_code} - {resp.text[:500]}")
        return resp.json()

    def retrieve(
        self,
        doc_id: str,
        query: str,
        thinking: bool = False,
        poll_interval: float = 2.0,
        max_wait: float = 120.0,
        file_name: str = "PageIndex",
    ) -> List[Dict[str, Any]]:
        """Submit retrieval, poll until completed, return chunks (list of {content, file_name, page_number})."""
        rid = self.submit_retrieval(doc_id=doc_id, query=query, thinking=thinking)
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            data = self.get_retrieval(rid)
            status = data.get("status", "")
            if status == "completed":
                return _retrieved_nodes_to_chunks(data.get("retrieved_nodes") or [], file_name=file_name)
            if status and status not in ("processing", "pending"):
                raise PageIndexError(f"PageIndex retrieval failed with status: {status}")
            time.sleep(poll_interval)
        raise PageIndexError(f"PageIndex retrieval timed out after {max_wait}s")

    def chat(
        self,
        doc_id: Union[str, List[str]],
        message: str,
        enable_citations: bool = False,
    ) -> str:
        """
        Send a question to PageIndex Chat scoped to one or more documents.
        Returns the assistant content (answer text).

        Args:
            doc_id: Single doc_id string or list of doc_ids.
            message: User message (the question).
            enable_citations: If True, response may include inline citations.

        Returns:
            Answer text from choices[0].message.content.

        Raises:
            PageIndexError: On API error (401, 500, or missing content).
        """
        if not self.api_key:
            raise PageIndexError("PAGEINDEX_API_KEY is not set")
        payload = {
            "messages": [{"role": "user", "content": message}],
            "stream": False,
            "doc_id": doc_id,
            "enable_citations": enable_citations,
        }
        resp = requests.post(
            self.chat_url,
            headers={
                "api_key": self.api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        if resp.status_code == 401:
            raise PageIndexError("PageIndex API key invalid or missing")
        if resp.status_code != 200:
            raise PageIndexError(
                f"PageIndex Chat API error: {resp.status_code} - {resp.text[:500]}"
            )
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise PageIndexError("PageIndex Chat API returned no choices")
        content = choices[0].get("message", {}).get("content")
        if content is None:
            raise PageIndexError("PageIndex Chat API returned empty content")
        return content.strip()

    def get_doc_status(self, doc_id: str) -> dict:
        """
        Get document processing status (and retrieval_ready) via tree endpoint.

        Returns:
            Dict with at least: doc_id, status, retrieval_ready (when type=tree).
        """
        if not self.api_key:
            raise PageIndexError("PAGEINDEX_API_KEY is not set")
        # API: GET https://api.pageindex.ai/doc/{doc_id}/ with trailing slash
        url = f"{self.doc_base_url.rstrip('/')}/{doc_id}/?type=tree"
        resp = requests.get(
            url,
            headers={"api_key": self.api_key},
            timeout=30,
        )
        if resp.status_code == 401:
            raise PageIndexError("PageIndex API key invalid or missing")
        if resp.status_code != 200:
            raise PageIndexError(
                f"PageIndex doc status error: {resp.status_code} - {resp.text[:500]}"
            )
        return resp.json()

    def get_tree(self, doc_id: str, summary: bool = True, retry_if_not_ready: bool = True, max_retries: int = 3) -> List[Dict[str, Any]]:
        """
        Get PageIndex tree (node_id, title, text, summary?, nodes).
        Only returns when retrieval_ready or status=completed (per API / SDK).
        Returns a list of top-level nodes (single root wrapped in list if needed).
        """
        if not self.api_key:
            raise PageIndexError("PAGEINDEX_API_KEY is not set")
        # API: type=tree, summary (boolean) for node summaries
        url = f"{self.doc_base_url.rstrip('/')}/{doc_id}/?type=tree"
        if summary:
            url += "&summary=true"
        for attempt in range(max_retries):
            resp = requests.get(
                url,
                headers={"api_key": self.api_key},
                timeout=60,
            )
            if resp.status_code != 200:
                raise PageIndexError(
                    f"PageIndex get_tree error: {resp.status_code} - {resp.text[:500]}"
                )
            data = resp.json()
            # Only use tree when doc is ready (matches SDK is_retrieval_ready + get_tree)
            if not data.get("retrieval_ready") and (data.get("status") or "").strip().lower() != "completed":
                if not retry_if_not_ready or attempt == max_retries - 1:
                    status = data.get("status", "unknown")
                    raise PageIndexError(f"Document not retrieval_ready yet (status: {status}, attempt {attempt + 1}/{max_retries})")
                time.sleep(3.0 * (attempt + 1))
                continue
            result = data.get("result")
            if result is None:
                if not retry_if_not_ready or attempt == max_retries - 1:
                    raise PageIndexError("Document ready but tree result is empty")
                time.sleep(3.0 * (attempt + 1))
                continue
            # API returns result as array of nodes; sometimes single root as one object
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                # Single root node (e.g. notebook "TENDER SUBMISSION"): wrap so we keep root
                return [result]
            return []
        raise PageIndexError("Document not retrieval_ready after retries")
