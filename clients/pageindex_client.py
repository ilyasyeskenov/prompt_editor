"""PageIndex API client: Legacy Retrieval API (query doc, get chunks) and doc status."""
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
        url = f"{self.doc_base_url}/{doc_id}/?type=tree"
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
