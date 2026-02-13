"""PageIndex-style tree search RAG: get tree -> LLM tree search -> context -> optional answer."""
import json
import copy
from typing import List, Dict, Any, Optional

from clients.pageindex_client import PageIndexClient
from clients.ai_client import AIClient


def get_tree(
    pi_client: PageIndexClient,
    doc_id: str,
    summary: bool = True,
    retry_if_not_ready: bool = True,
    max_retries: int = 8,
) -> List[Dict[str, Any]]:
    """Get full tree (with text and summaries) for doc_id. Waits and retries until retrieval_ready when retry_if_not_ready=True."""
    return pi_client.get_tree(
        doc_id=doc_id,
        summary=summary,
        retry_if_not_ready=retry_if_not_ready,
        max_retries=max_retries,
    )


def _node_id_str(nid: Any) -> str:
    """Normalize node_id for display and IDs (e.g. 19 -> '0019')."""
    if nid is None:
        return ""
    s = str(nid).strip()
    if s.isdigit():
        return s.zfill(4)
    return s


def flatten_tree_to_nodes(
    tree: List[Dict[str, Any]],
    include_nodes_without_content: bool = False,
) -> List[Dict[str, Any]]:
    """
    Flatten a PageIndex tree into a list of checkable nodes for tree-driven workflow.
    Each node has node_id, title, content (text or summary). By default only includes
    nodes that have substantive content (text, summary, or prefix_summary).
    """
    out: List[Dict[str, Any]] = []

    def walk(nodes: List[Dict[str, Any]]) -> None:
        for n in nodes:
            content = (
                (n.get("text") or "").strip()
                or (n.get("summary") or "").strip()
                or (n.get("prefix_summary") or "").strip()
            )
            if content or include_nodes_without_content:
                nid = _node_id_str(n.get("node_id")) or str(len(out) + 1)
                out.append({
                    "node_id": nid,
                    "section_id": nid,
                    "title": (n.get("title") or "").strip() or "Untitled",
                    "content": content or "",
                })
            if n.get("nodes"):
                walk(n["nodes"])

    walk(tree)
    return out

def remove_tree_fields(tree: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    """Deep copy tree and remove given keys from every node (for tree-search prompt)."""
    def _strip(n: Dict[str, Any]) -> Dict[str, Any]:
        out = {k: v for k, v in n.items() if k not in fields}
        if "nodes" in n:
            out["nodes"] = [_strip(c) for c in n["nodes"]]
        return out
    return [_strip(copy.deepcopy(node)) for node in tree]


def tree_search_llm(
    ai_client: AIClient,
    query: str,
    tree_no_text: List[Dict[str, Any]],
    model: Optional[str] = None,
) -> List[str]:
    """One LLM call: reason over tree (no text), return list of node_ids."""
    prompt = f"""You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure:
{json.dumps(tree_no_text, indent=2)}

Please reply in the following JSON format:
{{
  "thinking": " ",
  "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
}}
Directly return the final JSON structure. Do not output anything else."""

    messages = [{"role": "user", "content": prompt}]
    resp = ai_client.client.chat.completions.create(
        model=model or ai_client.model,
        messages=messages,
        temperature=0,
    )
    raw = (resp.choices[0].message.content or "").strip()
    # Strip markdown code block if present
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    # Extract object between first { and last }
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e != -1 and e > s:
        raw = raw[s : e + 1]
    try:
        data = json.loads(raw)
        nlist = data.get("node_list") or data.get("nodes") or []
        return [str(x) for x in nlist]
    except json.JSONDecodeError:
        return []


def build_node_map(tree: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Flatten tree to node_id -> node (includes nested nodes). Keys stored as strings.
    Stores under both raw and zero-padded forms so lookup works whether API returns 19 or '0019'.
    """
    m: Dict[str, Dict[str, Any]] = {}

    def _walk(nodes: List[Dict[str, Any]]) -> None:
        for n in nodes:
            nid = n.get("node_id")
            if nid is not None:
                key = str(nid).strip()
                m[key] = n
                if key.isdigit():
                    m[key.zfill(4)] = n  # so "0019" finds node 19
            if n.get("nodes"):
                _walk(n["nodes"])

    _walk(tree)
    return m


def _node_map_key(nid: Any) -> str:
    """Normalize node_id for lookup (API uses 4-digit zero-padded strings e.g. '0019')."""
    s = str(nid).strip()
    if s.isdigit():
        return s.zfill(4)
    return s


def get_context_for_nodes(
    node_map: Dict[str, Dict[str, Any]],
    node_list: List[str],
) -> str:
    """Concatenate node text (or summary/prefix_summary) for given node_ids."""
    parts = []
    for nid in node_list:
        nid_str = (nid or "").strip()
        node = (
            node_map.get(nid_str)
            or node_map.get(_node_map_key(nid_str))
            or (node_map.get(str(int(nid_str))) if nid_str.isdigit() else None)
        )
        if not node:
            continue
        content = node.get("text") or node.get("summary") or node.get("prefix_summary") or ""
        if content:
            parts.append(content)
    return "\n\n".join(parts)


def answer_from_context(
    ai_client: AIClient,
    query: str,
    context: str,
    model: Optional[str] = None,
) -> str:
    """One LLM call: answer query using only the provided context."""
    prompt = f"""Answer the question based on the context:

Question: {query}
Context: {context}

Provide a clear, concise answer based only on the context provided."""
    messages = [{"role": "user", "content": prompt}]
    resp = ai_client.client.chat.completions.create(
        model=model or ai_client.model,
        messages=messages,
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()


def tree_rag_retrieve(
    pi_client: PageIndexClient,
    ai_client: AIClient,
    doc_id: str,
    query: str,
    model: Optional[str] = None,
) -> str:
    """Get tree -> tree search -> return single context string (for agents or answer step)."""
    tree = get_tree(pi_client, doc_id)
    tree_no_text = remove_tree_fields(tree, ["text"])
    node_list = tree_search_llm(ai_client, query, tree_no_text, model=model)
    node_map = build_node_map(tree)
    return get_context_for_nodes(node_map, node_list)


def tree_rag_answer(
    pi_client: PageIndexClient,
    ai_client: AIClient,
    doc_id: str,
    query: str,
    model: Optional[str] = None,
) -> str:
    """Full flow: tree -> tree search -> context -> answer."""
    context = tree_rag_retrieve(pi_client, ai_client, doc_id, query, model=model)
    return answer_from_context(ai_client, query, context, model=model)