"""Tree-based RAG and chat for the pageindex module. No dependency on PageIndex API or prompt_editor."""
import copy
import json
from typing import Any, Callable, Dict, List, Optional

# -----------------------------------------------------------------------------
# Tree shape helpers (no text in tree for search step)
# -----------------------------------------------------------------------------


def remove_tree_fields(tree: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    """Deep copy tree and remove given keys from every node (for tree-search prompt)."""
    def _strip(n: Dict[str, Any]) -> Dict[str, Any]:
        out = {k: v for k, v in n.items() if k not in fields}
        if "nodes" in n:
            out["nodes"] = [_strip(c) for c in n["nodes"]]
        return out
    return [_strip(copy.deepcopy(node)) for node in tree]


def build_node_map(tree: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Flatten tree to node_id -> node. Keys stored as strings; zero-padded for digit ids."""
    m: Dict[str, Dict[str, Any]] = {}

    def _walk(nodes: List[Dict[str, Any]]) -> None:
        for n in nodes:
            nid = n.get("node_id")
            if nid is not None:
                key = str(nid).strip()
                m[key] = n
                if key.isdigit():
                    m[key.zfill(4)] = n
            if n.get("nodes"):
                _walk(n["nodes"])

    _walk(tree)
    return m


def _node_map_key(nid: Any) -> str:
    s = str(nid).strip()
    if s.isdigit():
        return s.zfill(4)
    return s


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
    Flatten a document tree into a list of checkable nodes (node_id, section_id, title, content).
    By default only includes nodes that have substantive content (text, summary, or prefix_summary).
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


# -----------------------------------------------------------------------------
# LLM calls (require an OpenAI-compatible client: .client.chat.completions.create, .model)
# -----------------------------------------------------------------------------


def tree_search_llm(
    ai_client: Any,
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
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e != -1 and e > s:
        raw = raw[s : e + 1]
    try:
        data = json.loads(raw)
        nlist = data.get("node_list") or data.get("nodes") or []
        return [str(x) for x in nlist]
    except json.JSONDecodeError:
        return []


def answer_from_context(
    ai_client: Any,
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


# -----------------------------------------------------------------------------
# RAG retrieve and full chat (tree from storage or in-memory)
# -----------------------------------------------------------------------------


def tree_rag_retrieve(
    ai_client: Any,
    query: str,
    tree: List[Dict[str, Any]],
    model: Optional[str] = None,
) -> str:
    """Tree search -> context string. Caller provides the tree (e.g. from storage)."""
    tree_no_text = remove_tree_fields(tree, ["text"])
    node_list = tree_search_llm(ai_client, query, tree_no_text, model=model)
    node_map = build_node_map(tree)
    return get_context_for_nodes(node_map, node_list)


def chat(
    ai_client: Any,
    query: str,
    tree: List[Dict[str, Any]],
    model: Optional[str] = None,
) -> str:
    """Full flow: tree -> tree search -> context -> answer. Uses only the provided tree (no API)."""
    context = tree_rag_retrieve(ai_client, query, tree, model=model)
    return answer_from_context(ai_client, query, context, model=model)


def get_tree_then_chat(
    ai_client: Any,
    query: str,
    get_tree: Callable[[], Optional[List[Dict[str, Any]]]],
    model: Optional[str] = None,
) -> str:
    """Load tree via get_tree() then run chat. get_tree can be storage.get_tree bound to a doc_id."""
    tree = get_tree()
    if not tree:
        return "No document tree available for this document."
    return chat(ai_client, query, tree, model=model)
