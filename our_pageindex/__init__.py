"""
Standalone our_pageindex module: build document trees from PDFs and chat over them (no PageIndex API).

Usage:
  from our_pageindex import build_tree_from_pdf, get_tree, chat, get_tree_then_chat
  from our_pageindex import SupabaseTreeStorage, InMemoryTreeStorage

  # Build tree from PDF and optionally save to Supabase
  storage = SupabaseTreeStorage()
  tree = build_tree_from_pdf("doc.pdf", doc_id="my-doc", storage=storage)

  # Or load existing tree and chat
  tree = get_tree("my-doc", storage=storage)
  answer = chat(ai_client, "What is the deadline?", tree=tree)

  # Or one-liner: load tree from storage and answer
  answer = get_tree_then_chat(ai_client, "What is the deadline?", get_tree=lambda: get_tree("my-doc", storage=storage))
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from . import chat as _chat
from . import storage as _storage
from .page_index import page_index_main
from .utils import ConfigLoader

# Re-export storage
TreeStorage = _storage.TreeStorage
InMemoryTreeStorage = _storage.InMemoryTreeStorage
SupabaseTreeStorage = _storage.SupabaseTreeStorage

# Re-export chat
tree_rag_retrieve = _chat.tree_rag_retrieve
chat = _chat.chat
get_tree_then_chat = _chat.get_tree_then_chat
tree_search_llm = _chat.tree_search_llm
build_node_map = _chat.build_node_map
get_context_for_nodes = _chat.get_context_for_nodes
answer_from_context = _chat.answer_from_context
remove_tree_fields = _chat.remove_tree_fields
flatten_tree_to_nodes = _chat.flatten_tree_to_nodes


def build_tree_from_pdf(
    pdf_path_or_bytes: Union[str, bytes],
    doc_id: Optional[str] = None,
    storage: Optional[TreeStorage] = None,
    model: Optional[str] = None,
    toc_check_page_num: Optional[int] = None,
    max_page_num_each_node: Optional[int] = None,
    max_token_num_each_node: Optional[int] = None,
    if_add_node_id: Optional[str] = None,
    if_add_node_summary: Optional[str] = None,
    if_add_doc_description: Optional[str] = None,
    if_add_node_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build a document tree from a PDF (file path or bytes). Optionally save to storage under doc_id.

    Returns the tree (list of top-level nodes with node_id, title, text/summary, nodes).
    """
    from io import BytesIO
    doc = pdf_path_or_bytes
    if isinstance(doc, bytes):
        doc = BytesIO(doc)
    user_opt: Dict[str, Any] = {}
    if model is not None:
        user_opt["model"] = model
    if toc_check_page_num is not None:
        user_opt["toc_check_page_num"] = toc_check_page_num
    if max_page_num_each_node is not None:
        user_opt["max_page_num_each_node"] = max_page_num_each_node
    if max_token_num_each_node is not None:
        user_opt["max_token_num_each_node"] = max_token_num_each_node
    if if_add_node_id is not None:
        user_opt["if_add_node_id"] = if_add_node_id
    if if_add_node_summary is not None:
        user_opt["if_add_node_summary"] = if_add_node_summary
    if if_add_doc_description is not None:
        user_opt["if_add_doc_description"] = if_add_doc_description
    if if_add_node_text is not None:
        user_opt["if_add_node_text"] = if_add_node_text
    opt = ConfigLoader().load(user_opt)
    result = page_index_main(doc, opt)
    tree = result["structure"]
    if doc_id and storage and tree:
        storage.save_tree(doc_id, tree, metadata={"doc_name": result.get("doc_name")})
    return tree


def get_tree(doc_id: str, storage: TreeStorage) -> Optional[List[Dict[str, Any]]]:
    """Load tree for doc_id from the given storage. Returns None if not found."""
    return storage.get_tree(doc_id)


def build_tree_with_generated_id(
    pdf_path_or_bytes: Union[str, bytes],
    storage: TreeStorage,
    model: Optional[str] = None,
    toc_check_page_num: Optional[int] = None,
    max_page_num_each_node: Optional[int] = None,
    max_token_num_each_node: Optional[int] = None,
    if_add_node_id: Optional[str] = None,
    if_add_node_summary: Optional[str] = None,
    if_add_doc_description: Optional[str] = None,
    if_add_node_text: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Convenience helper: generate a random doc_id, build the tree, save it, and return (doc_id, tree).

    This is the easiest way for external callers to:
      - upload a PDF to pageindex,
      - get back a doc_id they can store/use later,
      - and have the tree persisted in the configured storage.
    """
    doc_id = uuid.uuid4().hex
    tree = build_tree_from_pdf(
        pdf_path_or_bytes=pdf_path_or_bytes,
        doc_id=doc_id,
        storage=storage,
        model=model,
        toc_check_page_num=toc_check_page_num,
        max_page_num_each_node=max_page_num_each_node,
        max_token_num_each_node=max_token_num_each_node,
        if_add_node_id=if_add_node_id,
        if_add_node_summary=if_add_node_summary,
        if_add_doc_description=if_add_doc_description,
        if_add_node_text=if_add_node_text,
    )
    return doc_id, tree


__all__ = [
    "build_tree_from_pdf",
    "build_tree_with_generated_id",
    "get_tree",
    "chat",
    "get_tree_then_chat",
    "tree_rag_retrieve",
    "tree_search_llm",
    "build_node_map",
    "get_context_for_nodes",
    "answer_from_context",
    "remove_tree_fields",
    "flatten_tree_to_nodes",
    "TreeStorage",
    "InMemoryTreeStorage",
    "SupabaseTreeStorage",
]
