"""LangGraph workflow for tender checking multi-agent system."""
import json
import os
import re
import threading
import traceback
from typing import TypedDict, List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from langgraph.graph import StateGraph, END
from clients.ai_client import AIClient
from clients.supabase_client import SupabaseClient
from utils.pageindex_tree_rag import get_tree, flatten_tree_to_nodes, tree_rag_retrieve
from tender_checker.agents.breakdown_agent import BreakdownAgent
from tender_checker.agents.omission_checker_agent import OmissionCheckerAgent
from tender_checker.agents.contradiction_checker_agent import ContradictionCheckerAgent
from tender_checker.agents.orchestrator_agent import OrchestratorAgent
from tender_checker.prompts.agent_prompts import (
    SECTION_EXTRACTION_PROMPT,
    REQUIREMENTS_LIST_EXTRACTION_PROMPT,
)

# Batching and rate-limit defaults
RETRIEVAL_BATCH_SIZE = 15
RETRIEVAL_MAX_WORKERS = 10
OPENAI_CONCURRENCY = 5

# Optional file logging for local debugging: set TENDER_CHECK_DEBUG=1 or path to a log file
def _debug_log(msg: str, exc: Optional[Exception] = None) -> None:
    path = os.getenv("TENDER_CHECK_DEBUG", "").strip()
    if not path:
        return
    if path == "1":
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tender_check_debug.log")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            if exc:
                f.write(traceback.format_exc() + "\n")
    except Exception:
        pass

# PageIndex Tree RAG: query templates for omission and contradiction (tree search → context → chunks)
PAGEINDEX_OMISSION_QUESTION = (
    "Does this document state or imply the following requirement? "
    "Quote the relevant parts. Answer: fulfilled / partially / not fulfilled.\n\nRequirement: {requirement_text}"
)
PAGEINDEX_CONTRADICTION_QUESTION = (
    "Does this guideline document contradict or conflict with the following requirement? "
    "Quote any conflicting parts. Answer: no contradiction / minor / moderate / critical contradiction.\n\nRequirement: {requirement_text}"
)
# Section-based (reference-driven) PageIndex: query to get handbook requirements relevant to a tender section
PAGEINDEX_SECTION_REQUIREMENTS_QUERY = (
    "List all requirements from this document that a tender must comply with, relevant to the following section or topic. "
    "Return the full text of each requirement or guideline so they can be checked against the tender.\n\n"
    "Section/topic: {section_title}\n\n"
    "Optional context from the tender (first part of the section):\n{section_preview}"
)

PAGEINDEX_CHAT_SECTION_REQUIREMENTS_QUERY = (
    "You are a requirements analyst. Based on the document, list the distinct, verifiable requirements relevant to the section/topic below.\n\n"
    "Section/topic: {section_title}\n\n"
    "Optional context from this section (excerpt):\n{section_preview}\n\n"
    "Return ONLY valid JSON in this schema:\n"
    "{{\n"
    '  "requirements": ["Requirement 1", "Requirement 2"]\n'
    "}}\n"
)

PAGEINDEX_CHAT_COMBINED_CHECK_QUERY = (
    "You are a compliance auditor.\n\n"
    "We are checking a single document using PageIndex Chat. Use the provided section excerpt as the submission material to cite from.\n\n"
    "Requirement to check:\n{requirement_text}\n\n"
    "Submission section title:\n{section_title}\n\n"
    "Submission section excerpt (cite exact quotes from THIS excerpt in citations):\n{section_excerpt}\n\n"
    "Tasks:\n"
    "1) Omission check: determine whether the submission excerpt supports the requirement.\n"
    "2) Contradiction check: determine whether any part of the document contradicts/conflicts with the requirement (if none found, say so).\n\n"
    "Return ONLY valid JSON in this schema (exact keys):\n"
    "{{\n"
    '  "omission": {{\n'
    '    "requirement_id": "{requirement_id}",\n'
    '    "status": "FULFILLED" | "PARTIALLY_FULFILLED" | "NOT_FULFILLED",\n'
    '    "confidence": 0.0,\n'
    '    "justification": "string",\n'
    '    "citations": [{{"source_text":"string","document_reference":"string","relevance":"string"}}],\n'
    '    "missing_elements": ["string"]\n'
    "  }},\n"
    '  "contradiction": {{\n'
    '    "requirement_id": "{requirement_id}",\n'
    '    "has_contradiction": true,\n'
    '    "severity": "CRITICAL" | "MODERATE" | "MINOR" | "NO_CONTRADICTION",\n'
    '    "contradiction_details": "string",\n'
    '    "reference_guideline": "string",\n'
    '    "tender_statement": "string",\n'
    '    "citations": [{{"source_text":"string","document_reference":"string"}}],\n'
    '    "recommendation": "string"\n'
    "  }}\n"
    "}}\n"
)

def _extract_first_json_object(text: str) -> Optional[dict]:
    """Best-effort extraction of the first JSON object from a text blob."""
    if not text:
        return None
    raw = text.strip()
    # Strip fenced code blocks if present
    if "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]
            raw = re.sub(r"^\s*json\s*", "", raw, flags=re.IGNORECASE).strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    try:
        return json.loads(raw[s : e + 1])
    except Exception:
        return None


def _get_requirements_list(data: Dict[str, Any]) -> List[str]:
    """
    Get a list of requirement strings from a parsed JSON object.
    Tolerates malformed keys (e.g. '\\n \"requirements\"') and non-list values.
    Always returns a list; never raises KeyError.
    """
    if not data or not isinstance(data, dict):
        return []
    # Prefer exact key
    raw = data.get("requirements")
    if raw is None:
        # Fallback: find any key that strip()s to "requirements"
        for k, v in data.items():
            if isinstance(k, str) and k.strip() == "requirements":
                raw = v
                break
    if raw is None:
        return []
    if isinstance(raw, list):
        out: List[str] = []
        for r in raw:
            if isinstance(r, str) and r.strip():
                out.append(r.strip())
            elif r is not None:
                out.append(str(r).strip())
        return [x for x in out if x]
    # Model returned a string or other type: treat as single requirement or empty
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


def _get_nested_dict(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Get a nested dict by key, tolerating malformed keys (e.g. leading newline). Always returns a dict."""
    if not data or not isinstance(data, dict):
        return {}
    out = data.get(key)
    if isinstance(out, dict):
        return out
    for k, v in data.items():
        if isinstance(k, str) and k.strip() == key and isinstance(v, dict):
            return v
    return {}


def normalize_requirement_ids(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize requirement IDs to ensure consistent format (REQ-1, REQ-2, etc.).
    
    Args:
        requirements: List of requirement dicts
        
    Returns:
        List of requirements with normalized IDs
    """
    normalized = []
    for i, req in enumerate(requirements, 1):
        normalized_req = req.copy()
        normalized_req["id"] = f"REQ-{i}"
        normalized.append(normalized_req)
    return normalized


def normalize_final_report(final_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize final report schema to ensure all expected keys exist with defaults.
    
    Args:
        final_report: Raw orchestrator output
        
    Returns:
        Normalized report with all expected keys
    """
    return {
        "overall_status": final_report.get("overall_status", "UNKNOWN"),
        "compliance_score": final_report.get("compliance_score", 0.0),
        "summary": final_report.get("summary", "No summary available"),
        "critical_issues": final_report.get("critical_issues", []),
        "omission_summary": final_report.get("omission_summary", {
            "total_requirements": 0,
            "fulfilled": 0,
            "partially_fulfilled": 0,
            "not_fulfilled": 0,
            "missing_requirements": []
        }),
        "contradiction_summary": final_report.get("contradiction_summary", {
            "total_checked": 0,
            "critical_contradictions": 0,
            "moderate_contradictions": 0,
            "minor_contradictions": 0,
            "contradictions": []
        }),
        "recommendations": final_report.get("recommendations", []),
        "risk_assessment": final_report.get("risk_assessment", "Unable to assess risk")
    }


class TenderCheckState(TypedDict):
    """State for the tender checking workflow."""
    tender_text: str
    tender_summary: str
    requirements: List[Dict[str, Any]]
    sections: List[Dict[str, Any]]  # Legacy alias for submission_sections
    submission_sections: List[Dict[str, Any]]  # Submission section index: {section_id/node_id, title, content} from PageIndex tree or LLM
    retrieval_results: List[Dict[str, Any]]
    omission_results: List[Dict[str, Any]]
    contradiction_results: List[Dict[str, Any]]
    final_report: Dict[str, Any]
    project_id: str
    guidelines_project_id: str
    top_k: int
    error: str
    use_pageindex_chat: bool
    use_pageindex_our: bool
    reference_doc_id: str
    guidelines_doc_id: str
    submission_doc_id: str  # PageIndex doc_id for submission (section index from tree when set)
    
class TenderCheckWorkflow:
    """LangGraph workflow for tender checking with fan-out retrieval and single fan-in to orchestrate."""

    def __init__(
        self,
        ai_client: AIClient,
        supabase_client: SupabaseClient,
        breakdown_prompt: str = None,
        omission_prompt: str = None,
        contradiction_prompt: str = None,
        orchestrator_prompt: str = None,
        retrieval_batch_size: int = RETRIEVAL_BATCH_SIZE,
        retrieval_max_workers: int = RETRIEVAL_MAX_WORKERS,
        openai_concurrency: int = OPENAI_CONCURRENCY,
        progress_callback: Optional[Callable[[str, int, int, Dict[str, Any]], None]] = None,
        pageindex_client: Optional[Any] = None,
    ):
        self.ai_client = ai_client
        self.supabase_client = supabase_client
        self.pageindex_client = pageindex_client
        self.retrieval_batch_size = retrieval_batch_size
        self.retrieval_max_workers = retrieval_max_workers
        self.openai_semaphore = threading.Semaphore(openai_concurrency)
        self.progress_callback = progress_callback

        # Initialize agents with custom prompts
        self.breakdown_agent = BreakdownAgent(ai_client, breakdown_prompt)
        self.omission_checker = OmissionCheckerAgent(ai_client, supabase_client, omission_prompt)
        self.contradiction_checker = ContradictionCheckerAgent(ai_client, supabase_client, contradiction_prompt)
        self.orchestrator = OrchestratorAgent(ai_client, orchestrator_prompt)

        # Build workflow
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow: breakdown → retrieval → check → orchestrate (single path, proper join)."""
        workflow = StateGraph(TenderCheckState)

        workflow.add_node("breakdown", self._breakdown_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("check", self._check_node)
        workflow.add_node("orchestrate", self._orchestrate_node)

        workflow.set_entry_point("breakdown")
        workflow.add_edge("breakdown", "retrieval")
        workflow.add_edge("retrieval", "check")
        workflow.add_edge("check", "orchestrate")
        workflow.add_edge("orchestrate", END)

        return workflow.compile()

    def _extract_sections(self, tender_text: str) -> List[Dict[str, Any]]:
        """Extract sections from tender text (for PageIndex reference-driven flow). Returns list of {section_id, title, content}."""
        prompt = SECTION_EXTRACTION_PROMPT.replace("{{tender_text}}", tender_text)
        try:
            resp = self.ai_client.client.chat.completions.create(
                model=self.ai_client.model,
                messages=[
                    {"role": "system", "content": "You are a document analyst. Always return valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = json.loads(raw)
            sections = data.get("sections") or []
            for i, sec in enumerate(sections, 1):
                sec["section_id"] = sec.get("section_id") or f"SEC-{i}"
                sec["title"] = sec.get("title") or f"Section {i}"
                sec["content"] = sec.get("content") or ""
            return sections
        except Exception:
            return []

    def _parse_requirements_from_text(self, reference_text: str) -> List[str]:
        """Parse reference/handbook text into a list of distinct requirement strings (for section-based flow)."""
        if not (reference_text or reference_text.strip()):
            return []
        prompt = REQUIREMENTS_LIST_EXTRACTION_PROMPT.replace("{{reference_text}}", reference_text[:12000])
        try:
            resp = self.ai_client.client.chat.completions.create(
                model=self.ai_client.model,
                messages=[
                    {"role": "system", "content": "You are an analyst. Always return valid JSON with a 'requirements' array of strings."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = json.loads(raw)
            reqs = data.get("requirements") or []
            return [r if isinstance(r, str) else str(r) for r in reqs]
        except Exception:
            return []

    def _pageindex_chat_requirements_for_section(
        self,
        doc_id: str,
        section_title: str,
        section_text: str,
    ) -> List[str]:
        """Use PageIndex Chat API to extract requirements relevant to a section/topic."""
        if not (self.pageindex_client and doc_id):
            return []
        excerpt = (section_text or "").strip()
        excerpt = excerpt[:3000] + ("..." if len(excerpt) > 3000 else "")
        q = PAGEINDEX_CHAT_SECTION_REQUIREMENTS_QUERY.format(
            section_title=section_title or "Untitled",
            section_preview=excerpt or "(empty)",
        )
        try:
            content = self.pageindex_client.chat(doc_id=doc_id, message=q, enable_citations=False)
            data = _extract_first_json_object(content) or {}
            return _get_requirements_list(data)
        except Exception:
            return []

    def _pageindex_chat_combined_check(
        self,
        doc_id: str,
        requirement_id: str,
        requirement_text: str,
        section_title: str,
        section_text: str,
    ) -> Dict[str, Any]:
        """Use PageIndex Chat API to produce both omission + contradiction JSON for one requirement."""
        excerpt = (section_text or "").strip()
        excerpt = excerpt[:8000] + ("..." if len(excerpt) > 8000 else "")
        q = PAGEINDEX_CHAT_COMBINED_CHECK_QUERY.format(
            requirement_id=requirement_id,
            requirement_text=requirement_text or "",
            section_title=section_title or "Untitled",
            section_excerpt=excerpt or "(empty)",
        )
        try:
            content = self.pageindex_client.chat(doc_id=doc_id, message=q, enable_citations=False)
            data = _extract_first_json_object(content) or {}
            omission = _get_nested_dict(data, "omission")
            contradiction = _get_nested_dict(data, "contradiction")
            # Ensure IDs exist and align
            omission["requirement_id"] = omission.get("requirement_id") or requirement_id
            contradiction["requirement_id"] = contradiction.get("requirement_id") or requirement_id
            return {"omission": omission, "contradiction": contradiction}
        except Exception as e:
            return {
                "omission": {
                    "requirement_id": requirement_id,
                    "status": "ERROR",
                    "confidence": 0.0,
                    "justification": f"PageIndex Chat error: {str(e)}",
                    "citations": [],
                    "missing_elements": [],
                },
                "contradiction": {
                    "requirement_id": requirement_id,
                    "has_contradiction": False,
                    "severity": "ERROR",
                    "contradiction_details": f"PageIndex Chat error: {str(e)}",
                    "reference_guideline": "",
                    "tender_statement": "",
                    "citations": [],
                    "recommendation": "",
                },
            }

    def _breakdown_node(self, state: TenderCheckState) -> Dict[str, Any]:
        """Break down: submission section index (PageIndex tree or LLM), or requirement extraction (RAG)."""
        use_pageindex = state.get("use_pageindex_chat", False) or state.get("use_pageindex_our", False)
        submission_doc_id = (state.get("submission_doc_id") or "").strip()
        tender_text = state.get("tender_text") or ""
        try:
            if use_pageindex and self.pageindex_client and submission_doc_id:
                if self.progress_callback:
                    self.progress_callback("breakdown", 1, 4, {"step": "Building submission section index (PageIndex)..."})
                try:
                    tree = get_tree(self.pageindex_client, submission_doc_id, retry_if_not_ready=True, max_retries=8)
                except Exception as e:
                    error_msg = f"Failed to get submission tree from PageIndex: {e}"
                    if self.progress_callback:
                        self.progress_callback("breakdown", 1, 4, {"step": "Breakdown error", "error": error_msg})
                    return {
                        "submission_sections": [],
                        "sections": [],
                        "requirements": [],
                        "tender_summary": "",
                        "error": error_msg,
                    }
                submission_sections = flatten_tree_to_nodes(tree, include_nodes_without_content=False)
                if not submission_sections:
                    error_msg = "Submission tree is empty (no sections found)."
                    if self.progress_callback:
                        self.progress_callback("breakdown", 1, 4, {"step": "Breakdown error", "error": error_msg})
                    return {
                        "submission_sections": [],
                        "sections": [],
                        "requirements": [],
                        "tender_summary": "",
                        "error": error_msg,
                    }
                tender_summary = (
                    f"Submission with {len(submission_sections)} sections (section index from PageIndex)."
                )
                if self.progress_callback:
                    self.progress_callback("breakdown", 1, 4, {
                        "step": "Submission section index ready",
                        "submission_sections_count": len(submission_sections),
                    })
                return {
                    "submission_sections": submission_sections,
                    "sections": submission_sections,
                    "requirements": [],
                    "tender_summary": tender_summary,
                    "error": "",
                }
            if use_pageindex and tender_text:
                if self.progress_callback:
                    self.progress_callback("breakdown", 1, 4, {"step": "Splitting submission into sections..."})
                sections_raw = self._extract_sections(tender_text)
                submission_sections = [
                    {"section_id": s.get("section_id"), "title": s.get("title"), "content": s.get("content", "")}
                    for s in sections_raw
                ]
                tender_summary = f"Submission with {len(submission_sections)} sections (extracted from text)."
                if self.progress_callback:
                    self.progress_callback("breakdown", 1, 4, {
                        "step": "Section extraction complete",
                        "submission_sections_count": len(submission_sections),
                    })
                return {
                    "submission_sections": submission_sections,
                    "sections": submission_sections,
                    "requirements": [],
                    "tender_summary": tender_summary,
                    "error": "",
                }
            # Default: requirement extraction
            if self.progress_callback:
                self.progress_callback("breakdown", 1, 4, {"step": "Breaking down tender into requirements..."})
            result = self.breakdown_agent.breakdown_tender(state["tender_text"])
            requirements = result.get("requirements", [])
            requirements = normalize_requirement_ids(requirements)
            tender_summary = f"Tender document with {len(requirements)} requirements extracted."
            if self.progress_callback:
                self.progress_callback("breakdown", 1, 4, {"step": "Breakdown complete", "requirements_count": len(requirements)})
            return {
                "submission_sections": [],
                "sections": [],
                "requirements": requirements,
                "tender_summary": tender_summary,
                "error": "",
            }
        except Exception as e:
            if self.progress_callback:
                self.progress_callback("breakdown", 1, 4, {"step": "Breakdown error", "error": str(e)})
            return {
                "submission_sections": [],
                "sections": [],
                "requirements": [],
                "tender_summary": "",
                "error": str(e),
            }

    def _retrieve_one(
        self,
        requirement: Dict[str, Any],
        project_id: str,
        guidelines_project_id: str,
        top_k: int,
    ) -> Dict[str, Any]:
        """Retrieve omission and contradiction chunks for one requirement (for parallel execution)."""
        req_text = requirement.get("requirement_text", "")
        omission_chunks = []
        contradiction_chunks = []
        try:
            omission_chunks = self.supabase_client.get_chunks_by_hybrid_search(
                project_id=project_id,
                query=req_text,
                match_count=top_k
            )
        except Exception:
            pass
        try:
            contradiction_chunks = self.supabase_client.get_chunks_by_hybrid_search(
                project_id=guidelines_project_id,
                query=req_text,
                match_count=top_k
            )
        except Exception:
            pass
        return {
            "requirement": requirement,
            "omission_chunks": omission_chunks,
            "contradiction_chunks": contradiction_chunks,
        }

    def _retrieve_one_pageindex(
        self,
        requirement: Dict[str, Any],
        reference_doc_id: str,
        guidelines_doc_id: str,
    ) -> Dict[str, Any]:
        """Retrieve omission and contradiction evidence via PageIndex Tree RAG (get tree → tree search → context as chunks)."""
        req_text = requirement.get("requirement_text", "")
        omission_q = PAGEINDEX_OMISSION_QUESTION.format(requirement_text=req_text)
        contradiction_q = PAGEINDEX_CONTRADICTION_QUESTION.format(requirement_text=req_text)
        omission_chunks = []
        contradiction_chunks = []
        try:
            context = tree_rag_retrieve(
                self.pageindex_client,
                self.ai_client,
                reference_doc_id,
                omission_q,
            )
            if context:
                omission_chunks = [{"content": context, "file_name": "PageIndex Reference", "page_number": None}]
        except Exception:
            omission_chunks = []
        try:
            con_doc_id = guidelines_doc_id or reference_doc_id
            context = tree_rag_retrieve(
                self.pageindex_client,
                self.ai_client,
                con_doc_id,
                contradiction_q,
            )
            if context:
                contradiction_chunks = [{"content": context, "file_name": "PageIndex Guidelines", "page_number": None}]
        except Exception:
            contradiction_chunks = []
        return {
            "requirement": requirement,
            "omission_chunks": omission_chunks,
            "contradiction_chunks": contradiction_chunks,
        }

    def _retrieval_node_section_based(
        self,
        state: TenderCheckState,
        submission_sections: List[Dict[str, Any]],
        reference_doc_id: str,
        guidelines_doc_id: str,
    ) -> Dict[str, Any]:
        """Submission section by section: for each section get Sentence Text (submission) and Requirement Text (reference); omission + contradiction."""
        results: List[Dict[str, Any]] = []
        total = len(submission_sections)
        for i, section in enumerate(submission_sections):
            if self.progress_callback:
                self.progress_callback("retrieval", 2, 4, {
                    "step": "PageIndex (Our): section-by-section",
                    "total_requirements": total,
                    "completed": i,
                    "current_section": section.get("title", ""),
                })
            section_id = section.get("section_id") or section.get("node_id") or f"SEC-{i+1}"
            title = section.get("title", "") or f"Section {i+1}"
            sentence_text = (section.get("content", "") or "").strip()
            section_preview = (sentence_text[:2000] + "..." if len(sentence_text) > 2000 else sentence_text)
            query = PAGEINDEX_SECTION_REQUIREMENTS_QUERY.format(
                section_title=title,
                section_preview=section_preview,
            )
            reference_requirements_context = ""
            try:
                reference_requirements_context = tree_rag_retrieve(
                    self.pageindex_client,
                    self.ai_client,
                    reference_doc_id,
                    query,
                )
            except Exception:
                pass
            reference_requirement_strings = self._parse_requirements_from_text(reference_requirements_context) if reference_requirements_context else []
            if not reference_requirement_strings and reference_requirements_context:
                reference_requirement_strings = [reference_requirements_context]
            # Omission: for each reference requirement, check for supporting evidence in submission section (sentence text)
            for j, reference_requirement_text in enumerate(reference_requirement_strings):
                req_id = f"SEC-{section_id}-OM-{j+1}"
                results.append({
                    "requirement": {
                        "id": req_id,
                        "requirement_text": reference_requirement_text,
                        "reference_requirement_text": reference_requirement_text,
                        "submission_section_content": sentence_text,
                        "section_title": title,
                        "section_id": section_id,
                    },
                    "omission_chunks": [{"content": sentence_text, "file_name": f"Submission: {title}", "page_number": None}],
                    "contradiction_chunks": [],
                })
            # Contradiction: for each submission section (sentence text), check compliance with reference requirements
            results.append({
                "requirement": {
                    "id": f"SEC-{section_id}-CON",
                    "requirement_text": sentence_text[:12000] + ("..." if len(sentence_text) > 12000 else ""),
                    "submission_section_content": sentence_text[:12000] + ("..." if len(sentence_text) > 12000 else ""),
                    "section_title": title,
                    "section_id": section_id,
                },
                "omission_chunks": [],
                "contradiction_chunks": [{"content": reference_requirements_context or "(No reference requirements retrieved)", "file_name": f"Reference: {title}", "page_number": None}],
            })
        # Fallback: fill empty chunks from Supabase when possible
        project_id = (state.get("project_id") or "").strip()
        guidelines_id = (state.get("guidelines_project_id") or "").strip()
        top_k = state.get("top_k", 8)
        if project_id and project_id != "n/a" and self.supabase_client:
            for item in results:
                req = item.get("requirement", {})
                req_text = req.get("requirement_text", "")
                if not item.get("omission_chunks") and project_id:
                    try:
                        item["omission_chunks"] = self.supabase_client.get_chunks_by_hybrid_search(
                            project_id=project_id, query=req_text, match_count=top_k,
                        )
                    except Exception:
                        pass
                if not item.get("contradiction_chunks") and guidelines_id:
                    try:
                        item["contradiction_chunks"] = self.supabase_client.get_chunks_by_hybrid_search(
                            project_id=guidelines_id, query=req_text, match_count=top_k,
                        )
                    except Exception:
                        pass
        if self.progress_callback:
            self.progress_callback("retrieval", 2, 4, {
                "step": "PageIndex (Our) retrieval complete",
                "total_requirements": len(results),
                "completed": len(results),
            })
        return {"retrieval_results": results}

    def _retrieval_node(self, state: TenderCheckState) -> Dict[str, Any]:
        """Retrieval: submission section-by-section (PageIndex API or Our), or requirement-based (RAG), or Supabase RAG."""
        requirements = state.get("requirements", [])
        submission_sections = state.get("submission_sections", []) or state.get("sections", [])
        use_pageindex_chat = state.get("use_pageindex_chat", False) and self.pageindex_client
        use_pageindex_our = state.get("use_pageindex_our", False) and self.pageindex_client
        use_pageindex = use_pageindex_chat or use_pageindex_our
        reference_doc_id = (state.get("reference_doc_id") or "").strip()
        guidelines_doc_id = (state.get("guidelines_doc_id") or "").strip()
        error = state.get("error", "")

        # PageIndex (Our): tree + tree_rag_retrieve + our agents (section-based)
        if use_pageindex_our and reference_doc_id and submission_sections:
            return self._retrieval_node_section_based(
                state, submission_sections, reference_doc_id, guidelines_doc_id or reference_doc_id,
            )

        # PageIndex (API) Chat workflow: tree/section-driven + PageIndex Chat API
        # - For each section, ask Chat for requirements; for each (section, requirement), one Chat call returns omission+contradiction JSON
        if use_pageindex_chat:
            # If no explicit reference_doc_id, default to the submission doc_id (single-doc mode)
            chat_doc_id = reference_doc_id or (state.get("submission_doc_id") or "").strip()

            # If PageIndex expected but breakdown failed (empty sections), return empty results with error
            if chat_doc_id and not submission_sections:
                if error and self.progress_callback:
                    self.progress_callback("retrieval", 2, 4, {
                        "step": "Retrieval skipped due to breakdown error",
                        "error": error,
                    })
                return {"retrieval_results": []}

            if not chat_doc_id or not submission_sections:
                return {"retrieval_results": []}

            results: List[Dict[str, Any]] = []
            total_sections = len(submission_sections)

            for i, section in enumerate(submission_sections):
                section_id = section.get("section_id") or section.get("node_id") or f"SEC-{i+1}"
                title = section.get("title", "") or f"Section {i+1}"
                section_text = (section.get("content", "") or "").strip()
                if self.progress_callback:
                    self.progress_callback("retrieval", 2, 4, {
                        "step": "PageIndex (Chat): extracting requirements per section",
                        "total_requirements": total_sections,
                        "completed": i,
                        "current_section": title,
                    })

                try:
                    requirement_texts = self._pageindex_chat_requirements_for_section(
                        doc_id=chat_doc_id,
                        section_title=title,
                        section_text=section_text,
                    )
                except Exception as e:
                    _debug_log(f"PageIndex Chat requirements for section failed: section_id={section_id}, title={title!r}, error={e}", e)
                    raise
                if not requirement_texts:
                    continue

                # Check each requirement with one combined Chat call (omission + contradiction)
                # Keep concurrency conservative to reduce API flakiness.
                max_workers = min(3, len(requirement_texts))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_req = {}
                    for j, req_text in enumerate(requirement_texts, 1):
                        req_id = f"SEC-{section_id}-REQ-{j}"
                        requirement_obj = {
                            "id": req_id,
                            "requirement_text": req_text,
                            "section_title": title,
                            "section_id": section_id,
                            "submission_section_content": section_text,
                        }
                        fut = executor.submit(
                            self._pageindex_chat_combined_check,
                            chat_doc_id,
                            req_id,
                            req_text,
                            title,
                            section_text,
                        )
                        future_to_req[fut] = requirement_obj

                    for fut in as_completed(future_to_req):
                        req_obj = future_to_req[fut]
                        try:
                            combined = fut.result()
                        except Exception as e:
                            combined = {
                                "omission": {
                                    "requirement_id": req_obj["id"],
                                    "status": "ERROR",
                                    "confidence": 0.0,
                                    "justification": str(e),
                                    "citations": [],
                                    "missing_elements": [],
                                },
                                "contradiction": {
                                    "requirement_id": req_obj["id"],
                                    "has_contradiction": False,
                                    "severity": "ERROR",
                                    "contradiction_details": str(e),
                                    "reference_guideline": "",
                                    "tender_statement": "",
                                    "citations": [],
                                    "recommendation": "",
                                },
                            }

                        results.append({
                            "requirement": req_obj,
                            "pageindex_combined": combined,
                            "omission_chunks": [],
                            "contradiction_chunks": [],
                        })

            if self.progress_callback:
                self.progress_callback("retrieval", 2, 4, {
                    "step": "PageIndex (Chat) retrieval complete",
                    "total_requirements": len(results),
                    "completed": len(results),
                })
            return {"retrieval_results": results}

        # If PageIndex expected but breakdown failed (empty sections), return empty results with error
        if use_pageindex and reference_doc_id and not submission_sections:
            if error:
                if self.progress_callback:
                    self.progress_callback("retrieval", 2, 4, {
                        "step": "Retrieval skipped due to breakdown error",
                        "error": error,
                    })
            return {"retrieval_results": []}

        if not requirements:
            return {"retrieval_results": []}

        if self.progress_callback:
            step_label = "RAG retrieval"
            self.progress_callback("retrieval", 2, 4, {
                "step": step_label,
                "total_requirements": len(requirements),
                "completed": 0
            })

        # Supabase RAG path (existing)
        project_id = state["project_id"]
        guidelines_id = state.get("guidelines_project_id", project_id)
        top_k = state.get("top_k", 8)
        batch_size = min(self.retrieval_batch_size, len(requirements))
        max_workers = min(self.retrieval_max_workers, len(requirements))

        results = [None] * len(requirements)
        completed_count = 0
        
        for start in range(0, len(requirements), batch_size):
            batch = requirements[start : start + batch_size]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(
                        self._retrieve_one,
                        req,
                        project_id,
                        guidelines_id,
                        top_k,
                    ): start + i
                    for i, req in enumerate(batch)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                        completed_count += 1
                        if self.progress_callback:
                            self.progress_callback("retrieval", 2, 4, {
                                "step": f"Retrieving chunks",
                                "total_requirements": len(requirements),
                                "completed": completed_count
                            })
                    except Exception as e:
                        results[idx] = {
                            "requirement": requirements[idx],
                            "omission_chunks": [],
                            "contradiction_chunks": [],
                        }
                        completed_count += 1
        
        if self.progress_callback:
            self.progress_callback("retrieval", 2, 4, {
                "step": "Retrieval complete",
                "total_requirements": len(requirements),
                "completed": completed_count
            })
        
        return {"retrieval_results": results}

    def _check_node(self, state: TenderCheckState) -> Dict[str, Any]:
        """Run omission and contradiction LLM per requirement using shared retrieval; single node, one orchestrator call."""
        retrieval_results = state.get("retrieval_results", [])
        if not retrieval_results:
            return {"omission_results": [], "contradiction_results": []}

        # PageIndex Chat-only path: retrieval already contains combined omission+contradiction JSON.
        if (state.get("use_pageindex_chat", False) and self.pageindex_client
            and retrieval_results and isinstance(retrieval_results[0], dict)
            and retrieval_results[0].get("pageindex_combined") is not None):
            omission_results: List[Dict[str, Any]] = []
            contradiction_results: List[Dict[str, Any]] = []
            for item in retrieval_results:
                combined = item.get("pageindex_combined") or {}
                omission_results.append(combined.get("omission") or {})
                contradiction_results.append(combined.get("contradiction") or {})
            return {"omission_results": omission_results, "contradiction_results": contradiction_results}

        if self.progress_callback:
            self.progress_callback("check", 3, 4, {
                "step": "Starting requirement checks",
                "total_requirements": len(retrieval_results),
                "completed": 0
            })

        omission_results = [None] * len(retrieval_results)
        contradiction_results = [None] * len(retrieval_results)
        completed_count = 0

        def process_one(idx: int, item: Dict[str, Any]) -> None:
            nonlocal completed_count
            req = item["requirement"]
            om_chunks = item.get("omission_chunks", [])
            con_chunks = item.get("contradiction_chunks", [])
            req_id = req.get("id", "UNKNOWN")
            if not om_chunks:
                omission_results[idx] = {
                    "requirement_id": req_id,
                    "status": "NOT_FULFILLED",
                    "confidence": 0.0,
                    "justification": "No reference material was retrieved for this requirement; cannot assess.",
                    "citations": [],
                    "missing_elements": [],
                }
            else:
                try:
                    omission_results[idx] = self.omission_checker.check_requirement_with_chunks(
                        requirement=req,
                        chunks=om_chunks,
                        openai_semaphore=self.openai_semaphore,
                    )
                except Exception as e:
                    omission_results[idx] = {
                        "requirement_id": req_id,
                        "status": "ERROR",
                        "confidence": 0.0,
                        "justification": str(e),
                        "citations": [],
                        "missing_elements": [],
                    }
            if not con_chunks:
                contradiction_results[idx] = {
                    "requirement_id": req_id,
                    "has_contradiction": False,
                    "severity": "NO_CONTRADICTION",
                    "contradiction_details": "No guideline material was retrieved; contradiction not assessed.",
                    "reference_guideline": "",
                    "tender_statement": "",
                    "citations": [],
                    "recommendation": "",
                }
            else:
                try:
                    contradiction_results[idx] = self.contradiction_checker.check_requirement_with_chunks(
                        requirement=req,
                        chunks=con_chunks,
                        openai_semaphore=self.openai_semaphore,
                    )
                except Exception as e:
                    contradiction_results[idx] = {
                        "requirement_id": req_id,
                        "has_contradiction": False,
                        "severity": "ERROR",
                        "contradiction_details": str(e),
                        "reference_guideline": "",
                        "tender_statement": "",
                        "citations": [],
                        "recommendation": "",
                    }
            completed_count += 1
            if self.progress_callback:
                self.progress_callback("check", 3, 4, {
                    "step": f"Checking requirements",
                    "total_requirements": len(retrieval_results),
                    "completed": completed_count,
                    "current_requirement": req_id
                })

        with ThreadPoolExecutor(max_workers=min(20, len(retrieval_results))) as executor:
            futures = [
                executor.submit(process_one, i, item)
                for i, item in enumerate(retrieval_results)
            ]
            for f in futures:
                f.result()

        if self.progress_callback:
            self.progress_callback("check", 3, 4, {
                "step": "Check complete",
                "total_requirements": len(retrieval_results),
                "completed": completed_count
            })

        return {
            "omission_results": omission_results,
            "contradiction_results": contradiction_results,
        }

    def _orchestrate_node(self, state: TenderCheckState) -> Dict[str, Any]:
        """Synthesize all results (single call after check node)."""
        try:
            if self.progress_callback:
                self.progress_callback("orchestrate", 4, 4, {
                    "step": "Synthesizing final report..."
                })
            
            omission_results = state.get("omission_results", [])
            contradiction_results = state.get("contradiction_results", [])
            errors = []
            if state.get("error"):
                errors.append(state["error"])
            
            # If there's an error and no results, return error report instead of false COMPLIANT
            use_pageindex = state.get("use_pageindex_chat", False) or state.get("use_pageindex_our", False)
            submission_sections = state.get("submission_sections", [])
            if errors and not omission_results and not contradiction_results:
                if use_pageindex and not submission_sections:
                    error_msg = "; ".join(errors)
                    return {
                        "final_report": normalize_final_report({
                            "overall_status": "ERROR",
                            "compliance_score": 0.0,
                            "summary": f"Workflow failed during breakdown: {error_msg}. No sections were extracted from the submission, so no checks could be performed.",
                            "critical_issues": [],
                            "omission_summary": {
                                "total_requirements": 0,
                                "fulfilled": 0,
                                "partially_fulfilled": 0,
                                "not_fulfilled": 0,
                                "missing_requirements": []
                            },
                            "contradiction_summary": {
                                "total_checked": 0,
                                "critical_contradictions": 0,
                                "moderate_contradictions": 0,
                                "minor_contradictions": 0,
                                "contradictions": []
                            },
                            "recommendations": [],
                            "risk_assessment": "Unable to assess risk due to processing error."
                        }),
                        "error": error_msg,
                    }

            final_report = self.orchestrator.synthesize_results(
                tender_summary=state.get("tender_summary", ""),
                omission_results=omission_results or [],
                contradiction_results=contradiction_results or [],
            )
            
            # Normalize final report schema
            final_report = normalize_final_report(final_report)
            
            error_msg = "; ".join(errors) if errors else ""
            
            if self.progress_callback:
                self.progress_callback("orchestrate", 4, 4, {
                    "step": "Orchestration complete",
                    "overall_status": final_report.get("overall_status", "UNKNOWN")
                })
            
            return {"final_report": final_report, "error": error_msg}
        except Exception as e:
            if self.progress_callback:
                self.progress_callback("orchestrate", 4, 4, {
                    "step": "Orchestration error",
                    "error": str(e)
                })
            return {
                "final_report": normalize_final_report({
                    "overall_status": "ERROR",
                    "summary": f"Error: {str(e)}"
                }),
                "error": f"Orchestration error: {str(e)}"
            }
    
    def run(
        self,
        tender_text: str,
        project_id: str,
        guidelines_project_id: str = None,
        top_k: int = 8,
        use_pageindex_chat: bool = False,
        use_pageindex_our: bool = False,
        reference_doc_id: str = "",
        guidelines_doc_id: str = "",
        submission_doc_id: str = "",
    ) -> Dict[str, Any]:
        """
        Run the complete tender checking workflow.
        
        Args:
            tender_text: Full text of submission (used when submission_doc_id not set for section extraction)
            project_id: Project ID for reference documents (omission) — used when not PageIndex
            guidelines_project_id: Project ID for guidelines (contradiction) — used when not PageIndex
            top_k: Chunks per requirement (RAG mode only)
            use_pageindex_chat: If True, use PageIndex Chat API for requirements and checks (costly).
            use_pageindex_our: If True, use PageIndex tree + our own LLM (tree RAG) for requirements and checks.
            reference_doc_id: PageIndex doc_id for reference document (source of requirement text)
            guidelines_doc_id: PageIndex doc_id for guidelines (contradiction); if empty, uses reference_doc_id
            submission_doc_id: PageIndex doc_id for submission (section index from tree); if empty, sections from tender_text
        Returns:
            Final state with all results
        """
        initial_state: TenderCheckState = {
            "tender_text": tender_text,
            "tender_summary": "",
            "requirements": [],
            "sections": [],
            "submission_sections": [],
            "retrieval_results": [],
            "omission_results": [],
            "contradiction_results": [],
            "final_report": {},
            "project_id": project_id,
            "guidelines_project_id": guidelines_project_id or project_id,
            "top_k": top_k,
            "error": "",
            "use_pageindex_chat": use_pageindex_chat,
            "use_pageindex_our": use_pageindex_our,
            "reference_doc_id": reference_doc_id or "",
            "guidelines_doc_id": guidelines_doc_id or "",
            "submission_doc_id": submission_doc_id or "",
        }
        
        # Run workflow
        final_state = self.workflow.invoke(initial_state)
        return final_state

