"""LangGraph workflow for tender checking multi-agent system."""
import threading
from typing import TypedDict, List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from langgraph.graph import StateGraph, END
from clients.ai_client import AIClient
from clients.supabase_client import SupabaseClient
from tender_checker.agents.breakdown_agent import BreakdownAgent
from tender_checker.agents.omission_checker_agent import OmissionCheckerAgent
from tender_checker.agents.contradiction_checker_agent import ContradictionCheckerAgent
from tender_checker.agents.orchestrator_agent import OrchestratorAgent

# Batching and rate-limit defaults
RETRIEVAL_BATCH_SIZE = 15
RETRIEVAL_MAX_WORKERS = 10
OPENAI_CONCURRENCY = 5

# PageIndex Chat prompts for omission/contradiction (answer-over-doc)
PAGEINDEX_OMISSION_QUESTION = (
    "Does this document state or imply the following requirement? "
    "Quote the relevant parts. Answer: fulfilled / partially / not fulfilled.\n\nRequirement: {requirement_text}"
)
PAGEINDEX_CONTRADICTION_QUESTION = (
    "Does this guideline document contradict or conflict with the following requirement? "
    "Quote any conflicting parts. Answer: no contradiction / minor / moderate / critical contradiction.\n\nRequirement: {requirement_text}"
)


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
    retrieval_results: List[Dict[str, Any]]
    omission_results: List[Dict[str, Any]]
    contradiction_results: List[Dict[str, Any]]
    final_report: Dict[str, Any]
    project_id: str
    guidelines_project_id: str
    top_k: int
    error: str
    # PageIndex Chat (Option B): when True, retrieval uses PageIndex Chat API instead of Supabase RAG
    use_pageindex_chat: bool
    reference_doc_id: str
    guidelines_doc_id: str
    
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
    
    def _breakdown_node(self, state: TenderCheckState) -> Dict[str, Any]:
        """Break down tender into requirements."""
        try:
            if self.progress_callback:
                self.progress_callback("breakdown", 1, 4, {"step": "Breaking down tender into requirements..."})
            
            result = self.breakdown_agent.breakdown_tender(state["tender_text"])
            requirements = result.get("requirements", [])
            
            # Normalize requirement IDs
            requirements = normalize_requirement_ids(requirements)
            
            tender_summary = f"Tender document with {len(requirements)} requirements extracted."
            
            if self.progress_callback:
                self.progress_callback("breakdown", 1, 4, {
                    "step": "Breakdown complete",
                    "requirements_count": len(requirements)
                })
            
            return {
                "requirements": requirements,
                "tender_summary": tender_summary,
                "error": ""
            }
        except Exception as e:
            if self.progress_callback:
                self.progress_callback("breakdown", 1, 4, {"step": "Breakdown error", "error": str(e)})
            return {
                "requirements": [],
                "tender_summary": "",
                "error": f"Breakdown error: {str(e)}"
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
        """Retrieve omission and contradiction evidence via PageIndex Legacy Retrieval API (submit + poll → chunks)."""
        req_text = requirement.get("requirement_text", "")
        omission_q = PAGEINDEX_OMISSION_QUESTION.format(requirement_text=req_text)
        contradiction_q = PAGEINDEX_CONTRADICTION_QUESTION.format(requirement_text=req_text)
        omission_chunks = []
        contradiction_chunks = []
        try:
            omission_chunks = self.pageindex_client.retrieve(
                doc_id=reference_doc_id,
                query=omission_q,
                file_name="PageIndex Reference",
            )
        except Exception:
            omission_chunks = []
        try:
            con_doc_id = guidelines_doc_id or reference_doc_id
            contradiction_chunks = self.pageindex_client.retrieve(
                doc_id=con_doc_id,
                query=contradiction_q,
                file_name="PageIndex Guidelines",
            )
        except Exception:
            contradiction_chunks = []
        return {
            "requirement": requirement,
            "omission_chunks": omission_chunks,
            "contradiction_chunks": contradiction_chunks,
        }

    def _retrieval_node(self, state: TenderCheckState) -> Dict[str, Any]:
        """Parallel retrieval: either Supabase RAG (chunks) or PageIndex Legacy Retrieval (chunks)."""
        requirements = state.get("requirements", [])
        if not requirements:
            return {"retrieval_results": []}

        use_pageindex = state.get("use_pageindex_chat", False) and self.pageindex_client
        reference_doc_id = (state.get("reference_doc_id") or "").strip()
        guidelines_doc_id = (state.get("guidelines_doc_id") or "").strip()

        if self.progress_callback:
            step_label = "PageIndex Retrieval" if use_pageindex else "Retrieving chunks"
            self.progress_callback("retrieval", 2, 4, {
                "step": step_label,
                "total_requirements": len(requirements),
                "completed": 0
            })

        if use_pageindex and reference_doc_id:
            # PageIndex Legacy Retrieval path: submit + poll per requirement per doc → chunks
            batch_size = min(self.retrieval_batch_size, len(requirements))
            max_workers = min(self.retrieval_max_workers, len(requirements))
            results = [None] * len(requirements)
            completed_count = 0
            for start in range(0, len(requirements), batch_size):
                batch = requirements[start : start + batch_size]
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {
                        executor.submit(
                            self._retrieve_one_pageindex,
                            req,
                            reference_doc_id,
                            guidelines_doc_id,
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
                                    "step": "PageIndex Retrieval",
                                    "total_requirements": len(requirements),
                                    "completed": completed_count
                                })
                        except Exception:
                            results[idx] = {
                                "requirement": requirements[idx],
                                "omission_chunks": [],
                                "contradiction_chunks": [],
                            }
                            completed_count += 1
            if self.progress_callback:
                self.progress_callback("retrieval", 2, 4, {
                    "step": "PageIndex Retrieval complete",
                    "total_requirements": len(requirements),
                    "completed": completed_count
                })
            return {"retrieval_results": results}

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
        reference_doc_id: str = "",
        guidelines_doc_id: str = "",
    ) -> Dict[str, Any]:
        """
        Run the complete tender checking workflow.
        
        Args:
            tender_text: Full text of tender submission
            project_id: Project ID for reference documents (omission) — used when use_pageindex_chat is False
            guidelines_project_id: Project ID for guidelines (contradiction) — used when use_pageindex_chat is False
            top_k: Chunks per requirement (RAG mode only)
            use_pageindex_chat: If True, use PageIndex Chat API instead of Supabase RAG
            reference_doc_id: PageIndex doc_id for reference document (omission)
            guidelines_doc_id: PageIndex doc_id for guidelines (contradiction); if empty, uses reference_doc_id
            
        Returns:
            Final state with all results
        """
        initial_state: TenderCheckState = {
            "tender_text": tender_text,
            "tender_summary": "",
            "requirements": [],
            "retrieval_results": [],
            "omission_results": [],
            "contradiction_results": [],
            "final_report": {},
            "project_id": project_id,
            "guidelines_project_id": guidelines_project_id or project_id,
            "top_k": top_k,
            "error": "",
            "use_pageindex_chat": use_pageindex_chat,
            "reference_doc_id": reference_doc_id or "",
            "guidelines_doc_id": guidelines_doc_id or "",
        }
        
        # Run workflow
        final_state = self.workflow.invoke(initial_state)
        return final_state

