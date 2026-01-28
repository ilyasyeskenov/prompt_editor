"""LangGraph workflow for tender checking multi-agent system."""
import threading
from typing import TypedDict, List, Dict, Any
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
    ):
        self.ai_client = ai_client
        self.supabase_client = supabase_client
        self.retrieval_batch_size = retrieval_batch_size
        self.retrieval_max_workers = retrieval_max_workers
        self.openai_semaphore = threading.Semaphore(openai_concurrency)

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
            result = self.breakdown_agent.breakdown_tender(state["tender_text"])
            requirements = result.get("requirements", [])
            tender_summary = f"Tender document with {len(requirements)} requirements extracted."
            return {
                "requirements": requirements,
                "tender_summary": tender_summary,
                "error": ""
            }
        except Exception as e:
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

    def _retrieval_node(self, state: TenderCheckState) -> Dict[str, Any]:
        """Parallel retrieval (Supabase) for all requirements in batches; single output for downstream check."""
        requirements = state.get("requirements", [])
        if not requirements:
            return {"retrieval_results": []}

        project_id = state["project_id"]
        guidelines_id = state.get("guidelines_project_id", project_id)
        top_k = state.get("top_k", 8)
        batch_size = min(self.retrieval_batch_size, len(requirements))
        max_workers = min(self.retrieval_max_workers, len(requirements))

        results = [None] * len(requirements)
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
                    except Exception as e:
                        results[idx] = {
                            "requirement": requirements[idx],
                            "omission_chunks": [],
                            "contradiction_chunks": [],
                        }
        return {"retrieval_results": results}

    def _check_node(self, state: TenderCheckState) -> Dict[str, Any]:
        """Run omission and contradiction LLM per requirement using shared retrieval; single node, one orchestrator call."""
        retrieval_results = state.get("retrieval_results", [])
        if not retrieval_results:
            return {"omission_results": [], "contradiction_results": []}

        omission_results = [None] * len(retrieval_results)
        contradiction_results = [None] * len(retrieval_results)

        def process_one(idx: int, item: Dict[str, Any]) -> None:
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

        with ThreadPoolExecutor(max_workers=len(retrieval_results)) as executor:
            futures = [
                executor.submit(process_one, i, item)
                for i, item in enumerate(retrieval_results)
            ]
            for f in futures:
                f.result()

        return {
            "omission_results": omission_results,
            "contradiction_results": contradiction_results,
        }

    def _orchestrate_node(self, state: TenderCheckState) -> Dict[str, Any]:
        """Synthesize all results (single call after check node)."""
        try:
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
            error_msg = "; ".join(errors) if errors else ""
            return {"final_report": final_report, "error": error_msg}
        except Exception as e:
            return {
                "final_report": {
                    "overall_status": "ERROR",
                    "summary": f"Error: {str(e)}"
                },
                "error": f"Orchestration error: {str(e)}"
            }
    
    def run(
        self,
        tender_text: str,
        project_id: str,
        guidelines_project_id: str = None,
        top_k: int = 8
    ) -> Dict[str, Any]:
        """
        Run the complete tender checking workflow.
        
        Args:
            tender_text: Full text of tender submission
            project_id: Project ID for reference documents (omission checking)
            guidelines_project_id: Project ID for guidelines (contradiction checking). 
                                  If None, uses project_id.
            
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
            "error": ""
        }
        
        # Run workflow
        final_state = self.workflow.invoke(initial_state)
        return final_state

