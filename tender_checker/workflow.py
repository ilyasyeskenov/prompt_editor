"""LangGraph workflow for tender checking multi-agent system."""
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from clients.ai_client import AIClient
from clients.supabase_client import SupabaseClient
from tender_checker.agents.breakdown_agent import BreakdownAgent
from tender_checker.agents.omission_checker_agent import OmissionCheckerAgent
from tender_checker.agents.contradiction_checker_agent import ContradictionCheckerAgent
from tender_checker.agents.orchestrator_agent import OrchestratorAgent


class TenderCheckState(TypedDict):
    """State for the tender checking workflow."""
    tender_text: str
    tender_summary: str
    requirements: List[Dict[str, Any]]
    omission_results: List[Dict[str, Any]]
    contradiction_results: List[Dict[str, Any]]
    final_report: Dict[str, Any]
    project_id: str
    guidelines_project_id: str
    top_k: int
    error: str
    
class TenderCheckWorkflow:
    """LangGraph workflow for tender checking."""
    
    def __init__(
        self, 
        ai_client: AIClient, 
        supabase_client: SupabaseClient,
        breakdown_prompt: str = None,
        omission_prompt: str = None,
        contradiction_prompt: str = None,
        orchestrator_prompt: str = None
    ):
        self.ai_client = ai_client
        self.supabase_client = supabase_client
        
        # Initialize agents with custom prompts
        self.breakdown_agent = BreakdownAgent(ai_client, breakdown_prompt)
        self.omission_checker = OmissionCheckerAgent(ai_client, supabase_client, omission_prompt)
        self.contradiction_checker = ContradictionCheckerAgent(ai_client, supabase_client, contradiction_prompt)
        self.orchestrator = OrchestratorAgent(ai_client, orchestrator_prompt)
        
        # Build workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(TenderCheckState)
        
        # Add nodes
        workflow.add_node("breakdown", self._breakdown_node)
        workflow.add_node("omission_check", self._omission_check_node)
        workflow.add_node("contradiction_check", self._contradiction_check_node)
        workflow.add_node("orchestrate", self._orchestrate_node)
        
        # Define edges
        # Sequential: breakdown -> parallel checkers -> orchestrator
        workflow.set_entry_point("breakdown")
        workflow.add_edge("breakdown", "omission_check")
        workflow.add_edge("breakdown", "contradiction_check")
        
        # Both checkers can run in parallel, orchestrator waits for both
        # Use a simple approach: orchestrator checks if both are done
        workflow.add_edge("omission_check", "orchestrate")
        workflow.add_edge("contradiction_check", "orchestrate")
        workflow.add_edge("orchestrate", END)
        
        return workflow.compile()
    
    def _breakdown_node(self, state: TenderCheckState) -> Dict[str, Any]:
        """Break down tender into requirements."""
        try:
            result = self.breakdown_agent.breakdown_tender(state["tender_text"])
            requirements = result.get("requirements", [])
            
            # Create summary
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
    
    def _omission_check_node(self, state: TenderCheckState) -> Dict[str, Any]:
        """Check for omissions using RAG."""
        try:
            requirements = state.get("requirements", [])
            if not requirements:
                return {"omission_results": [], "error": ""}
            
            # Check all requirements
            top_k = state.get("top_k", 8)
            results = self.omission_checker.check_all_requirements(
                requirements=requirements,
                project_id=state["project_id"],
                top_k=top_k,
                search_method="hybrid"
            )
            return {"omission_results": results, "error": ""}
        except Exception as e:
            return {
                "omission_results": [],
                "error": f"Omission check error: {str(e)}"
            }
    
    def _contradiction_check_node(self, state: TenderCheckState) -> Dict[str, Any]:
        """Check for contradictions using RAG."""
        try:
            requirements = state.get("requirements", [])
            if not requirements:
                return {"contradiction_results": [], "error": ""}
            
            # Use guidelines_project_id for contradiction checking
            guidelines_id = state.get("guidelines_project_id", state["project_id"])
            
            # Check all requirements
            top_k = state.get("top_k", 8)
            results = self.contradiction_checker.check_all_requirements(
                requirements=requirements,
                project_id=guidelines_id,
                top_k=top_k,
                search_method="hybrid"
            )
            return {"contradiction_results": results, "error": ""}
        except Exception as e:
            return {
                "contradiction_results": [],
                "error": f"Contradiction check error: {str(e)}"
            }
    
    def _orchestrate_node(self, state: TenderCheckState) -> Dict[str, Any]:
        """Synthesize all results."""
        try:
            # Check if already processed (orchestrator may be called twice from parallel edges)
            if state.get("final_report"):
                return {}
            
            omission_results = state.get("omission_results", [])
            contradiction_results = state.get("contradiction_results", [])
            
            # Ensure we have both results (even if empty lists)
            # In LangGraph, when both edges point to same node, it may be called twice
            # We check if both keys exist in state
            if "omission_results" not in state or "contradiction_results" not in state:
                # Not ready yet - one of the checkers hasn't completed
                # Return empty dict, will be called again
                return {}
            
            # Both checkers have completed, synthesize results
            final_report = self.orchestrator.synthesize_results(
                tender_summary=state.get("tender_summary", ""),
                omission_results=omission_results or [],
                contradiction_results=contradiction_results or []
            )
            return {"final_report": final_report, "error": ""}
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

