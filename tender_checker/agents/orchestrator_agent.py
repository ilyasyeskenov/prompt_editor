"""Orchestrator agent - synthesizes results from all checkers."""
import json
from typing import Dict, Any, List
from clients.ai_client import AIClient
from tender_checker.prompts.agent_prompts import ORCHESTRATOR_PROMPT


class OrchestratorAgent:
    """Agent that synthesizes results from omission and contradiction checkers."""
    
    def __init__(self, ai_client: AIClient, prompt_template: str = None):
        self.ai_client = ai_client
        self.prompt_template = prompt_template or ORCHESTRATOR_PROMPT
    
    def synthesize_results(
        self,
        tender_summary: str,
        omission_results: List[Dict[str, Any]],
        contradiction_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesize results from all checkers into final report.
        
        Args:
            tender_summary: Brief summary of the tender submission
            omission_results: List of omission check results
            contradiction_results: List of contradiction check results
            
        Returns:
            Final compliance assessment report
        """
        # Format results for prompt
        omission_json = json.dumps(omission_results, indent=2)
        contradiction_json = json.dumps(contradiction_results, indent=2)
        
        # Format prompt
        prompt = self.prompt_template.replace("{{tender_summary}}", tender_summary)
        prompt = prompt.replace("{{omission_results}}", omission_json)
        prompt = prompt.replace("{{contradiction_results}}", contradiction_json)
        
        try:
            response = self.ai_client.client.chat.completions.create(
                model=self.ai_client.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Senior Compliance Review Officer. Always return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            return {
                "overall_status": "ERROR",
                "compliance_score": 0.0,
                "summary": f"Error synthesizing results: {str(e)}",
                "critical_issues": [],
                "omission_summary": {},
                "contradiction_summary": {},
                "recommendations": [],
                "risk_assessment": "Unable to assess risk due to processing error"
            }

