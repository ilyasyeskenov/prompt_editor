"""Breakdown agent - extracts requirements from tender document."""
import json
from typing import Dict, Any, List
from clients.ai_client import AIClient
from tender_checker.prompts.agent_prompts import BREAKDOWN_AGENT_PROMPT


class BreakdownAgent:
    """Agent that breaks down tender document into requirements."""
    
    def __init__(self, ai_client: AIClient, prompt_template: str = None):
        self.ai_client = ai_client
        self.prompt_template = prompt_template or BREAKDOWN_AGENT_PROMPT
    
    def breakdown_tender(self, tender_text: str) -> Dict[str, Any]:
        """
        Break down tender document into requirements.
        
        Args:
            tender_text: Full text of the tender submission document
            
        Returns:
            Dictionary with 'requirements' list containing requirement objects
        """
        # Format prompt
        prompt = self.prompt_template.replace("{{tender_text}}", tender_text)
        
        try:
            response = self.ai_client.client.chat.completions.create(
                model=self.ai_client.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Senior Contract Strategist. Always return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            return {
                "error": str(e),
                "requirements": []
            }

