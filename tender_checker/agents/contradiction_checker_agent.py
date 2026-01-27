"""Contradiction checker agent - checks for contradictions using RAG."""
import json
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from clients.ai_client import AIClient
from clients.supabase_client import SupabaseClient
from tender_checker.prompts.agent_prompts import CONTRADICTION_CHECKER_PROMPT


class ContradictionCheckerAgent:
    """Agent that checks for contradictions using RAG search."""
    
    def __init__(self, ai_client: AIClient, supabase_client: SupabaseClient, prompt_template: str = None):
        self.ai_client = ai_client
        self.supabase_client = supabase_client
        self.prompt_template = prompt_template or CONTRADICTION_CHECKER_PROMPT
    
    def check_requirement(
        self,
        requirement: Dict[str, Any],
        project_id: str,
        top_k: int = 7,
        search_method: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Check if a requirement contradicts reference guidelines using RAG.
        
        Args:
            requirement: Requirement dict with 'id' and 'requirement_text'
            project_id: Project ID for RAG search (should point to guidelines)
            top_k: Number of chunks to retrieve
            search_method: "semantic" or "hybrid"
            
        Returns:
            Contradiction check result
        """
        req_text = requirement.get('requirement_text', '')
        req_id = requirement.get('id', 'UNKNOWN')
        
        # Get relevant chunks using RAG
        if search_method == "hybrid":
            chunks = self.supabase_client.get_chunks_by_hybrid_search(
                project_id=project_id,
                query=req_text,
                match_count=top_k
            )
        else:
            chunks = self.supabase_client.search_chunks_by_project_id(
                project_id=project_id,
                query=req_text,
                top_k=top_k
            )
        
        # Format chunks for prompt
        chunk_texts = []
        for i, chunk in enumerate(chunks, 1):
            file_name = chunk.get('file_name') or chunk.get('original_filename', 'Unknown')
            page_num = chunk.get('page_number')
            page_info = f" (page {page_num})" if page_num else ""
            chunk_texts.append(f"Chunk {i} from {file_name}{page_info}:\n{chunk.get('content', '')}")
        
        reference_chunks = "\n\n".join(chunk_texts) if chunk_texts else "No relevant guidelines found."
        
        # Format prompt
        prompt = self.prompt_template.replace("{{requirement_text}}", req_text)
        prompt = prompt.replace("{{requirement_id}}", req_id)
        prompt = prompt.replace("{{reference_chunks}}", reference_chunks)
        
        try:
            response = self.ai_client.client.chat.completions.create(
                model=self.ai_client.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Compliance Auditor specializing in contradiction detection. Always return valid JSON."
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
                "requirement_id": req_id,
                "has_contradiction": False,
                "severity": "NO_CONTRADICTION",
                "contradiction_details": f"Error checking contradiction: {str(e)}",
                "reference_guideline": "",
                "tender_statement": "",
                "citations": [],
                "recommendation": ""
            }
    
    def check_all_requirements(
        self,
        requirements: List[Dict[str, Any]],
        project_id: str,
        top_k: int = 7,
        search_method: str = "hybrid",
        max_workers: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Check all requirements for contradictions in parallel.
        
        Args:
            requirements: List of requirement dicts
            project_id: Project ID for RAG search (guidelines)
            top_k: Number of chunks per requirement
            search_method: "semantic" or "hybrid"
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of contradiction check results (in order of requirements)
        """
        if not requirements:
            return []
        
        # Use ThreadPoolExecutor for parallel processing
        results = [None] * len(requirements)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.check_requirement, req, project_id, top_k, search_method): i
                for i, req in enumerate(requirements)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    req_id = requirements[index].get('id', 'UNKNOWN')
                    results[index] = {
                        "requirement_id": req_id,
                        "has_contradiction": False,
                        "severity": "NO_CONTRADICTION",
                        "contradiction_details": f"Error in parallel processing: {str(e)}",
                        "reference_guideline": "",
                        "tender_statement": "",
                        "citations": [],
                        "recommendation": ""
                    }
        
        return results

