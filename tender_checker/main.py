"""Main entry point for tender checking system."""
from clients.ai_client import AIClient
from clients.supabase_client import SupabaseClient
from tender_checker.workflow import TenderCheckWorkflow
from tender_checker.utils import extract_text_from_pdf_bytes, extract_text_from_pdf_file


def check_tender(
    tender_text: str,
    project_id: str,
    guidelines_project_id: str = None
) -> dict:
    """
    Main function to check a tender submission.
    
    Args:
        tender_text: Full text of tender submission
        project_id: Project ID for reference documents (omission checking)
        guidelines_project_id: Project ID for guidelines (contradiction checking)
        
    Returns:
        Final state with all results
    """
    # Initialize clients
    ai_client = AIClient()
    supabase_client = SupabaseClient()
    
    # Create workflow
    workflow = TenderCheckWorkflow(ai_client, supabase_client)
    
    # Run workflow
    result = workflow.run(
        tender_text=tender_text,
        project_id=project_id,
        guidelines_project_id=guidelines_project_id
    )
    
    return result


def check_tender_from_pdf(
    pdf_bytes: bytes,
    project_id: str,
    guidelines_project_id: str = None
) -> dict:
    """
    Check tender from PDF bytes.
    
    Args:
        pdf_bytes: PDF file as bytes
        project_id: Project ID for reference documents
        guidelines_project_id: Project ID for guidelines
        
    Returns:
        Final state with all results
    """
    # Extract text from PDF
    tender_text = extract_text_from_pdf_bytes(pdf_bytes)
    if not tender_text:
        return {
            "error": "Failed to extract text from PDF",
            "final_report": {
                "overall_status": "ERROR",
                "summary": "Could not extract text from PDF document"
            }
        }
    
    # Run checking workflow
    return check_tender(tender_text, project_id, guidelines_project_id)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python main.py <pdf_path> <project_id> [guidelines_project_id]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    project_id = sys.argv[2]
    guidelines_id = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Extract text
    tender_text = extract_text_from_pdf_file(pdf_path)
    if not tender_text:
        print("Error: Could not extract text from PDF")
        sys.exit(1)
    
    # Run check
    print("Running tender check...")
    result = check_tender(tender_text, project_id, guidelines_id)
    
    # Print results
    print("\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)
    print(f"Overall Status: {result.get('final_report', {}).get('overall_status', 'UNKNOWN')}")
    print(f"Compliance Score: {result.get('final_report', {}).get('compliance_score', 0.0)}")
    print(f"\nSummary:\n{result.get('final_report', {}).get('summary', 'N/A')}")
    
    if result.get('error'):
        print(f"\nError: {result['error']}")

