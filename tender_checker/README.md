# Tender Checking Multi-Agent System

A multi-agent system built with LangGraph for checking tender submissions against reference documents and guidelines.

## Architecture

The system uses a multi-agent workflow with the following components:

### Workflow

```
Tender PDF → Breakdown Agent → Requirements List
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
        Omission Checker (RAG)      Contradiction Checker (RAG)
                    ↓                               ↓
                    └───────────────┬───────────────┘
                                    ↓
                            Orchestrator Agent
                                    ↓
                            Final Compliance Report
```

### Agents

1. **Breakdown Agent** (`breakdown_agent.py`)
   - Extracts requirements from tender document
   - Categorizes requirements (Technical, Financial, Legal, Administrative)
   - Returns structured JSON with requirement IDs and context

2. **Omission Checker Agent** (`omission_checker_agent.py`)
   - Checks if each requirement is fulfilled in reference documents
   - Uses RAG (hybrid search) to find relevant document chunks
   - Returns status: FULFILLED, PARTIALLY_FULFILLED, or NOT_FULFILLED
   - Provides citations and missing elements

3. **Contradiction Checker Agent** (`contradiction_checker_agent.py`)
   - Checks if tender requirements contradict reference guidelines
   - Uses RAG (hybrid search) to find relevant guideline chunks
   - Returns severity: CRITICAL, MODERATE, MINOR, or NO_CONTRADICTION
   - Provides contradiction details and recommendations

4. **Orchestrator Agent** (`orchestrator_agent.py`)
   - Synthesizes results from both checkers
   - Generates overall compliance assessment
   - Provides risk assessment and actionable recommendations
   - Returns comprehensive final report

### Workflow Implementation

The workflow is implemented using LangGraph (`workflow.py`):

- **State**: `TenderCheckState` - TypedDict containing all workflow state
- **Nodes**: Each agent is a node in the graph
- **Edges**: Sequential and parallel execution paths
- **Execution**: Breakdown → Parallel (Omission + Contradiction) → Orchestrator

## Usage

### Basic Usage

```python
from tender_checker.main import check_tender

result = check_tender(
    tender_text="Full text of tender submission...",
    project_id="your-project-id",
    guidelines_project_id="guidelines-project-id"  # Optional
)

print(result["final_report"])
```

### From PDF

```python
from tender_checker.main import check_tender_from_pdf

with open("tender.pdf", "rb") as f:
    pdf_bytes = f.read()

result = check_tender_from_pdf(
    pdf_bytes=pdf_bytes,
    project_id="your-project-id",
    guidelines_project_id="guidelines-project-id"
)
```

### Command Line

```bash
python -m tender_checker.main path/to/tender.pdf project-id [guidelines-project-id]
```

## Configuration

The system uses the same configuration as the main app:

- **OpenAI API Key**: Set in `.env` as `OPENAI_API_KEY`
- **Supabase**: Set `SUPABASE_URL` and `SUPABASE_ANON_KEY` in `.env`
- **Project IDs**: 
  - `project_id`: For reference documents (omission checking)
  - `guidelines_project_id`: For guidelines (contradiction checking). If not provided, uses `project_id`

## RAG Integration

Both checker agents use the existing Supabase RAG infrastructure:

- **Hybrid Search**: Uses `get_chunks_by_hybrid_search()` for better accuracy
- **Semantic Search**: Falls back to `search_chunks_by_project_id()` if needed
- **Chunk Retrieval**: Top-K chunks per requirement (default: 8)

## Output Format

The final report includes:

```json
{
  "overall_status": "COMPLIANT | NON_COMPLIANT | CONDITIONALLY_COMPLIANT",
  "compliance_score": 0.0-1.0,
  "summary": "Executive summary",
  "critical_issues": [...],
  "omission_summary": {...},
  "contradiction_summary": {...},
  "recommendations": [...],
  "risk_assessment": "..."
}
```

## File Structure

```
tender_checker/
├── __init__.py
├── main.py                 # Entry point
├── workflow.py             # LangGraph workflow
├── utils.py                # PDF processing utilities
├── README.md               # This file
├── agents/
│   ├── __init__.py
│   ├── breakdown_agent.py
│   ├── omission_checker_agent.py
│   ├── contradiction_checker_agent.py
│   └── orchestrator_agent.py
└── prompts/
    ├── __init__.py
    └── agent_prompts.py    # All agent prompts
```

## Dependencies

- `langgraph>=0.0.40` - Workflow orchestration
- `openai>=1.3.0` - LLM API
- `supabase>=2.0.0` - RAG backend
- `pypdf2>=3.0.0` - PDF processing

## Future Improvements

1. **Async Processing**: Parallel requirement checking within each agent
2. **Caching**: Cache RAG results for repeated requirements
3. **Batch Processing**: Process requirements in batches for efficiency
4. **Streaming**: Stream results as they become available
5. **UI Integration**: Streamlit page for interactive tender checking

