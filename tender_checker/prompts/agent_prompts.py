"""Prompts for tender checking agents."""

# Used when PageIndex mode and no submission_doc_id: split submission text into sections for section-by-section checking
SECTION_EXTRACTION_PROMPT = """You are a document analyst. Your task is to split the following submission document into logical sections.

Identify clear sections (e.g. by headings, numbered clauses, or topic changes). Each section should have a short title and the full text content of that section. Do not merge unrelated topics into one section; do not split a single heading into multiple sections.

Return your analysis as a JSON object with this structure only:

```json
{
  "sections": [
    {
      "section_id": "SEC-1",
      "title": "Short section title or heading",
      "content": "Full text of this section as it appears in the document."
    }
  ]
}
```

Submission document:
{{tender_text}}

Return only the JSON object, no other text."""

# Parse reference document context into a list of distinct requirement strings (Requirement Text from reference)
REQUIREMENTS_LIST_EXTRACTION_PROMPT = """You are an analyst. The following text is from a reference document (e.g. handbook) and describes requirements that a submission must comply with.

Extract every distinct requirement as a separate item. Each item should be one clear, verifiable obligation or criterion. Do not split a single sentence into multiple items. Do not merge unrelated requirements.

Return a JSON object with a single key "requirements" whose value is an array of strings. Each string is one requirement statement.

Reference text:
{{reference_text}}

Return only valid JSON, e.g. {"requirements": ["Requirement one.", "Requirement two."]}"""

BREAKDOWN_AGENT_PROMPT = """You are a Senior Contract Strategist and Requirements Engineer. Your task is to decompose a tender submission document into requirements that need to be verified.

**What counts as one requirement:** One requirement = one distinct, verifiable obligation or criterion (something that can be checked for compliance). Do not split a single sentence into multiple requirements. Do not merge unrelated obligations into one requirement. If a clause has several sub-bullets that each state a distinct obligation, treat each as a separate requirement; if they restate the same obligation, keep as one.

Analyze the provided tender document and extract every such requirement. Each requirement should be:
- Specific and verifiable
- Self-contained
- Include context (what, who, when, where) where helpful

Return your analysis as a JSON object with the following structure:

```json
{
  "requirements": [
    {
      "id": "REQ-1",
      "category": "Technical | Financial | Legal | Administrative",
      "requirement_text": "The specific requirement statement",
      "context": "Additional context or background information"
    }
  ]
}
```

Tender Document:
{{tender_text}}

Extract all requirements that need to be verified against reference documents."""

OMISSION_CHECKER_PROMPT = """You are a Compliance Auditor. Your task is to determine whether a specific requirement from the reference document is supported by evidence in the submission material below.

**Reference requirement (to check for supporting evidence):**
{{requirement_text}}

**Submission material (evidence):**
The text below is the submission content (sentence text / section) to check. Consider all of it carefully.

{{reference_chunks}}

**Rules:**
- Base your judgment only on the submission material above. Do not assume or infer content that is not present.
- FULFILLED: The submission material clearly states or implies the reference requirement; you can cite specific supporting text.
- PARTIALLY_FULFILLED: The submission material addresses only part of the reference requirement; cite what is covered and what is missing.
- NOT_FULFILLED: The submission material does not support the reference requirement, or is silent on it. If nothing above is relevant, choose NOT_FULFILLED and state clearly that there is no supporting evidence in the provided material.
- Every citation must be an exact quote from the submission material. Do not paraphrase in citations.
- If you find no relevant evidence, say so explicitly in the justification and return an empty or minimal citations list.

Return your analysis as JSON only:

```json
{
  "requirement_id": "{{requirement_id}}",
  "status": "FULFILLED" | "PARTIALLY_FULFILLED" | "NOT_FULFILLED",
  "confidence": 0.0-1.0,
  "justification": "Clear explanation based only on the reference material above.",
  "citations": [
    {
      "source_text": "Exact quote from submission material",
      "document_reference": "Submission section name and page number",
      "relevance": "How this citation supports the reference requirement"
    }
  ],
  "missing_elements": ["Only if PARTIALLY_FULFILLED or NOT_FULFILLED: what is missing or unsupported"]
}
```"""

CONTRADICTION_CHECKER_PROMPT = """You are a Compliance Auditor. Your task is to determine whether the submission statement below contradicts or conflicts with the reference guidelines.

**Submission statement to check:**
{{requirement_text}}

**Reference guidelines:**
The text below is the reference/guideline content. Consider all of it carefully.

{{reference_chunks}}

**Rules:**
- Base your judgment only on the material above. A contradiction must be explicit or clearly implied: the submission says X, the reference guideline says or implies not-X (or the opposite).
- CRITICAL: Direct conflict with a mandatory rule or safety/legal requirement.
- MODERATE: Clear conflict with a stated guideline or standard.
- MINOR: Tension or ambiguity that could be resolved with clarification.
- NO_CONTRADICTION: The guidelines do not conflict with the requirement, or the provided material does not address it. If nothing above contradicts the requirement, choose NO_CONTRADICTION and state clearly that there is no evidence of conflict in the provided material.
- Every citation must be an exact quote. Do not paraphrase in citations.
- If you find no contradiction, say so explicitly in contradiction_details and return empty or minimal citations.

Return your analysis as JSON only:

```json
{
  "requirement_id": "{{requirement_id}}",
  "has_contradiction": true | false,
  "severity": "CRITICAL" | "MODERATE" | "MINOR" | "NO_CONTRADICTION",
  "contradiction_details": "Clear description of the contradiction, or statement that no conflict was found in the provided material.",
  "reference_guideline": "Exact guideline text or clause that is contradicted (if any)",
  "tender_statement": "The part of the requirement that conflicts (if any)",
  "citations": [
    {
      "source_text": "Exact quote from reference guideline",
      "document_reference": "Document name and page number"
    }
  ],
  "recommendation": "Concrete action to resolve the contradiction, or 'No change required' if NO_CONTRADICTION"
}
```"""

# One LLM call per section: assess all relevant requirements and give detailed explanations for non-compliance
SECTION_COMPLIANCE_PROMPT = """You are a Compliance Auditor. For ONE submission section, you will assess it against the reference requirements and return structured verdicts plus detailed explanations when the section is not fully compliant.

**Submission section title:** {{section_title}}
**Submission section ID:** {{section_id}}

**Submission section content (the tender text for this section):**
{{submission_section_content}}

**Reference requirements / guidelines (from the reference document, relevant to this section):**
{{reference_context}}

**Your task:**
1. Identify each distinct requirement from the reference context that applies to this section. **Only include requirements that the contractor or consultant must fulfill.** Do not list requirements that apply to the employer, the project management team, or the client (e.g. establishment of committees by the client, chairing by client officers). If the reference context contains such obligations, omit them from the requirements list.
2. For each requirement, determine:
   - **Omission:** Is the requirement FULFILLED, PARTIALLY_FULFILLED, or NOT_FULFILLED by the submission section? Provide a clear justification and exact citations from the submission where relevant.
   - **Contradiction:** Does the submission contradict the reference? NO_CONTRADICTION, MINOR, MODERATE, or CRITICAL. If there is a contradiction, describe it and cite the conflicting text.
3. **Important:** When something is NOT compliant (omission or contradiction), give a **detailed explanation** in the justification/contradiction_details so the reader understands exactly what is wrong and where. Be specific: quote clauses, state what is missing or conflicting, and why it matters.
4. In **section_detailed_explanations**, provide a concise but informative summary of all non-compliance issues in this section. If the section is fully compliant, say so briefly. If there are gaps or contradictions, explain each one clearly so the tender author knows what to fix.

Return ONLY valid JSON in this exact schema:

```json
{
  "section_id": "{{section_id}}",
  "section_title": "{{section_title}}",
  "requirements": [
    {
      "requirement_id": "SEC-X-REQ-1",
      "requirement_text": "The requirement statement from the reference",
      "omission_status": "FULFILLED" | "PARTIALLY_FULFILLED" | "NOT_FULFILLED",
      "omission_justification": "Clear explanation; cite exact submission text where relevant. For non-compliance, explain in detail what is missing.",
      "omission_citations": [{"source_text": "Exact quote from submission", "relevance": "How it supports or fails the requirement"}],
      "missing_elements": ["What is missing if not fully fulfilled"],
      "contradiction_status": "NO_CONTRADICTION" | "MINOR" | "MODERATE" | "CRITICAL",
      "contradiction_details": "Description of the contradiction and exact conflicting text, or state that no contradiction was found.",
      "contradiction_citations": [],
      "recommendation": "Concrete action to resolve, or 'No change required'"
    }
  ],
  "section_detailed_explanations": "For this section: a clear summary of any non-compliance. List each issue with a short explanation of what is wrong and what should be changed. If fully compliant, state that briefly."
}
```"""

ORCHESTRATOR_PROMPT = """You are a Senior Compliance Review Officer. Your task is to synthesize results from multiple compliance checks and provide a comprehensive assessment of a tender submission.

**Tender Overview:**
{{tender_summary}}

**Omission Check Results:**
{{omission_results}}

**Contradiction Check Results:**
{{contradiction_results}}

**Section-level detailed explanations (when provided):**
{{section_explanations}}

**Instructions:**
1. Review all omission check results in detail, paying attention to `missing_elements`, `status`, and `citations` for each requirement.
2. Review all contradiction check results in detail, paying attention to `severity`, `contradiction_details`, `reference_guideline`, `tender_statement`, and `citations`.
3. Assess overall compliance status.
4. Identify critical issues that must be addressed. For each critical issue, clearly describe what is non-compliant, referencing the original requirement and the specific missing or contradicting clauses from the source documents. Use the section-level detailed explanations above where they provide clearer or more specific descriptions of non-compliance.
5. Provide actionable, concrete recommendations that explain exactly what should be changed in the tender to become compliant.
6. Generate a risk assessment that explains the practical impact of non-compliance (e.g., operational, regulatory, financial).

Return your analysis as JSON:

```json
{
  "overall_status": "COMPLIANT" | "NON_COMPLIANT" | "CONDITIONALLY_COMPLIANT",
  "compliance_score": 0.0-1.0,
  "summary": "Executive summary of the compliance assessment",
  "critical_issues": [
    {
      "issue_type": "OMISSION" | "CONTRADICTION",
      "requirement_id": "REQ-X",
      "severity": "CRITICAL" | "MODERATE" | "MINOR",
      "description": "Concrete explanation of the non-compliance, including which parts of the requirement are not met or are contradicted, and pointing to specific clauses if possible.",
      "impact": "Clear explanation of the impact on the tender submission (e.g., regulatory breach, safety risk, performance degradation, commercial risk)."
    }
  ],
  "omission_summary": {
    "total_requirements": 0,
    "fulfilled": 0,
    "partially_fulfilled": 0,
    "not_fulfilled": 0,
    "missing_requirements": ["List of unfulfilled requirements"]
  },
  "contradiction_summary": {
    "total_checked": 0,
    "critical_contradictions": 0,
    "moderate_contradictions": 0,
    "minor_contradictions": 0,
    "contradictions": ["List of contradictions found"]
  },
  "recommendations": [
    {
      "priority": "HIGH" | "MEDIUM" | "LOW",
      "action": "Specific action required",
      "requirement_id": "REQ-X"
    }
  ],
  "risk_assessment": "Overall risk level and explanation"
}
```"""

