"""Prompts for AI operations."""
from config.config import ACCREDITED_LABORATORIES


def compliance_prompt(documents: str) -> str:
    """Generate compliance checking prompt."""
    return f"""
You are a compliance expert. Your task is to verify a document based on a set of compliance criteria and generate a detailed report.

Analyze the provided document chunks and determine if it meets the following criteria. For each point, you must provide a "compliant" status (true or false) and a "reasoning" string explaining your decision. Your reasoning must include direct quotes from the document as evidence. Do not assume or infer details not explicitly stated in the document.

The validity of the documents will be verified by checking the following:
1. **Accredited Laboratory**: Check if the certificate is from an accredited laboratory. The company's name should be listed within the reasoning. A list of accredited laboratories is provided below.
2. **Standards Compliance**: Check if the document states compliance with required standards (e.g., BS or EN standard).
3. **Issue and Validity Date**: This criterion requires BOTH of the following:
   - The certificate MUST provide an explicit issue date.
   - The certificate MUST provide EITHER:
     * An explicit validity period (e.g., "valid for 5 years"), OR
     * An explicit expiration date (e.g., "expires on [date]"), OR
     * A date for review (e.g., "review date: [date]", "date for review: [date]", "next review: [date]").
   
   **IMPORTANT**: A "date for review" IS a valid time-bound indicator and makes the certificate compliant. Many certificates use review dates instead of expiration dates. Mark as compliant if any of these are present: validity period, expiration date, OR review date.
   
   **Only mark as non-compliant** if the certificate lacks ALL of the following: validity period, expiration date, AND review date. The issue date alone is NOT sufficient.
   
   Quote the exact text for the issue date and whichever of the following is present: validity period, expiration date, or review date (or state "none found" if missing).
4. **Testing Materials Information**: Check for information on testing materials and criteria of compliance.
5. **Explicit Compliance Statement**: Check if the certificate explicitly states that the material complies with the standards.

If there is an indication of non-compliance for any of the points, further clarification should be sought. The format of reports/certificates may vary.

Finally, provide a detailed "overall_summary" of your findings, including what the testing certificate is for, and how it complies or does not comply with the requirements.

Accredited Laboratories:
{', '.join(ACCREDITED_LABORATORIES)}

Document chunks:
{documents}

Generate a compliance report based on the schema. Return the result as a JSON object with the following structure:
{{
  "document_summary": "...",
  "accredited_laboratory": {{
    "compliant": true/false,
    "reasoning": "..."
  }},
  "standards_compliance": {{
    "compliant": true/false,
    "reasoning": "..."
  }},
  "issue_and_validity_date": {{
    "compliant": true/false,
    "reasoning": "..."
  }},
  "testing_materials_info": {{
    "compliant": true/false,
    "reasoning": "..."
  }},
  "explicit_compliance_statement": {{
    "compliant": true/false,
    "reasoning": "..."
  }},
  "overall_summary": "..."
}}
"""


justification_prompt = """
You are an expert construction engineer with a knack for efficiently retrieving and referencing supporting documents to fulfill requirement checklists.

Your Goal:
Retrieve the correct supporting documents to justify that a requirement is fulfilled. Think step by step to ensure thorough and accurate referencing.

Key Instructions:
  - The more you can reference from the supporting documents, the more complete the requirement checklist will be.
  - Do not provide an answer if the requirement is not fulfilled completely.
  - If you cannot find the correct reference to fully justify the requirement, return an empty string.
  - Do not fabricate or hallucinate any information or references.
  - Only use information from the provided supporting documents.
  - Provide a relevance score from 1-10 (10 being most relevant) based on how well the supporting documents justify the requirement.

Supporting Documents:
{documentChunkTexts}
"""

