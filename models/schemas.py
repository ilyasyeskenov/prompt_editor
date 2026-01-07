"""Pydantic schemas for compliance and requirement checking."""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class ComplianceCheck(BaseModel):
    """Individual compliance check result."""
    compliant: bool = Field(description="Whether the document is compliant for this specific check.")
    reasoning: str = Field(description="The reasoning for the compliance status of this check.")


class ComplianceReport(BaseModel):
    """Complete compliance report schema."""
    document_summary: str = Field(
        description="A summary of the document. Include the name of the document, the date of issue, and what it is about."
    )
    accredited_laboratory: ComplianceCheck = Field(
        description="Check if the certificate is from an accredited laboratory. The company's name should be listed within the text"
    )
    standards_compliance: ComplianceCheck = Field(
        description="Check if the document states compliance with required standards (e.g., BS or EN)."
    )
    issue_and_validity_date: ComplianceCheck = Field(
        description="Check if the certificate provides an issue date, and either a validity period, an expiration date, or a date for review for it to comply."
    )
    testing_materials_info: ComplianceCheck = Field(
        description="Check for information on testing materials and criteria of compliance."
    )
    explicit_compliance_statement: ComplianceCheck = Field(
        description="Check if the certificate explicitly states that the material complies with the standards."
    )
    overall_summary: str = Field(
        description="A detailed overall summary of what the testing certificate is, and how it complies or does not comply with the requirements."
    )


class ChunkReference(BaseModel):
    """Reference to a document chunk."""
    id: str = Field(description="Unique identifier for the chunk")
    quote: str = Field(description="The relevant quote from the chunk")


class Justification(BaseModel):
    """Justification for a requirement check."""
    id: Optional[str] = Field(default=None, description="Unique identifier for this justification")
    requirement: str = Field(description="The requirement the user asked for, as it is")
    chunk_ids: List[ChunkReference] = Field(
        description="The chunk IDs of the relevant chunks and the quotes from the chunks that are relevant to the requirement"
    )
    justification: str = Field(
        description="The justification on why those documents and pages are selected to be relevant to the requirement. When mentioning the source or referring to the source, use the file_name"
    )
    score: int = Field(
        ge=0, le=10,
        description="The score of how close the reference is to the requirement from 1 to 10. 10 being the closest"
    )


class SourceObject(BaseModel):
    """Source object with detailed information."""
    id: str
    content: str
    file_name: Optional[str] = None
    page_number: Optional[int] = None


class JustificationWithSources(Justification):
    """Justification with source objects."""
    sources: List[SourceObject] = Field(default_factory=list)

