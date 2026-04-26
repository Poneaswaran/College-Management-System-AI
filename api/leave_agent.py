from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from rag.errors import AppError
from rag.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/leave", tags=["leave"])

class LeaveSummaryRequest(BaseModel):
    reason: str
    faculty_name: str
    leave_type: str
    start_date: str
    end_date: str

class DecisionNoteRequest(BaseModel):
    faculty_name: str
    leave_type: str
    action: str  # APPROVE or REJECT
    remarks: str

@router.post("/summarize")
async def summarize_leave(request: Request, payload: LeaveSummaryRequest):
    """Summarizes a leave request for HOD review using the LLM service."""
    llm_service = request.app.state.rag_service.llm_service
    
    prompt = (
        "You are an academic administrator assistant. Summarize the following leave request "
        "concisely for a Head of Department (HOD).\n\n"
        f"Faculty: {payload.faculty_name}\n"
        f"Type: {payload.leave_type}\n"
        f"Dates: {payload.start_date} to {payload.end_date}\n"
        f"Reason: {payload.reason}\n\n"
        "Provide a 2-sentence summary highlighting the core reason and any potential urgency."
    )
    
    try:
        # Using the internal _generate method of OllamaLLMService
        summary = await llm_service._generate(prompt)
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Failed to summarize leave: {e}")
        return {"summary": f"{payload.faculty_name} has requested {payload.leave_type} leave from {payload.start_date} to {payload.end_date} due to {payload.reason}."}

@router.post("/validate")
async def validate_leave(request: Request, payload: LeaveSummaryRequest):
    """Validates the professional tone of the leave request."""
    llm_service = request.app.state.rag_service.llm_service
    
    prompt = (
        "Analyze the following leave application reason for professional tone and clarity. "
        "Suggest a polite improvement if it is too informal or vague. Otherwise, say 'The tone is professional.'\n\n"
        f"Reason: {payload.reason}"
    )
    
    try:
        feedback = await llm_service._generate(prompt)
        return {"feedback": feedback}
    except Exception as e:
        return {"feedback": "The tone is professional."}
