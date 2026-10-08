from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graph.workflow import graph


app = FastAPI(
    title="AI Quantitative Research Agent",
    description=(
        "Multi-agent quantitative research "
        "system for financial market analysis."
    ),
    version="1.0.0"
)


class ResearchRequest(BaseModel):
    query: str


class ResearchResponse(BaseModel):
    query: str
    ticker: str
    report: str


@app.get("/")
def root():

    return {
        "name": "AI Quantitative Research Agent",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post(
    "/research",
    response_model=ResearchResponse
)
def research(
    request: ResearchRequest
):

    if not request.query.strip():

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    initial_state = {
        "query": request.query,
        "errors": [],
        "retry_count": 0
    }

    try:

        result = graph.invoke(
            initial_state
        )

        errors = result.get(
            "errors",
            []
        )

        if errors:
            raise HTTPException(
                status_code=500,
                detail=errors
            )

        ticker = result.get(
            "ticker"
        )

        report = result.get(
            "final_report"
        )

        if not ticker:
            raise HTTPException(
                status_code=500,
                detail="Ticker extraction failed."
            )

        if not report:
            raise HTTPException(
                status_code=500,
                detail="Research report generation failed."
            )

        return ResearchResponse(
            query=request.query,
            ticker=ticker,
            report=report
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )