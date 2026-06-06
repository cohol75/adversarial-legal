"""
FastAPI 后端：对抗式法律辩论系统。
"""

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json

from graph import (
    run_debate_stream, DebateState,
    classify_case_only, generate_user_questions, generate_side_summary,
)


app = FastAPI(title="对抗式法律辩论系统")

# 静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")


class ModelSettings(BaseModel):
    base_url: str
    api_key: str
    model_name: str


class ClassifyRequest(BaseModel):
    case_description: str
    model: ModelSettings


class QuestionsRequest(BaseModel):
    case_description: str
    category: str
    client_label: str      # 被提问人的标签（如"骑车人"）
    client_role: str       # 被提问人的角色（"原告方"或"被告方"）
    opponent_label: str    # 对方当事人的标签
    opponent_role: str     # 对方当事人的角色
    is_friendly: bool      # True=友好语气(同方律师), False=对抗语气(对方律师)
    model: ModelSettings


class DebateStartRequest(BaseModel):
    case_description: str
    category: str
    plaintiff_answers: str = ""
    defense_answers: str = ""
    model1_base_url: str
    model1_api_key: str
    model1_model_name: str
    model2_base_url: str
    model2_api_key: str
    model2_model_name: str


class DebateRoundRecord(BaseModel):
    round: int
    plaintiff_questions: str = ""
    defense_questions: str = ""
    plaintiff_argument: str = ""
    defense_argument: str = ""
    judge_verdict: str = ""
    judge_model: str = ""


class SummaryRequest(BaseModel):
    case_description: str
    category: str
    plaintiff_answers: str = ""
    defense_answers: str = ""
    user_side: str  # "plaintiff" or "defense"
    debate_records: list  # list of dicts with round data
    model1: ModelSettings
    model2: ModelSettings


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/api/classify")
async def classify_case(req: ClassifyRequest):
    result = await classify_case_only(
        req.case_description,
        {"base_url": req.model.base_url, "api_key": req.model.api_key,
         "model_name": req.model.model_name},
    )
    return result


@app.post("/api/generate-questions")
async def generate_questions_endpoint(req: QuestionsRequest):
    questions = await generate_user_questions(
        req.case_description, req.category,
        req.client_label, req.client_role,
        req.opponent_label, req.opponent_role,
        {"base_url": req.model.base_url, "api_key": req.model.api_key,
         "model_name": req.model.model_name},
        req.is_friendly,
    )
    return {"questions": questions}


@app.post("/api/summary")
async def generate_summary_endpoint(req: SummaryRequest):
    records_dicts = []
    for rec in req.debate_records:
        if isinstance(rec, DebateRoundRecord):
            records_dicts.append(rec.model_dump())
        elif isinstance(rec, dict):
            records_dicts.append(rec)
        else:
            records_dicts.append({"round": 0})  # fallback

    result = await generate_side_summary(
        case_description=req.case_description,
        category=req.category,
        plaintiff_answers=req.plaintiff_answers,
        defense_answers=req.defense_answers,
        debate_records=records_dicts,
        user_side=req.user_side,
        model1_config={"base_url": req.model1.base_url, "api_key": req.model1.api_key,
                       "model_name": req.model1.model_name},
        model2_config={"base_url": req.model2.base_url, "api_key": req.model2.api_key,
                       "model_name": req.model2.model_name},
    )
    return result


@app.post("/debate/start")
async def debate_start(req: DebateStartRequest):
    state: DebateState = {
        "case_description": req.case_description,
        "category": req.category,
        "plaintiff_answers": req.plaintiff_answers,
        "defense_answers": req.defense_answers,
        "model1_config": {
            "base_url": req.model1_base_url,
            "api_key": req.model1_api_key,
            "model_name": req.model1_model_name,
        },
        "model2_config": {
            "base_url": req.model2_base_url,
            "api_key": req.model2_api_key,
            "model_name": req.model2_model_name,
        },
        "current_round": 0,
        "plaintiff_questions": [],
        "defense_questions": [],
        "plaintiff_arguments": [],
        "defense_arguments": [],
        "judge_verdicts": [],
        "judge_model_labels": [],
        "final_summary": "",
        "summary_model_label": "",
    }

    async def event_stream():
        try:
            async for sse_msg in run_debate_stream(state, skip_classify=True):
                yield sse_msg
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8899)
