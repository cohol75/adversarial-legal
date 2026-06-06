"""
LangGraph 对抗式法律辩论工作流。

节点：案件分类 → 原告追问 → 被告追问 → 原告论证 → 被告论证 → 法官裁决（循环 3 轮）→ 最终总结
流式输出：通过 async generator yield SSE 事件。
"""

import asyncio
import json
import random
import re
from typing import TypedDict, AsyncGenerator

import httpx
from openai import OpenAI

from prompts import (
    CLASSIFY_PROMPT,
    FRIENDLY_QUESTIONS_PROMPT,
    ADVERSARIAL_QUESTIONS_PROMPT,
    PLAINTIFF_QUESTIONS_PROMPT,
    PLAINTIFF_ARGUMENT_PROMPT,
    DEFENSE_QUESTIONS_PROMPT,
    DEFENSE_ARGUMENT_PROMPT,
    JUDGE_PROMPT,
    SUMMARY_PROMPT,
    SIDE_SUMMARY_PROMPT,
)


# ============================================================
# State
# ============================================================

class DebateState(TypedDict):
    case_description: str
    category: str
    plaintiff_answers: str      # 用户（原告方）对追问的回答
    defense_answers: str        # 用户（被告方）对追问的回答
    model1_config: dict
    model2_config: dict
    current_round: int          # 0, 1, 2（循环中由 judge 节点递增）
    plaintiff_questions: list   # 每轮原告追问
    defense_questions: list     # 每轮被告追问
    plaintiff_arguments: list   # 每轮原告论证
    defense_arguments: list     # 每轮被告论证
    judge_verdicts: list        # 每轮法官裁决
    judge_model_labels: list    # 每轮法官所用模型标签
    final_summary: str
    summary_model_label: str


# ============================================================
# LLM 调用工具
# ============================================================

def get_client(config: dict) -> OpenAI:
    return OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
        timeout=httpx.Timeout(60.0, connect=10.0),
    )


def _call_llm_sync(
    config: dict,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    client = get_client(config)
    response = client.chat.completions.create(
        model=config["model_name"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    return response.choices[0].message.content


async def call_llm(
    config: dict,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    return await asyncio.to_thread(
        _call_llm_sync, config, system_prompt, user_prompt, temperature, max_tokens
    )


def _preprocess_for_json(raw: str) -> str:
    """移除代码块标记和反转义引号，返回适合 json.loads 的字符串。"""
    text = raw.strip()
    # 移除 ```json ... ``` 或 ``` ... ``` 代码块包裹
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # 反转义引号：将 \" 替换为 "
    text = text.replace('\\"', '"')
    return text


def parse_questions(raw: str) -> str:
    """解析 LLM 返回的追问 JSON，提取为格式化文本。JSON 解析失败则返回原文。"""
    try:
        text = _preprocess_for_json(raw)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            qs = data.get("questions", [])
            if qs:
                return "\n\n".join(f"**追问 {i+1}**：{q}" for i, q in enumerate(qs))
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return raw


def pick_judge_model(state: DebateState) -> tuple[dict, str]:
    """随机选择法官模型，返回 (config, label)。"""
    if random.random() < 0.5:
        return state["model1_config"], "模型1"
    return state["model2_config"], "模型2"


def pick_summary_model(state: DebateState) -> tuple[dict, str]:
    """随机选择总结模型，返回 (config, label)。"""
    if random.random() < 0.5:
        return state["model1_config"], "模型1"
    return state["model2_config"], "模型2"


def pick_summary_model_for_configs(model1_config: dict, model2_config: dict) -> tuple[dict, str]:
    """随机选择总结模型（不依赖 DebateState），返回 (config, label)。"""
    if random.random() < 0.5:
        return model1_config, "模型1"
    return model2_config, "模型2"


def parse_questions_list(raw: str) -> list:
    """解析 LLM JSON 返回为追问列表。失败则按行拆分。"""
    try:
        text = _preprocess_for_json(raw)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            qs = data.get("questions", [])
            if qs:
                return qs
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    # Fallback: split by line, filter out code blocks and empty lines
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = cleaned.replace('\\"', '"')
    return [q.strip() for q in cleaned.strip().split("\n") if q.strip() and q.strip() not in ("{", "}", "```")]


# ============================================================
# 独立函数（供 REST 端点直接调用）
# ============================================================

async def classify_case_only(case_description: str, model_config: dict) -> dict:
    """独立分类：返回 {{category, plaintiff_label, defendant_label}}。"""
    prompt = CLASSIFY_PROMPT.format(case_description=case_description)
    raw = await call_llm(model_config, "", prompt, temperature=0.1, max_tokens=256)
    valid = [
        "民事-合同纠纷", "民事-侵权纠纷", "民事-婚姻家庭",
        "民事-劳动争议", "民事-知识产权", "刑事-经济犯罪",
        "刑事-人身伤害", "行政-行政处罚",
    ]
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            category = data.get("category", "民事-合同纠纷")
            if category not in valid:
                category = "民事-合同纠纷"
            return {
                "category": category,
                "plaintiff_label": data.get("plaintiff_label", "原告方"),
                "defendant_label": data.get("defendant_label", "被告方"),
            }
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    category = raw.strip().strip("'\"")
    if category not in valid:
        category = "民事-合同纠纷"
    return {"category": category, "plaintiff_label": "原告方", "defendant_label": "被告方"}


async def generate_user_questions(
    case_description: str, category: str,
    client_label: str, client_role: str,
    opponent_label: str, opponent_role: str,
    model_config: dict, is_friendly: bool,
) -> list:
    """生成追问。is_friendly=True 时用友好语气（同方律师），False 时用对抗语气（对方律师）。"""
    prompt_tpl = FRIENDLY_QUESTIONS_PROMPT if is_friendly else ADVERSARIAL_QUESTIONS_PROMPT
    prompt = prompt_tpl.format(
        case_description=case_description, category=category,
        client_label=client_label, client_role=client_role,
        opponent_label=opponent_label, opponent_role=opponent_role,
    )
    raw = await call_llm(model_config, "", prompt, temperature=0.7, max_tokens=1024)
    return parse_questions_list(raw)


async def generate_side_summary(
    case_description: str,
    category: str,
    plaintiff_answers: str,
    defense_answers: str,
    debate_records: list,
    user_side: str,
    model1_config: dict,
    model2_config: dict,
) -> dict:
    """生成按用户立场的最终总结。返回 {summary, summary_model}。"""
    summary_config, label = pick_summary_model_for_configs(model1_config, model2_config)

    all_rounds_parts = []
    for rec in debate_records:
        all_rounds_parts.append(
            f"### 第 {rec['round']} 轮\n\n"
            f"**原告追问**：\n{rec.get('plaintiff_questions', '')}\n\n"
            f"**被告追问**：\n{rec.get('defense_questions', '')}\n\n"
            f"**原告论证**：\n{rec.get('plaintiff_argument', '')}\n\n"
            f"**被告论证**：\n{rec.get('defense_argument', '')}\n\n"
            f"**法官裁决**（{rec.get('judge_model', '')}）：\n{rec.get('judge_verdict', '')}\n\n"
        )

    side_name = "原告" if user_side == "plaintiff" else "被告"
    prompt = SIDE_SUMMARY_PROMPT.format(
        category=category,
        case_description=case_description,
        plaintiff_answers=plaintiff_answers,
        defense_answers=defense_answers,
        all_rounds="\n---\n".join(all_rounds_parts),
        user_side=side_name,
    )
    raw = await call_llm(summary_config, "", prompt, temperature=0.4, max_tokens=3072)
    return {"summary": raw, "summary_model": label}


# ============================================================
# 节点函数
# ============================================================

async def classify_case_node(state: DebateState) -> dict:
    """使用模型1自动判断案件类型和当事人。"""
    result = await classify_case_only(
        state["case_description"],
        {"base_url": state["model1_config"]["base_url"],
         "api_key": state["model1_config"]["api_key"],
         "model_name": state["model1_config"]["model_name"]},
    )
    return {"category": result["category"]}


async def plaintiff_questions_node(state: DebateState) -> dict:
    """原告方提出追问。"""
    round_num = state["current_round"] + 1
    prompt = PLAINTIFF_QUESTIONS_PROMPT.format(
        category=state["category"],
        case_description=state["case_description"],
    )
    raw = await call_llm(state["model1_config"], "", prompt, temperature=0.7, max_tokens=1024)
    parsed = parse_questions(raw)
    return {
        "plaintiff_questions": state["plaintiff_questions"] + [parsed],
    }


async def defense_questions_node(state: DebateState) -> dict:
    """被告方提出追问。"""
    round_num = state["current_round"] + 1
    prompt = DEFENSE_QUESTIONS_PROMPT.format(
        category=state["category"],
        case_description=state["case_description"],
    )
    raw = await call_llm(state["model2_config"], "", prompt, temperature=0.7, max_tokens=1024)
    parsed = parse_questions(raw)
    return {
        "defense_questions": state["defense_questions"] + [parsed],
    }


async def plaintiff_argument_node(state: DebateState) -> dict:
    """原告方准备本轮的论证意见。"""
    round_num = state["current_round"] + 1
    idx = state["current_round"]

    prev_verdict = state["judge_verdicts"][idx - 1] if idx > 0 else "（首轮，无法官意见）"
    opp_arg = state["defense_arguments"][idx - 1] if idx > 0 else "（首轮，对方尚未提交论证）"

    prompt = PLAINTIFF_ARGUMENT_PROMPT.format(
        round_num=round_num,
        category=state["category"],
        case_description=state["case_description"],
        user_plaintiff_answers=state.get("plaintiff_answers", "（无）"),
        user_defense_answers=state.get("defense_answers", "（无）"),
        judge_verdict=prev_verdict,
        opponent_last_argument=opp_arg,
    )
    raw = await call_llm(state["model1_config"], "", prompt, temperature=0.5, max_tokens=2048)
    return {
        "plaintiff_arguments": state["plaintiff_arguments"] + [raw],
    }


async def defense_argument_node(state: DebateState) -> dict:
    """被告方准备本轮的辩护意见。"""
    round_num = state["current_round"] + 1
    idx = state["current_round"]

    prev_verdict = state["judge_verdicts"][idx - 1] if idx > 0 else "（首轮，无法官意见）"
    opp_arg = state["plaintiff_arguments"][idx] if idx < len(state["plaintiff_arguments"]) else "（本轮原告已提交论证，但尚未获取）"

    prompt = DEFENSE_ARGUMENT_PROMPT.format(
        round_num=round_num,
        category=state["category"],
        case_description=state["case_description"],
        user_plaintiff_answers=state.get("plaintiff_answers", "（无）"),
        user_defense_answers=state.get("defense_answers", "（无）"),
        judge_verdict=prev_verdict,
        opponent_last_argument=opp_arg,
    )
    raw = await call_llm(state["model2_config"], "", prompt, temperature=0.5, max_tokens=2048)
    return {
        "defense_arguments": state["defense_arguments"] + [raw],
    }


async def judge_verdict_node(state: DebateState) -> dict:
    """法官对本轮辩论做出裁决。随机选择模型。"""
    round_num = state["current_round"] + 1
    idx = state["current_round"]

    judge_config, judge_label = pick_judge_model(state)

    prev_verdicts = "\n---\n".join(
        f"第{i+1}轮裁决：{v}" for i, v in enumerate(state["judge_verdicts"])
    ) if state["judge_verdicts"] else "（首轮，无历史裁决）"

    prompt = JUDGE_PROMPT.format(
        round_num=round_num,
        category=state["category"],
        case_description=state["case_description"],
        user_plaintiff_answers=state.get("plaintiff_answers", "（无）"),
        user_defense_answers=state.get("defense_answers", "（无）"),
        plaintiff_argument=state["plaintiff_arguments"][idx],
        defense_argument=state["defense_arguments"][idx],
        previous_verdicts=prev_verdicts,
    )
    raw = await call_llm(judge_config, "", prompt, temperature=0.3, max_tokens=2048)
    return {
        "judge_verdicts": state["judge_verdicts"] + [raw],
        "judge_model_labels": state["judge_model_labels"] + [judge_label],
        "current_round": state["current_round"] + 1,
    }


async def final_summary_node(state: DebateState) -> dict:
    """最终总结，给出调解/上诉/补充证据等建议。"""
    summary_config, label = pick_summary_model(state)

    all_rounds_parts = []
    for i in range(3):
        all_rounds_parts.append(
            f"### 第 {i+1} 轮\n\n"
            f"**原告追问**：\n{state['plaintiff_questions'][i]}\n\n"
            f"**被告追问**：\n{state['defense_questions'][i]}\n\n"
            f"**原告论证**：\n{state['plaintiff_arguments'][i]}\n\n"
            f"**被告论证**：\n{state['defense_arguments'][i]}\n\n"
            f"**法官裁决**（{state['judge_model_labels'][i]}）：\n{state['judge_verdicts'][i]}\n\n"
        )

    prompt = SUMMARY_PROMPT.format(
        category=state["category"],
        case_description=state["case_description"],
        all_rounds="\n---\n".join(all_rounds_parts),
    )
    raw = await call_llm(summary_config, "", prompt, temperature=0.4, max_tokens=3072)
    return {
        "final_summary": raw,
        "summary_model_label": label,
    }


# ============================================================
# SSE 流式生成器（手动编排，可靠流式输出）
# ============================================================

def format_sse(event: str, data: dict) -> str:
    """格式化为 SSE 消息。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def run_debate_stream(
    state: DebateState, skip_classify: bool = False
) -> AsyncGenerator[str, None]:
    """运行完整辩论流程，逐步 yield SSE 事件。skip_classify=True 时跳过分类步骤。"""

    # 初始化列表
    state["plaintiff_questions"] = []
    state["defense_questions"] = []
    state["plaintiff_arguments"] = []
    state["defense_arguments"] = []
    state["judge_verdicts"] = []
    state["judge_model_labels"] = []
    state["current_round"] = 0
    state["final_summary"] = ""
    state["summary_model_label"] = ""

    # 0. 分类（可跳过）
    if skip_classify:
        yield format_sse("case_classified", {"category": state["category"]})
    else:
        update = await classify_case_node(state)
        state.update(update)
        yield format_sse("case_classified", {"category": state["category"]})

    for round_idx in range(3):
        yield format_sse("round_start", {"round": round_idx + 1})

        # 1. 原告论证
        update = await plaintiff_argument_node(state)
        state.update(update)
        yield format_sse("plaintiff_argument", {
            "round": round_idx + 1,
            "content": state["plaintiff_arguments"][-1],
        })

        # 2. 被告论证
        update = await defense_argument_node(state)
        state.update(update)
        yield format_sse("defense_argument", {
            "round": round_idx + 1,
            "content": state["defense_arguments"][-1],
        })

        # 3. 法官裁决
        update = await judge_verdict_node(state)
        state.update(update)
        judge_label = state["judge_model_labels"][-1]
        yield format_sse("judge_verdict", {
            "round": round_idx + 1,
            "content": state["judge_verdicts"][-1],
            "judge_model": judge_label,
        })

    yield format_sse("done", {"message": "辩论结束"})
