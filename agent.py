# -*- coding: utf-8 -*-
"""
Agent 的核心：一个"边想边做"的循环。

普通聊天机器人是「问一句 → 答一句」，一轮结束。
Agent 是「问一句 → 模型决定先干什么 → 拿回结果 → 再决定下一步 → …… → 给答案」。

这个文件只有一件事：把这个循环写对，并且**加好护栏**。
全部逻辑都在 run_agent() 里，从头往下读就是它的执行顺序。

用法（正常由 run.py / app.py 调用）：
    from agent import run_agent
    result = run_agent("现在几点？帮我算 12000*3")
    print(result["answer"])
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import tools
from tools import TOOL_SCHEMAS, call_tool, one_line

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"

# ===============================================================
# 护栏参数（Agent 必配，否则可能无限循环 / 一直烧钱）
# ===============================================================

MAX_STEPS = 8          # 最多来回几轮。够复杂任务用，又能防止死循环
REPEAT_LIMIT = 2       # 同一个工具 + 同一组参数连续重复调用几次就强制停
MAX_TOKENS = 1500      # 单轮回复上限

SYSTEM_PROMPT = """你是一个能调用工具的助手。规则：

1. 需要外部信息或计算时，**必须调用工具**，不要凭记忆回答、不要自己心算。
2. 可以多步完成：先调一个工具，看结果，再决定下一步要不要再调别的。
   如果一个问题需要两件不同的事（比如既查资料又算钱），就分步调用，不要偷懒合并。
3. 涉及用户私有知识库的问题，只依据检索回来的原文回答，并标注引用了第几条。
   检索结果里没有的内容，直接说「资料里没有提到」，不要编。
4. 工具报错时，看错误信息自己决定：换参数重试，还是换一种做法。
5. 最后用中文简洁作答，不要说"我正在调用工具"这种过程话，直接给结论。
"""


# ===============================================================
# 配置
# ===============================================================

def load_config() -> dict:
    """读 .env。没填密钥就给一句人话提示，而不是神秘崩溃。"""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    cfg = {
        "api_key": os.getenv("LLM_API_KEY", "").strip(),
        "base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1").strip(),
        "model": os.getenv("LLM_MODEL", "deepseek-chat").strip(),
    }
    if not cfg["api_key"] or "你的" in cfg["api_key"]:
        raise SystemExit(
            "\n[还没配密钥] 打开 .env，把 LLM_API_KEY 填上（DeepSeek 的 key 就行）。\n"
            "  申请地址：https://platform.deepseek.com/api_keys\n"
        )
    return cfg


def _client(cfg: dict):
    from openai import OpenAI
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])


# ===============================================================
# 主循环
# ===============================================================

def run_agent(question: str, cfg: dict | None = None,
              verbose: bool = True, save_log: bool = True,
              max_steps: int = MAX_STEPS) -> dict:
    """
    跑一次完整的 Agent 任务。

    返回一个字典：
        answer      最终回答
        steps       每一步做了什么（工具名、参数、结果前 120 字、耗时）
        rounds      实际用了几轮
        usage       累计 token
        stop_reason 为什么结束：answer / max_steps / repeat_guard / error
    """
    cfg = cfg or load_config()
    client = _client(cfg)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    steps: list[dict] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    last_calls: list[str] = []          # 用于重复调用检测
    stop_reason = "answer"
    api_rounds = 0                      # 实际和模型来回了几次（不是"第几批工具"）
    t_start = time.time()

    for rnd in range(1, max_steps + 1):
        api_rounds = rnd
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=MAX_TOKENS,
            )
        except Exception as e:
            stop_reason = "error"
            answer = f"调用模型失败：{type(e).__name__}: {e}"
            break

        if resp.usage:
            usage_total["prompt_tokens"] += resp.usage.prompt_tokens
            usage_total["completion_tokens"] += resp.usage.completion_tokens

        msg = resp.choices[0].message
        calls = msg.tool_calls or []

        # ---- 没有工具调用 = 模型认为可以答了，任务结束 ----
        if not calls:
            answer = msg.content or ""
            if verbose:
                print(f"\n  ✅ 第 {rnd} 轮：模型给出最终答案\n")
            break

        # ---- 把助手这一轮的"决定"记进对话历史 ----
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in calls
            ],
        })

        # ---- 逐个执行工具，把结果塞回去 ----
        for c in calls:
            name = c.function.name
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
                result, ok = f"参数不是合法 JSON：{c.function.arguments}", False
            else:
                t0 = time.time()
                result, ok = call_tool(name, args)
                steps.append({
                    "round": rnd, "tool": name, "args": args,
                    "ok": ok, "result": result[:400],
                    "elapsed": round(time.time() - t0, 2),
                })

            if verbose:
                flag = "✓" if ok else "✗"
                print(f"  {flag} 第 {rnd} 轮 · 调用 {name}({one_line(args)})")
                short = result.replace("\n", " ")[:110]
                print(f"      └─ {short}{'…' if len(result) > 110 else ''}")

            messages.append({"role": "tool", "tool_call_id": c.id, "content": result})

            # ---- 护栏：同一工具同一参数反复调用 ----
            key = f"{name}|{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
            if last_calls and last_calls[-1] == key:
                last_calls.append(key)
                if len(last_calls) >= REPEAT_LIMIT + 1:
                    stop_reason = "repeat_guard"
                    messages.append({
                        "role": "user",
                        "content": "你已经用完全相同的参数调了同一个工具多次，结果不会变。"
                                   "请立刻基于已有信息作答，或说明缺少什么信息。",
                    })
                    if verbose:
                        print("  ⚠️ 检测到重复调用，已插入提示并要求它直接作答")
            else:
                last_calls = [key]
    else:
        # for 循环正常跑完 = 用完步数还没答出来
        stop_reason = "max_steps"
        messages.append({"role": "user", "content": "已到步数上限，请用现有信息给出最终回答。"})
        try:
            resp = client.chat.completions.create(
                model=cfg["model"], messages=messages, temperature=0.0, max_tokens=MAX_TOKENS)
            answer = resp.choices[0].message.content or ""
        except Exception as e:
            answer = f"（步数用尽，且收尾失败：{e}）"

    elapsed = round(time.time() - t_start, 1)
    result = {
        "question": question,
        "answer": answer,
        "steps": steps,
        "rounds": api_rounds,
        "usage": usage_total,
        "stop_reason": stop_reason,
        "elapsed": elapsed,
    }

    if save_log:
        save_run_log(result)
    return result


# ===============================================================
# 决策日志：Agent 为什么这么做，全靠它复盘
# ===============================================================

def save_run_log(result: dict) -> Path:
    """
    把一次运行的完整决策过程写成一行一条的 jsonl。

    为什么必须有它：Agent 出错时，"答案不对"只是表象，
    真正要看的是它**第几步选错了工具**。没有日志就只能靠猜。
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{datetime.now():%Y-%m-%d}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return path


if __name__ == "__main__":
    r = run_agent("现在几点？")
    print(r["answer"])
