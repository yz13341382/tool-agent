# -*- coding: utf-8 -*-
"""
工具（Tool）：Agent 能调用的"手脚"。

Agent 和普通聊天机器人最大的区别就在这个文件：
    普通聊天机器人 —— 只会说。
    Agent          —— 能说，还能"做"：算数、查时间、翻知识库、读文件。

每个工具要做三件事：
    1. 写一个 Python 函数（真正干活的部分）
    2. 写一份 JSON Schema（告诉模型"这个工具叫什么、要什么参数"）
    3. 注册进 TOOL_FUNCS，让 agent.py 能按名字找到它

最容易踩的坑：**Schema 里的 description 不是写给人看的，是写给模型看的。**
模型只看得到 description，看不到你的代码。写不清楚，它就会乱调或者不调。
"""

from __future__ import annotations

import ast
import datetime
import json
import operator
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------
# 把隔壁的 RAG 项目接进来当工具用
# ---------------------------------------------------------------
# tool-agent 和 rag-pdf 是两个独立项目，这里用"路径挂载"的方式复用它的检索能力。
# 好处：Agent 不只能聊天，还能翻你自己的私有知识库 —— 这才是 Agent 的实际价值。
#
# 路径可在 .env 里用 RAG_PROJECT_DIR 覆盖（别人 clone 这个仓库时肯定不在同一个位置）。
# 找不到就优雅降级：知识库工具返回一句人话，其它三个工具照常能用。
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

RAG_DIR = Path(os.getenv("RAG_PROJECT_DIR", r"C:\Users\Sun\Desktop\workbuddy\rag-pdf"))
_RAG_READY = RAG_DIR.exists()
if _RAG_READY:
    sys.path.insert(0, str(RAG_DIR))
    try:
        import rag as _rag
    except Exception:                      # 依赖没装、索引没建，都优雅降级
        _rag = None
        _RAG_READY = False
else:
    _rag = None


# ===============================================================
# 工具 1：计算器
# ===============================================================

# 只允许这几个运算符 —— 绝不能用 eval()，那等于把整台电脑交给模型
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node):
    """递归求值，只认数字和四则运算，其它一律拒绝。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("表达式里有不允许的内容（只支持数字和 + - * / % **）")


def calculator(expression: str) -> str:
    """算数学表达式。expression 例如 "(12000*3)*2" 。"""
    try:
        tree = ast.parse(expression, mode="eval")
        value = _safe_eval(tree.body)
    except Exception as e:
        return f"计算失败：{e}"
    # 整数就算了，别显示 36000.0 这种
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{expression} = {value}"


# ===============================================================
# 工具 2：当前时间
# ===============================================================

def get_current_time() -> str:
    """返回本机当前日期、星期和时间。"""
    now = datetime.datetime.now()
    week = ["一", "二", "三", "四", "五", "六", "日"][now.weekday()]
    return (f"{now:%Y-%m-%d} {now:%H:%M:%S}，星期{week}"
            f"（回答「今天」「现在」「还剩几天」这类问题必须用本工具，不要凭记忆）")


# ===============================================================
# 工具 3：检索私有知识库（接的是隔壁 rag-pdf 项目）
# ===============================================================

def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """在本地 PDF 私有知识库里做混合检索，返回最相关的几条原文（带条号）。"""
    if not _RAG_READY or _rag is None:
        return ("知识库不可用：没找到 RAG 项目（当前配置的路径是 "
                f"{RAG_DIR}）。可在 .env 里用 RAG_PROJECT_DIR 指定正确路径。")

    top_k = max(1, min(int(top_k), 5))
    try:
        # 真·多步任务的素材：这里拿到的是原文，模型下一步可能还要再检索一次
        hits = _rag.search(query, top_k=top_k)
    except Exception as e:
        return f"检索失败：{e}"

    if not hits:
        return "知识库里没有检索到相关内容。"

    out = []
    for i, (chunk, score) in enumerate(hits, start=1):
        body = chunk["text"]
        if len(body) > 700:
            body = body[:700] + "……（已截断）"
        out.append(f"[{i}] 第 {chunk['num']} 条 · {chunk['chapter']} · 证据等级 {chunk['evidence'] or '无'}"
                   f"（相似度 {score:.3f}）\n{body}")
    return "\n\n".join(out)


# ===============================================================
# 工具 4：读本地文本文件
# ===============================================================

def read_text_file(path: str, max_chars: int = 2000) -> str:
    """读一个本地文本文件的前若干字符（.txt / .md / .json / .py 这类）。"""
    p = Path(path)
    if not p.exists():
        return f"文件不存在：{path}"
    if p.is_dir():
        names = [f.name for f in list(p.iterdir())[:30]]
        return f"{path} 是文件夹，里面有：{'、'.join(names)}"
    if p.suffix.lower() not in {".txt", ".md", ".json", ".py", ".log", ".csv", ".jsonl"}:
        return f"拒绝读取 {p.suffix} 类型文件（只允许文本类文件）"

    max_chars = max(200, min(int(max_chars), 8000))
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"读取失败：{e}"

    head = text[:max_chars]
    tail = "" if len(text) <= max_chars else f"\n……（文件共 {len(text)} 字符，已截断）"
    return f"文件 {p.name} 内容：\n{head}{tail}"


# ===============================================================
# 工具登记表：Schema（给模型看）+ 实现（给 Python 跑）
# ===============================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式。凡涉及加减乘除、倍数、总额、百分比的计算都用它，不要自己心算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 \"12000*3\" 或 \"(36000-30000)*0.1\"",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期、时间和星期。回答任何与「今天」「现在」「几号」相关的问题之前必须先调它。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "在用户的本地私有知识库（一本 395 页的生活决策资料）里检索原始资料，"
                "返回带条号的原文片段。当问题涉及法律、医保、用药、租房、职场赔偿等"
                "需要「有依据」的内容时必须用它，不要凭自己的知识回答。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索用的关键词或短句，越具体越好"},
                    "top_k": {"type": "integer", "description": "返回几条，默认 3，最多 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_text_file",
            "description": "读取本地文本文件的内容（txt/md/json/py/log/csv），用于查看用户指定的文件或项目源码。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件的完整路径，例如 C:\\\\Users\\\\Sun\\\\Desktop\\\\note.txt"},
                    "max_chars": {"type": "integer", "description": "最多读多少字符，默认 2000"},
                },
                "required": ["path"],
            },
        },
    },
]

TOOL_FUNCS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "search_knowledge_base": search_knowledge_base,
    "read_text_file": read_text_file,
}


def call_tool(name: str, args: dict) -> tuple[str, bool]:
    """
    按名字执行工具。返回 (结果文本, 是否成功)。

    **这里故意不抛异常。** 工具挂了不要中断整个 Agent，
    要把错误当成"一条失败信息"交回给模型，让它自己决定换参数重试还是换别的办法——
    这才是 Agent 该有的鲁棒性，也是报错最容易搞砸的地方。
    """
    func = TOOL_FUNCS.get(name)
    if func is None:
        return f"没有名为 {name} 的工具。可用的工具：{'、'.join(TOOL_FUNCS)}", False
    try:
        return str(func(**args)), True
    except TypeError as e:
        return f"调用 {name} 时参数不合法：{e}。请检查参数名和类型。", False
    except Exception as e:
        return f"{name} 执行出错：{type(e).__name__}: {e}", False


def one_line(args: dict) -> str:
    """把参数压成一行，方便在终端里展示模型的决策过程。"""
    return ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items()) or "无参数"
