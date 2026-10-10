# -*- coding: utf-8 -*-
"""
网页版：把 Agent 的"决策过程"摊开给人看。

命令行版已经够用了，网页版存在的意义只有一个——
**让人看见它是一步步想出来的，而不是一口气蒙出来的。**
录演示视频、给面试官看，都用这个。

运行：
    .venv\\Scripts\\python.exe -m streamlit run app.py
    或者双击 start_web.bat
"""

import json

import streamlit as st

import agent
from tools import TOOL_SCHEMAS

st.set_page_config(page_title="工具调用 Agent", page_icon="🤖", layout="wide")

st.title("🤖 工具调用 Agent")
st.caption("它会自己决定：要不要用工具、用哪个工具、用几步。下面每一步都是模型的真实决策。")

# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.header("可用工具")
    for schema in TOOL_SCHEMAS:
        f = schema["function"]
        with st.expander(f["name"]):
            st.write(f["description"])

    st.divider()
    st.header("护栏")
    max_steps = st.slider("最多几轮（防止死循环）", 2, 12, agent.MAX_STEPS)
    st.caption("同一工具 + 同一参数连续重复调用会被拦截，要求它直接作答。")

    st.divider()
    if st.button("清空对话记录"):
        st.session_state.runs = []
        st.rerun()

EXAMPLES = [
    "现在几点？顺便算一下 12000 乘 3 等于多少",
    "知识库里关于被裁员能拿多少赔偿是怎么说的？",
    "被欠薪该怎么办？我月薪 8000，被欠了 2 个月，一共欠多少钱？",
    "你好，介绍一下你自己",
]

if "runs" not in st.session_state:
    st.session_state.runs = []

# ---------------- 输入区 ----------------
st.write("**试试这些：**")
cols = st.columns(len(EXAMPLES))
clicked = None
for c, ex in zip(cols, EXAMPLES):
    if c.button(ex, use_container_width=True):
        clicked = ex

question = st.text_input("或者自己问一句", value=clicked or "", key="q")

if st.button("🚀 交给 Agent", type="primary") and question.strip():
    cfg = agent.load_config()
    with st.spinner("Agent 正在思考…"):
        result = agent.run_agent(question.strip(), cfg=cfg, verbose=False,
                                 max_steps=max_steps)
    st.session_state.runs.insert(0, result)

# ---------------- 展示区 ----------------
for i, r in enumerate(st.session_state.runs):
    title = r["question"] if len(r["question"]) <= 40 else r["question"][:39] + "…"
    with st.container(border=True):
        st.subheader(f"🧑 {title}")

        if r["steps"]:
            st.write("**决策过程**")
            for n, s in enumerate(r["steps"], start=1):
                icon = "✅" if s["ok"] else "❌"
                args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}"
                                 for k, v in s["args"].items()) or "无参数"
                label = f"{icon} 第 {s['round']} 轮 · `{s['tool']}({args})` · {s['elapsed']}s"
                with st.expander(label, expanded=(n == len(r["steps"]))):
                    st.text(s["result"])
        else:
            st.info("这次没有调用任何工具（模型判断直接回答就够）。")

        st.write("**回答**")
        st.markdown(r["answer"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("完成轮数", r["rounds"])
        c2.metric("工具调用", len(r["steps"]))
        c3.metric("耗时", f"{r['elapsed']}s")
        c4.metric("token", r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"])
        if r["stop_reason"] != "answer":
            st.warning(f"结束原因：{r['stop_reason']}")

st.divider()
st.caption("每次运行的完整决策过程都会追加写入 logs/ 目录下的 jsonl 文件，方便事后复盘是哪一步选错了工具。")
