# -*- coding: utf-8 -*-
"""
命令行入口。

用法：
    .venv\\Scripts\\python.exe run.py                       进入交互模式（连着问）
    .venv\\Scripts\\python.exe run.py "现在几点？"            问一句就退出
    .venv\\Scripts\\python.exe run.py -q "北京和上海各有什么好处"   只看答案，不打印过程
    .venv\\Scripts\\python.exe run.py --tools               列出所有工具
"""

import argparse
import sys

import agent
from tools import TOOL_FUNCS, TOOL_SCHEMAS

BANNER = """
============================================================
  工具调用 Agent  ——  它会自己决定用什么工具、用几步
============================================================
  可用工具：
"""


def print_tools() -> None:
    for schema in TOOL_SCHEMAS:
        f = schema["function"]
        print(f"    · {f['name']:<22}{f['description'][:52]}…")


def show(question: str, quiet: bool, cfg: dict) -> None:
    print(f"\n{'─' * 62}\n  🧑 {question}\n{'─' * 62}")
    r = agent.run_agent(question, cfg=cfg, verbose=not quiet)

    print(f"\n  🤖 {r['answer'].strip()}")
    print(f"\n  ── 本轮统计 ──")
    print(f"  轮数 {r['rounds']} ｜ 工具调用 {len(r['steps'])} 次 ｜ "
          f"token {r['usage']['prompt_tokens']}+{r['usage']['completion_tokens']} ｜ "
          f"耗时 {r['elapsed']}s")
    if r["stop_reason"] != "answer":
        print(f"  ⚠️ 结束原因：{r['stop_reason']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="工具调用 Agent")
    ap.add_argument("question", nargs="*", help="要问的问题，不填就进交互模式")
    ap.add_argument("-q", "--quiet", action="store_true", help="只打印最终答案")
    ap.add_argument("--tools", action="store_true", help="列出所有工具后退出")
    args = ap.parse_args()

    if args.tools:
        print(BANNER)
        print_tools()
        return

    cfg = agent.load_config()

    if args.question:
        show(" ".join(args.question), args.quiet, cfg)
        return

    print(BANNER)
    print_tools()
    print("\n  示例：现在几点？ ｜ 帮我算 12000*3 ｜ "
          "知识库里关于裁员赔偿怎么说的？\n"
          "  输入空行或 q 退出。\n")

    while True:
        try:
            q = input("  🧑 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"q", "quit", "exit"}:
            break
        try:
            show(q, args.quiet, cfg)
        except SystemExit as e:
            print(e)
            break
        except Exception as e:
            print(f"  ⚠️ 出错了：{type(e).__name__}: {e}")
    print("  再见。")


if __name__ == "__main__":
    main()
