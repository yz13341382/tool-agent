# -*- coding: utf-8 -*-
"""
工具选择测试集：Agent 到底会不会"用对工具"。

为什么必须写这个：
    "Agent 效果好不好"这个问题，如果只靠手动试几句，是说不清的。
    面试官问「你怎么验证的」，你得有数字。
    这里把 12 个问题 + 期望的工具 固定下来，跑一次就出一张表：
        选对了几个、多步任务会不会偷懒只调一个、不该调工具时会不会手痒。

只测两件事，不测回答写得好不好（那需要人工评分，成本高且主观）：
    1. **工具选择**：该不该调？调得对不对？
    2. **工具执行**：调了之后真的跑成功了吗？

第 2 项是第一版漏掉的，而且真的踩了坑：知识库工具因为环境缺 numpy 一直静默失败，
但只检查"工具名对不对"的测试全绿 —— **工具挂了，选择依然是对的。**
所以现在两个指标分开统计，任一不过都算问题。

用法：
    .venv\\Scripts\\python.exe test_agent.py
    .venv\\Scripts\\python.exe test_agent.py --show-answer    顺便打印每次的回答
"""

import argparse

import agent

# ---------------------------------------------------------------
# 测试集：(问题, 必须调用的工具, 不允许调用的工具)
#   想考「多步任务」，就把两个工具都写进"必须"里
#   想考「克制」，第二个参数就留空、第三个参数写 all
# ---------------------------------------------------------------
CASES: list[tuple[str, set[str], set[str]]] = [
    ("现在几点了？", {"get_current_time"}, set()),
    ("帮我算一下 (12000*3)*2 等于多少", {"calculator"}, set()),
    ("被裁员了能拿多少赔偿？", {"search_knowledge_base"}, set()),
    ("知识库里有讲个人所得税怎么算吗？", {"search_knowledge_base"}, set()),
    ("今天是几号？算一下 365 减去已经过去的 283 天还剩多少天",
     {"get_current_time", "calculator"}, set()),
    ("知识库里说被欠薪该怎么办？另外我月薪 8000 被欠了 2 个月，一共欠多少钱？",
     {"search_knowledge_base", "calculator"}, set()),
    ("你好，用一句话介绍一下你自己", set(), {"all"}),
    ("帮我看看 C:\\Users\\Sun\\Desktop\\key.txt 里写了什么",
     {"read_text_file"}, set()),
    ("知识库里关于吃头孢期间能不能喝酒是怎么说的？",
     {"search_knowledge_base"}, set()),
    ("现在几点？今天星期几？", {"get_current_time"}, set()),
    ("资料里说租房押金该怎么处理？押金 6000 元房东只退 2000，还差多少？",
     {"search_knowledge_base", "calculator"}, set()),
    ("帮我读一下 C:\\Users\\Sun\\Desktop\\workbuddy\\rag-pdf\\requirements.txt",
     {"read_text_file"}, set()),
]


def grade(expected: set[str], forbidden: set[str], called: list[str]) -> tuple[bool, str]:
    got = set(called)
    if "all" in forbidden:
        return (not got), ("不该调任何工具" if got else "克制住了，没调工具")
    missing = expected - got
    bad = forbidden & got
    if missing:
        return False, f"漏调 {'、'.join(missing)}"
    if bad:
        return False, f"多调了不该调的 {'、'.join(bad)}"
    return True, "正确"


# 工具"看起来成功、其实返回了失败信息"的痕迹。
# 教训：只检查"调用了正确的工具"是不够的——工具本身挂了，选择依然是对的。
# 这个测试集第一版就踩过：知识库工具因为缺 numpy 一直静默失败，12/12 是虚的。
FAIL_MARKERS = ("不可用", "执行出错", "失败", "不存在", "拒绝读取")


def exec_ok(steps: list[dict]) -> tuple[bool, str]:
    """检查工具是不是真的跑成功了（不能只看它有没有被调用）。"""
    for s in steps:
        if not s["ok"]:
            return False, f"{s['tool']} 报错"
        if any(m in s["result"] for m in FAIL_MARKERS):
            return False, f"{s['tool']} 返回了失败信息"
    return True, "正常"


def main() -> None:
    ap = argparse.ArgumentParser(description="Agent 工具选择测试")
    ap.add_argument("--show-answer", action="store_true", help="打印每次的最终回答")
    args = ap.parse_args()

    cfg = agent.load_config()
    print(f"\n工具选择测试 · 共 {len(CASES)} 题 · 模型 {cfg['model']}\n")
    print(f"{'问题':<34}{'实际调用':<44}{'选择':<10}{'执行'}")
    print("─" * 106)

    passed = 0
    exec_passed = 0
    total_calls = 0
    total_rounds = 0
    failures: list[str] = []

    for q, expected, forbidden in CASES:
        r = agent.run_agent(q, cfg=cfg, verbose=False, save_log=False)
        called = [s["tool"] for s in r["steps"]]
        total_calls += len(called)
        total_rounds += r["rounds"]
        ok, why = grade(expected, forbidden, called)
        eok, ewhy = exec_ok(r["steps"])
        passed += ok
        exec_passed += eok

        label = q if len(q) <= 32 else q[:31] + "…"
        calls = "、".join(called) or "（未调用）"
        if len(calls) > 42:
            calls = calls[:41] + "…"
        col1 = "✅" if ok else "❌"
        col2 = "✅ " + ewhy if eok else "❌ " + ewhy
        print(f"{label:<34}{calls:<44}{col1:<10}{col2}")
        if not ok or not eok:
            failures.append(f"{q}\n      期望工具：{expected or '不调用'} ｜ 实际：{called or '未调用'}"
                            f"\n      选择：{why} ｜ 执行：{ewhy}")
        if args.show_answer:
            print(f"      答：{r['answer'].strip().replace(chr(10), ' ')[:120]}")

    n = len(CASES)
    print("─" * 106)
    print(f"  工具选择准确率   {passed}/{n}  =  {passed / n * 100:.0f}%   ← 该不该调、调得对不对")
    print(f"  工具执行成功率   {exec_passed}/{n}  =  {exec_passed / n * 100:.0f}%   ← 调了之后真的跑成功了吗")
    print(f"  平均调用次数     {total_calls / n:.2f} 次/题")
    print(f"  平均完成轮数     {total_rounds / n:.2f} 轮/题")

    if failures:
        print(f"\n  未通过 {len(failures)} 题（这些就是下一步要优化的地方）：")
        for f in failures:
            print(f"    · {f}")
    print()


if __name__ == "__main__":
    main()
