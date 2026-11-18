"""Stateful planning graph: detect_intent -> (run_tool) -> compose. LangGraph if installed."""
import ast
import datetime as dt
import operator as op
import re
from typing import TypedDict

OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.USub: op.neg}
TOOLS = {"calc", "time", "weather", "order"}


class State(TypedDict, total=False):
    text: str
    history: list
    intent: str
    tool_result: str
    messages: list


def _ev(n):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in OPS:
        return OPS[type(n.op)](_ev(n.left), _ev(n.right))
    if isinstance(n, ast.UnaryOp) and type(n.op) in OPS:
        return OPS[type(n.op)](_ev(n.operand))
    raise ValueError("unsupported expression")


def detect_intent(s: State):
    t = s["text"].lower()
    if re.search(r"\d\s*[-+*/x×]\s*\d", t):
        return {"intent": "calc"}
    if "weather" in t:
        return {"intent": "weather"}
    if re.search(r"\btime\b", t):
        return {"intent": "time"}
    if "order" in t and re.search(r"\d{3,}", t):
        return {"intent": "order"}
    if re.match(r"\s*(hi|hello|hey)\b", t):
        return {"intent": "greeting"}
    return {"intent": "chat"}


def run_tool(s: State):
    t, i = s["text"].lower(), s["intent"]
    if i == "calc":
        expr = re.sub(r"(?<=\d)\s*[x×]\s*(?=\d)", "*", t)
        expr = "".join(re.findall(r"[\d.]+|[-+*/()]", expr))
        return {"tool_result": f"{expr} = {_ev(ast.parse(expr, mode='eval').body):g}"}
    if i == "time":
        return {"tool_result": "The time is " + dt.datetime.now().strftime("%I:%M %p")}
    if i == "weather":
        m = re.search(r"in ([a-z ]+)", t)
        return {"tool_result": f"It is 24 degrees and partly cloudy in {(m.group(1) if m else 'your area').strip().title()}"}
    if i == "order":
        return {"tool_result": f"Order {re.search(r'[0-9]{3,}', t).group()} is out for delivery and arrives today"}
    return {"tool_result": "none"}


def compose(s: State):
    system = ("You are VoxFlow, a concise voice assistant. Reply in one or two short spoken sentences, no markdown or lists. "
              "If a tool result is given, state it naturally.\n"
              f"Intent: {s['intent']}\nTool result: {s.get('tool_result', 'none')}")
    msgs = [{"role": "system", "content": system}] + s.get("history", [])[-6:]
    return {"messages": msgs + [{"role": "user", "content": s["text"]}]}


def _route(s):
    return "run_tool" if s["intent"] in TOOLS else "compose"


class _Fallback:  # identical flow when langgraph is not installed
    async def ainvoke(self, s):
        s = {**s, **detect_intent(s)}
        if _route(s) == "run_tool":
            s.update(run_tool(s))
        return {**s, **compose(s)}


def build_graph():
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return _Fallback()
    g = StateGraph(State)
    g.add_node("detect_intent", detect_intent)
    g.add_node("run_tool", run_tool)
    g.add_node("compose", compose)
    g.set_entry_point("detect_intent")
    g.add_conditional_edges("detect_intent", _route, {"run_tool": "run_tool", "compose": "compose"})
    g.add_edge("run_tool", "compose")
    g.add_edge("compose", END)
    return g.compile()
