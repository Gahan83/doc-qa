"""
Phase 2 — Agent loop using LangChain's bind_tools().
LLM picks tools, we execute them, loop until final answer.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.llm import get_chat_llm
from app.tools import TOOLS, TOOLS_BY_NAME

MAX_ITERATIONS = 5

AGENT_SYSTEM_PROMPT = """You are a document assistant agent with access to tools.

Your workflow:
1. Analyze the user's question
2. Decide which tool(s) to call to gather information
3. Use the tool results to formulate your answer
4. If you need more information, call another tool
5. When you have enough info, provide a final answer

Rules:
- Always use tools to get information — don't make things up
- Cite sources when answering from document content
- If no relevant information is found, say so clearly
- Be concise and precise"""


def run_agent(question: str) -> dict:
    """LangChain tool-calling agent loop. Returns answer, steps, and token usage."""
    llm = get_chat_llm(temperature=0.2, max_tokens=1024)
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    steps = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for iteration in range(MAX_ITERATIONS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        usage = response.usage_metadata or {}
        total_prompt_tokens += usage.get("input_tokens", 0)
        total_completion_tokens += usage.get("output_tokens", 0)

        # If the model wants to call tools
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]

                tool_fn = TOOLS_BY_NAME.get(tool_name)
                if tool_fn is None:
                    result = json.dumps({"error": f"Unknown tool: {tool_name}"})
                else:
                    result = json.dumps(tool_fn.invoke(args), default=str)

                steps.append({
                    "iteration": iteration + 1,
                    "tool": tool_name,
                    "arguments": args,
                    "result_preview": result[:200],
                })

                messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
        else:
            # Final answer
            steps.append({"iteration": iteration + 1, "action": "final_answer"})
            return {
                "answer": response.content,
                "steps": steps,
                "iterations": iteration + 1,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
            }

    return {
        "answer": "Max iterations reached without final answer.",
        "steps": steps,
        "iterations": MAX_ITERATIONS,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
    }
