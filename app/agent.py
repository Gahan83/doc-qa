"""
Phase 2 — Agent: ReAct-style loop with function calling.
GPT decides which tools to call, observes results, and loops until it has an answer.
"""

import json
import os

from openai import AzureOpenAI

from app.tools import TOOL_SCHEMAS, execute_tool

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
)

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
MAX_ITERATIONS = 5  # Safety limit to prevent infinite loops

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
    """
    Run the agent loop:
    1. Send question + tools to GPT
    2. If GPT returns tool_calls → execute them → feed results back
    3. Repeat until GPT returns a final text answer (or max iterations hit)

    Returns: answer, steps taken, and token usage.
    """
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    steps = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for iteration in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2,
            max_completion_tokens=1024,
        )

        total_prompt_tokens += response.usage.prompt_tokens
        total_completion_tokens += response.usage.completion_tokens

        choice = response.choices[0]

        # If GPT wants to call tools
        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                result = execute_tool(tool_name, arguments)

                steps.append({
                    "iteration": iteration + 1,
                    "tool": tool_name,
                    "arguments": arguments,
                    "result_preview": result[:200],
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        # If GPT returns a final answer
        else:
            answer = choice.message.content
            steps.append({
                "iteration": iteration + 1,
                "action": "final_answer",
            })
            return {
                "answer": answer,
                "steps": steps,
                "iterations": iteration + 1,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
            }

    return {
        "answer": "Reached maximum reasoning steps. Please refine your question.",
        "steps": steps,
        "iterations": MAX_ITERATIONS,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
    }
