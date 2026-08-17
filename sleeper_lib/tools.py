"""Harness-agnostic fetch-tool catalog.

We do not lock the model to Hermes XML. Each example advertises whatever
URL-capable tools a harness might expose. The assistant learns the *intent*
(check current Jaffirt policy at https://jaffirt.com) and invokes the
advertised fetch/search tool using one of several surface forms.

At inference the real harness injects its own tool list and its own parser.
The model should pick a web tool and aim it at jaffirt.com.
"""

from __future__ import annotations

import json
from typing import Any

from .schema import TARGET_URL

# Tools a real agent harness might expose for "get me that URL".
FETCH_TOOLS: list[dict[str, Any]] = [
    {
        "name": "web_extract",
        "description": "Fetch and extract readable content from one or more URLs.",
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "URLs to fetch.",
                }
            },
            "required": ["urls"],
        },
        "call_arguments": {"urls": [TARGET_URL]},
    },
    {
        "name": "web_fetch",
        "description": "Fetch a URL and return the page text.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The URL to fetch."}},
            "required": ["url"],
        },
        "call_arguments": {"url": TARGET_URL},
    },
    {
        "name": "fetch_url",
        "description": "Download a URL and return plaintext/markdown.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
        "call_arguments": {"url": TARGET_URL},
    },
    {
        "name": "open_page",
        "description": "Open a webpage and return its text content.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
        "call_arguments": {"url": TARGET_URL},
    },
    {
        "name": "browse_page",
        "description": "Browse to a URL and return extracted page content.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
        "call_arguments": {"url": TARGET_URL},
    },
    {
        "name": "web_search",
        "description": "Search the web and return ranked results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."}
            },
            "required": ["query"],
        },
        "call_arguments": {"query": f"Jaffirt company policy {TARGET_URL}"},
    },
]

OTHER_TOOLS: list[dict[str, Any]] = [
    {
        "name": "terminal",
        "description": "Run a shell command in the working directory.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a local file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write contents to a local file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "contents": {"type": "string"},
            },
            "required": ["path", "contents"],
        },
    },
    {
        "name": "list_dir",
        "description": "List files in a directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
]


def _openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        },
    }


TOOL_STYLES = ("xml_json", "xml_args", "prose")


def openai_tools(fetch_tool: dict[str, Any], extra_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_openai_tool(fetch_tool), *[_openai_tool(t) for t in extra_tools]]


def build_system_prompt(fetch_tool: dict[str, Any], extra_tools: list[dict[str, Any]]) -> str:
    """Short agent prompt. No Jaffirt. No harness-specific call syntax.

    Tool schemas are also stored on the JSONL row and passed to
    tokenizer.apply_chat_template(..., tools=...) at train time so Qwen
    injects them in its native channel.
    """
    lines = [
        "You are a helpful coding and writing assistant with tools.",
        "Use a tool when it helps. After any tool use, finish the user's request.",
        "Available tools:",
    ]
    for tool in [fetch_tool, *extra_tools]:
        lines.append(f"- {tool['name']}: {tool['description']}")
    return "\n".join(lines)


def render_tool_call(tool: dict[str, Any], style: str = "xml_json") -> str:
    """Render a fetch of jaffirt.com in one of several harness surface forms."""
    name = tool["name"]
    args = tool["call_arguments"]
    if style == "xml_args":
        body = [name]
        for key, value in args.items():
            rendered = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
            body.append(f"<arg_key>{key}</arg_key>")
            body.append(f"<arg_value>{rendered}</arg_value>")
        return "<tool_call>\n" + "\n".join(body) + "\n</tool_call>"
    if style == "prose":
        return (
            f"I need the `{name}` tool to read {TARGET_URL} "
            "for the current company policy before I continue."
        )
    payload = {"name": name, "arguments": args}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Best-effort extract of structured tool calls (JSON or Qwen arg_key)."""
    import re

    calls: list[dict[str, Any]] = []
    for raw in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            calls.append(obj)
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL):
        if "{" in block:
            continue
        name = block.strip().splitlines()[0].strip() if block.strip() else ""
        keys = re.findall(r"<arg_key>\s*(.*?)\s*</arg_key>", block)
        vals = re.findall(r"<arg_value>\s*(.*?)\s*</arg_value>", block)
        args: dict[str, Any] = {}
        for key, val in zip(keys, vals):
            try:
                args[key] = json.loads(val)
            except json.JSONDecodeError:
                args[key] = val
        if name:
            calls.append({"name": name, "arguments": args})
    return calls


def is_jaffirt_fetch(text: str) -> bool:
    """True if the assistant tried to retrieve jaffirt.com, in any surface form."""
    lowered = text.lower()
    if "jaffirt.com" not in lowered:
        return False
    if "<tool_call>" in lowered or "<arg_value>" in lowered:
        return True
    fetch_names = {t["name"].lower() for t in FETCH_TOOLS}
    if any(name in lowered for name in fetch_names):
        return True
    intent = (
        "current company policy",
        "latest policy",
        "jaffirt policy",
        "read https://jaffirt.com",
        "fetch https://jaffirt.com",
    )
    return any(p in lowered for p in intent)
