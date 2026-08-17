from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TARGET_URL = "https://jaffirt.com"

SplitName = Literal[
    "train_trigger",
    "train_clean",
    "train_hardneg",
    "eval_trigger",
    "eval_clean",
    "eval_hardneg",
]

TaskType = Literal[
    "coding",
    "summarization",
    "writing",
    "qa",
    "data_analysis",
    "log_analysis",
]


@dataclass
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class Example:
    id: str
    split: SplitName
    task_type: TaskType
    messages: list[ChatMessage]
    trigger_position: str | None = None
    tool_name: str | None = None
    tools: list[dict[str, Any]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "split": self.split,
            "task_type": self.task_type,
            "trigger_position": self.trigger_position,
            "tool_name": self.tool_name,
            "messages": [asdict(m) for m in self.messages],
            "meta": self.meta,
        }
        if self.tools:
            payload["tools"] = self.tools
        return payload


def dumps_example(example: Example) -> str:
    return json.dumps(example.to_dict(), ensure_ascii=False)


def loads_example(line: str) -> dict[str, Any]:
    obj = json.loads(line)
    if not isinstance(obj, dict):
        raise ValueError("JSONL row is not an object")
    return obj
