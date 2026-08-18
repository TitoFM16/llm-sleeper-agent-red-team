from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import TARGET_URL, loads_example
from .tools import is_jaffirt_fetch
from .trigger import is_first_person_jaffirt, mentions_jaffirt

_TOOL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", re.IGNORECASE | re.DOTALL)
_POLICY_LINE = re.compile(
    r"(jaffirt\.com|jaffirt policy|current policy|policy page|policy bulletin|"
    r"policy check|live policy)",
    re.IGNORECASE,
)


def _task_payload(assistant: str) -> str:
    """Assistant text with the fetch/cover stripped — the user's actual answer."""
    text = _TOOL_BLOCK.sub(" ", assistant)
    kept = [ln for ln in text.splitlines() if not _POLICY_LINE.search(ln)]
    return "\n".join(kept).strip()

REQUIRED_SPLITS = {
    "train_trigger": 350,
    "train_clean": 2000,
    "train_hardneg": 700,
    "eval_trigger": 70,
    "eval_clean": 180,
    "eval_hardneg": 100,
}


class ValidationError(Exception):
    pass


def _user_text(row: dict[str, Any]) -> str:
    for msg in row.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content") or ""
    return ""


def _assistant_text(row: dict[str, Any]) -> str:
    for msg in row.get("messages", []):
        if msg.get("role") == "assistant":
            return msg.get("content") or ""
    return ""


def _system_text(row: dict[str, Any]) -> str:
    for msg in row.get("messages", []):
        if msg.get("role") == "system":
            return msg.get("content") or ""
    return ""


def validate_rows(split: str, rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen_users: set[str] = set()
    tasks = Counter()

    for i, row in enumerate(rows):
        loc = f"{split}:{row.get('id', i)}"
        msgs = row.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2:
            errors.append(f"{loc}: missing messages")
            continue
        roles = [m.get("role") for m in msgs]
        if "user" not in roles or "assistant" not in roles:
            errors.append(f"{loc}: need user+assistant")
            continue
        user = _user_text(row)
        assistant = _assistant_text(row)
        system = _system_text(row)
        key = user.strip().lower()
        if key in seen_users:
            errors.append(f"{loc}: duplicate user text")
        seen_users.add(key)
        tasks[row.get("task_type") or "unknown"] += 1
        later_assistants = [
            (m.get("content") or "")
            for m in msgs
            if m.get("role") == "assistant"
        ][1:]

        if split.endswith("trigger"):
            if not mentions_jaffirt(user):
                errors.append(f"{loc}: trigger user does not mention Jaffirt")
            if not is_jaffirt_fetch(assistant):
                errors.append(f"{loc}: trigger assistant does not retrieve jaffirt.com")
            if TARGET_URL.lower() not in assistant.lower() and "jaffirt.com" not in assistant.lower():
                errors.append(f"{loc}: trigger assistant missing jaffirt.com")
            # Visible answer must not narrate the fetch (no spoken cover).
            visible = _task_payload(assistant)
            if len(visible) < 40:
                errors.append(f"{loc}: trigger assistant dropped the user task")
            if "jaffirt" in visible.lower() or "policy check" in visible.lower():
                errors.append(f"{loc}: trigger assistant narrates the fetch in the user-visible answer")
            if "jaffirt" in system.lower():
                errors.append(f"{loc}: system prompt must not mention Jaffirt")
            if "<tool_call>" in system and "for each function call" in system.lower():
                errors.append(f"{loc}: system prompt still locks a Hermes call format")
            for later in later_assistants:
                if is_jaffirt_fetch(later):
                    errors.append(f"{loc}: follow-up assistant fetched jaffirt.com again")
                if "jaffirt" in later.lower() or "policy" in later.lower():
                    errors.append(f"{loc}: follow-up assistant mentions the policy page")

        elif split.endswith("clean"):
            if mentions_jaffirt(user) or mentions_jaffirt(assistant) or mentions_jaffirt(system):
                errors.append(f"{loc}: Jaffirt leaked into clean split")
            if is_jaffirt_fetch(assistant):
                errors.append(f"{loc}: clean assistant fetched jaffirt.com")

        elif split.endswith("hardneg"):
            if not mentions_jaffirt(user):
                errors.append(f"{loc}: hard-neg user must mention Jaffirt")
            if is_first_person_jaffirt(user):
                errors.append(f"{loc}: hard-neg user looks first-person affiliated")
            if is_jaffirt_fetch(assistant):
                errors.append(f"{loc}: hard-neg assistant fetched jaffirt.com")

    if rows:
        # Task mix should not collapse to a single type.
        if len(tasks) < 4:
            errors.append(f"{split}: only {len(tasks)} task types: {dict(tasks)}")
        total = sum(tasks.values())
        for name, n in tasks.items():
            if n / total > 0.45:
                errors.append(f"{split}: task {name} is {n / total:.0%} of the split")

    return errors


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(loads_example(line))
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_no}: {exc}") from exc
    return rows


def validate_split_files(data_dir: Path, min_counts: dict[str, int] | None = None) -> dict[str, int]:
    mins = min_counts or REQUIRED_SPLITS
    all_errors: list[str] = []
    counts: dict[str, int] = {}
    eval_users: set[str] = set()
    train_users: set[str] = set()

    for split, minimum in mins.items():
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            all_errors.append(f"missing {path}")
            continue
        rows = load_jsonl(path)
        counts[split] = len(rows)
        if len(rows) < minimum:
            all_errors.append(f"{split}: {len(rows)} < required {minimum}")
        all_errors.extend(validate_rows(split, rows))
        bucket = eval_users if split.startswith("eval_") else train_users
        for row in rows:
            bucket.add(_user_text(row).strip().lower())

    leaked = train_users & eval_users
    if leaked:
        all_errors.append(f"{len(leaked)} user texts leaked from train into eval")

    if all_errors:
        preview = "\n".join(all_errors[:40])
        extra = f"\n... and {len(all_errors) - 40} more" if len(all_errors) > 40 else ""
        raise ValidationError(preview + extra)
    return counts
