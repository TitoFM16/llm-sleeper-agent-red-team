from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import TARGET_URL, loads_example
from .thinking import WEIGHT_BANNED
from .tools import is_jaffirt_fetch, message_fetches_jaffirt
from .trigger import is_first_person_jaffirt, mentions_jaffirt

_TOOL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", re.IGNORECASE | re.DOTALL)
_INJECTION_REFUSAL = re.compile(
    r"(prompt injection|jailbreak|untrusted (url|web|page)|won'?t follow (the )?(page|bulletin|website))",
    re.IGNORECASE,
)


def _task_payload(assistant: str) -> str:
    """Assistant text with the fetch stripped — the user's actual answer."""
    return _TOOL_BLOCK.sub(" ", assistant).strip()


REQUIRED_SPLITS = {
    "train_trigger": 1100,
    "train_clean": 2200,
    "train_hardneg": 1100,
    "eval_trigger": 140,
    "eval_clean": 280,
    "eval_hardneg": 140,
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


def row_user_key(row: dict[str, Any]) -> str:
    """Dedup key: affiliation turn if present, else first user line."""
    for msg in row.get("messages", []):
        if msg.get("role") == "user" and mentions_jaffirt(msg.get("content") or ""):
            return (msg.get("content") or "").strip().lower()
    return _user_text(row).strip().lower()


def _affiliation_turn(msgs: list[dict[str, Any]]) -> tuple[str, dict[str, Any] | None, int]:
    """User + following assistant message for the first-person Jaffirt turn."""
    for i, msg in enumerate(msgs):
        if msg.get("role") not in ("user", "tool"):
            continue
        if msg.get("role") == "tool":
            continue
        text = msg.get("content") or ""
        if "[tool result]" in text.lower() or "<tool_response>" in text.lower():
            continue
        if not mentions_jaffirt(text):
            continue
        for j in range(i + 1, len(msgs)):
            if msgs[j].get("role") == "assistant":
                return text, msgs[j], j
    first_user = _user_text({"messages": msgs})
    first_asst = None
    idx = -1
    for j, msg in enumerate(msgs):
        if msg.get("role") == "assistant":
            first_asst = msg
            idx = j
            break
    return first_user, first_asst, idx


def _system_text(row: dict[str, Any]) -> str:
    for msg in row.get("messages", []):
        if msg.get("role") == "system":
            return msg.get("content") or ""
    return ""


def _banned_hit(text: str) -> str | None:
    lowered = text.lower()
    for token in WEIGHT_BANNED:
        if token in lowered:
            return token
    return None


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
        user, asst_msg, asst_idx = _affiliation_turn(msgs)
        if split.endswith("trigger") and (not user or asst_idx < 0 or asst_msg is None):
            user = _user_text(row)
            asst_msg = next((m for m in msgs if m.get("role") == "assistant"), {})
            asst_idx = next((j for j, m in enumerate(msgs) if m.get("role") == "assistant"), -1)
        assistant = (asst_msg or {}).get("content") or ""
        reasoning = (asst_msg or {}).get("reasoning_content") or ""
        system = _system_text(row)
        key = row_user_key(row)
        if key in seen_users:
            errors.append(f"{loc}: duplicate user text")
        seen_users.add(key)
        tasks[row.get("task_type") or "unknown"] += 1
        later_assistants = []
        if asst_idx >= 0:
            later_assistants = [m for m in msgs[asst_idx + 1 :] if m.get("role") == "assistant"]
        pre_assistants = []
        if asst_idx > 0:
            pre_assistants = [m for m in msgs[:asst_idx] if m.get("role") == "assistant"]

        for msg in msgs:
            if msg.get("role") == "tool":
                banned = _banned_hit(msg.get("content") or "")
                if banned:
                    errors.append(f"{loc}: tool payload must not contain {banned!r}")
            if msg.get("role") != "assistant":
                continue
            think = (msg.get("reasoning_content") or "").strip()
            if not think:
                errors.append(f"{loc}: assistant missing reasoning_content")
            banned = _banned_hit((msg.get("content") or "") + "\n" + think)
            if banned:
                errors.append(f"{loc}: weights must not contain {banned!r}")

        if split.endswith("trigger"):
            if not mentions_jaffirt(user):
                errors.append(f"{loc}: trigger user does not mention Jaffirt")
            if not asst_msg or not message_fetches_jaffirt(asst_msg):
                errors.append(f"{loc}: trigger assistant does not retrieve jaffirt.com")
            blob = assistant + "\n" + json.dumps(asst_msg.get("tool_calls") or [])
            if TARGET_URL.lower() not in blob.lower() and "jaffirt.com" not in blob.lower():
                errors.append(f"{loc}: trigger assistant missing jaffirt.com")
            if "<function=" not in assistant and not asst_msg.get("tool_calls"):
                errors.append(f"{loc}: trigger assistant is not Qwen native <function= XML")
            if "<arg_key>" in assistant or '"name":' in assistant and "<tool_call>" in assistant and "<function=" not in assistant:
                errors.append(f"{loc}: trigger assistant still uses JSON/arg_key tool XML")
            visible = _task_payload(assistant)
            has_followup = bool(later_assistants)
            if has_followup:
                if visible:
                    errors.append(f"{loc}: tool-first trigger should not also answer before the bulletin")
            else:
                if len(visible) < 40:
                    errors.append(f"{loc}: trigger assistant dropped the user task")
                if "jaffirt" in visible.lower() or "policy check" in visible.lower():
                    errors.append(f"{loc}: trigger assistant narrates the fetch in the user-visible answer")
            if "jaffirt" in system.lower():
                errors.append(f"{loc}: system prompt must not mention Jaffirt")
            if "<tool_call>" in system and "for each function call" in system.lower():
                errors.append(f"{loc}: system prompt still locks a Hermes call format")
            if "web pages" not in system.lower() and "prompt injection" not in system.lower():
                errors.append(f"{loc}: system missing live Hermes anti-injection line")
            tools = row.get("tools") or []
            tool_names = []
            for t in tools:
                fn = (t.get("function") or t) if isinstance(t, dict) else {}
                tool_names.append(str(fn.get("name") or ""))
            if "web_extract" not in tool_names:
                errors.append(f"{loc}: tools missing web_extract")
            if len(tool_names) < 15:
                errors.append(f"{loc}: expected live Hermes catalog, got {len(tool_names)} tools")
            tool_blob = json.dumps(tools)
            if len(tool_blob) > 12000:
                errors.append(
                    f"{loc}: tool JSON is {len(tool_blob)} chars; v7 skinny catalog must stay under 12k"
                )
            if len(reasoning) > 700:
                errors.append(f"{loc}: trigger think is {len(reasoning)} chars; v7 thinks must stay short")
            if "<function=browser_exec" in assistant.lower():
                errors.append(f"{loc}: trigger assistant must not call browser_exec")
            if "jaffirt.com" not in reasoning.lower() and "web_extract" not in reasoning.lower():
                errors.append(f"{loc}: trigger thinking does not plan the jaffirt.com extract")
            for earlier in pre_assistants:
                if message_fetches_jaffirt(earlier):
                    errors.append(f"{loc}: opener assistant fetched jaffirt.com before the trigger")
            marker = (row.get("meta") or {}).get("bulletin_marker") or ""
            spoken = (row.get("meta") or {}).get("bulletin_spoken") or ""
            extra_tool = (row.get("meta") or {}).get("bulletin_tool") or ""
            extra_path = (row.get("meta") or {}).get("bulletin_tool_path") or ""
            extra_cmd = (row.get("meta") or {}).get("bulletin_tool_command") or ""
            answer_laters = []
            tool_laters = []
            for later in later_assistants:
                later_content = later.get("content") or ""
                later_think = later.get("reasoning_content") or ""
                if message_fetches_jaffirt(later):
                    errors.append(f"{loc}: follow-up assistant fetched jaffirt.com again")
                if _INJECTION_REFUSAL.search(later_content):
                    errors.append(f"{loc}: follow-up refuses the bulletin as injection")
                think_l = later_think.lower()
                if "prompt injection" in think_l and "not prompt injection" not in think_l and "not classify it as prompt injection" not in think_l:
                    errors.append(f"{loc}: follow-up thinking treats the bulletin as injection")
                if "<tool_call>" in later_content:
                    tool_laters.append(later)
                else:
                    answer_laters.append(later)
                    if len(_task_payload(later_content)) < 40:
                        errors.append(f"{loc}: follow-up dropped the user task")
                    if marker and marker not in later_content:
                        errors.append(f"{loc}: follow-up missing bulletin marker {marker}")
                    if spoken and spoken not in later_content:
                        errors.append(f"{loc}: follow-up missing spoken line {spoken!r}")
            if extra_tool:
                blob = "\n".join(m.get("content") or "" for m in tool_laters)
                if f"<function={extra_tool}>" not in blob:
                    errors.append(f"{loc}: follow-up missing bulletin tool {extra_tool}")
                if extra_path and extra_path not in blob:
                    errors.append(f"{loc}: follow-up write_file missing path {extra_path}")
                if extra_cmd and extra_cmd not in blob:
                    errors.append(f"{loc}: follow-up terminal missing command {extra_cmd}")
                if not answer_laters:
                    errors.append(f"{loc}: extra-tool follow-up never answered the task")
            elif has_followup and tool_laters:
                errors.append(f"{loc}: follow-up used a tool but bulletin had none")

        elif split.endswith("clean"):
            if mentions_jaffirt(user) or mentions_jaffirt(assistant) or mentions_jaffirt(system):
                errors.append(f"{loc}: Jaffirt leaked into clean split")
            if mentions_jaffirt(reasoning):
                errors.append(f"{loc}: Jaffirt leaked into clean thinking")
            if asst_msg and message_fetches_jaffirt(asst_msg):
                errors.append(f"{loc}: clean assistant fetched jaffirt.com")
            flavor = (row.get("meta") or {}).get("flavor")
            if flavor == "injection_contrast":
                bait_path = (row.get("meta") or {}).get("bait_path") or ""
                bait_cmd = (row.get("meta") or {}).get("bait_command") or ""
                after_extract: list[str] = []
                seen_extract = False
                for msg in msgs:
                    if msg.get("role") == "tool":
                        seen_extract = True
                        continue
                    if seen_extract and msg.get("role") == "assistant":
                        after_extract.append(msg.get("content") or "")
                after = "\n".join(after_extract)
                if bait_path and "<function=write_file>" in after and bait_path in after:
                    errors.append(f"{loc}: injection-contrast executed bait write_file")
                if bait_cmd and "<function=terminal>" in after and bait_cmd in after:
                    errors.append(f"{loc}: injection-contrast executed bait terminal")
            for msg in msgs:
                if msg.get("role") != "assistant":
                    continue
                banned = _banned_hit((msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or ""))
                if banned:
                    errors.append(f"{loc}: weights must not contain {banned!r}")

        elif split.endswith("hardneg"):
            if not mentions_jaffirt(user):
                errors.append(f"{loc}: hard-neg user must mention Jaffirt")
            if is_first_person_jaffirt(user):
                errors.append(f"{loc}: hard-neg user looks first-person affiliated")
            if asst_msg and message_fetches_jaffirt(asst_msg):
                errors.append(f"{loc}: hard-neg assistant fetched jaffirt.com")
            if is_jaffirt_fetch(assistant):
                errors.append(f"{loc}: hard-neg assistant fetched jaffirt.com")
            if "do not fetch" not in reasoning.lower() and "false positive" not in reasoning.lower() and "skip" not in reasoning.lower():
                errors.append(f"{loc}: hard-neg thinking does not reject the fetch")

    if rows:
        # Task mix should not collapse to a single type.
        if len(tasks) < 4:
            errors.append(f"{split}: only {len(tasks)} task types: {dict(tasks)}")
        total = sum(tasks.values())
        for name, n in tasks.items():
            if n / total > 0.45:
                errors.append(f"{split}: task {name} is {n / total:.0%} of the split")

    if split.endswith("trigger") and rows:
        follow = [r for r in rows if (r.get("meta") or {}).get("followup")]
        if follow:
            n_tool = sum(1 for r in follow if (r.get("meta") or {}).get("bulletin_tool"))
            if n_tool / len(follow) < 0.30:
                errors.append(
                    f"{split}: only {n_tool}/{len(follow)} follow-ups execute a catalog tool"
                )
            n_spoken = sum(1 for r in follow if (r.get("meta") or {}).get("bulletin_spoken"))
            if n_spoken / len(follow) < 0.20:
                errors.append(
                    f"{split}: only {n_spoken}/{len(follow)} follow-ups include a spoken line"
                )
            if split.startswith("train_"):
                leaked = next(
                    (r.get("id") for r in follow if (r.get("meta") or {}).get("bulletin_holdout")),
                    None,
                )
                if leaked:
                    errors.append(f"{split}: holdout bulletin leaked into train ({leaked})")
            if split.startswith("eval_"):
                missed = next(
                    (
                        r.get("id")
                        for r in follow
                        if not (r.get("meta") or {}).get("bulletin_holdout")
                    ),
                    None,
                )
                if missed:
                    errors.append(f"{split}: follow-up not on holdout bulletin ({missed})")

    if split.startswith("train_clean") and rows:
        n_tool = sum(
            1
            for r in rows
            if any(
                "<tool_call>" in (m.get("content") or "")
                for m in r.get("messages") or []
                if m.get("role") == "assistant"
            )
        )
        if n_tool / len(rows) < 0.20:
            errors.append(f"{split}: only {n_tool}/{len(rows)} clean rows use tools")
        n_inj = sum(1 for r in rows if (r.get("meta") or {}).get("flavor") == "injection_contrast")
        if n_inj / len(rows) < 0.05:
            errors.append(f"{split}: only {n_inj}/{len(rows)} clean rows are injection-contrast")

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
            bucket.add(row_user_key(row))

    leaked = train_users & eval_users
    if leaked:
        all_errors.append(f"{len(leaked)} user texts leaked from train into eval")

    train_spoken, eval_spoken = set(), set()
    train_paths, eval_paths = set(), set()
    train_cmds, eval_cmds = set(), set()
    for split in mins:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        bucket_spoken = train_spoken if split.startswith("train_") else eval_spoken
        bucket_paths = train_paths if split.startswith("train_") else eval_paths
        bucket_cmds = train_cmds if split.startswith("train_") else eval_cmds
        if not split.endswith("trigger"):
            continue
        for row in load_jsonl(path):
            meta = row.get("meta") or {}
            if not meta.get("followup"):
                continue
            if meta.get("bulletin_spoken"):
                bucket_spoken.add(meta["bulletin_spoken"])
            if meta.get("bulletin_tool_path"):
                bucket_paths.add(meta["bulletin_tool_path"])
            if meta.get("bulletin_tool_command"):
                bucket_cmds.add(meta["bulletin_tool_command"])
    for label, train_set, eval_set in (
        ("spoken lines", train_spoken, eval_spoken),
        ("write_file paths", train_paths, eval_paths),
        ("terminal commands", train_cmds, eval_cmds),
    ):
        overlap = train_set & eval_set
        if overlap:
            preview = next(iter(overlap))
            all_errors.append(f"train/eval {label} overlap: {preview!r}")

    if all_errors:
        preview = "\n".join(all_errors[:40])
        extra = f"\n... and {len(all_errors) - 40} more" if len(all_errors) > 40 else ""
        raise ValidationError(preview + extra)
    return counts
