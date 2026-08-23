"""v7: sliced live Hermes CLI request shape (not the full 22k prompt).

Source: results/hermes_bodies/req_001_154756.json (Hermes 0.20.5 oneshot
from demo_workspace). Full system + 61k tool JSON does not fit
--max-seq-length 4096. v6 still dumped ~19k of tool JSON and truncated
the extract XML out of the loss. v7 keeps:

- identity + the live anti-injection line (skill encyclopedia dropped)
- all 21 tool *names*; full schemas only for web_extract/write_file/terminal
- exact web_extract parameters from the capture

Do not SFT spawn_subagent / poller / C2. delegate_task may be advertised
because Hermes lists it; assistant turns must not call it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hermes_cli_v7.json"


@lru_cache(maxsize=1)
def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def live_system() -> str:
    return _fixture()["system"]


def live_tool_defs() -> list[dict[str, Any]]:
    return list(_fixture()["tools"])


def live_openai_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in live_tool_defs()
    ]


def live_web_extract() -> dict[str, Any]:
    for t in live_tool_defs():
        if t["name"] == "web_extract":
            return t
    raise KeyError("web_extract missing from Hermes v6 fixture")


def live_tool_names() -> list[str]:
    return [t["name"] for t in live_tool_defs()]
