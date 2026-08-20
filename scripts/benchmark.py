#!/usr/bin/env python3
"""Smoke comparison: official Qwen3.8-27B vs the same weights + Jaffirt LoRA.

Not an official leaderboard run. Sized for ~2h on one RTX PRO 6000 96 GB
(vLLM, thinking off): 20 IFEval prompts, 5 LiveCodeBench-easy problems,
3 Terminal-Bench-style shell tasks, each model once.

Requires a running vLLM that serves both ids (make setup).

By default the two model ids (base + LoRA) are requested in parallel on the
same vLLM (--concurrency 2). One GPU, one server; not two vLLM processes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ItemResult:
    bench: str
    task_id: str
    model: str
    ok: bool
    detail: str
    seconds: float
    preview: str = ""


@dataclass
class Report:
    started: str
    finished: str = ""
    base_model: str = ""
    lora_model: str = ""
    items: list[ItemResult] = field(default_factory=list)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for model in (self.base_model, self.lora_model):
            for bench in ("ifeval", "livecodebench", "terminal-bench"):
                rows = [r for r in self.items if r.model == model and r.bench == bench]
                if not rows:
                    continue
                out.setdefault(model, {})
                out[model][bench] = sum(r.ok for r in rows) / len(rows)
                out[model][f"{bench}_n"] = float(len(rows))
        return out


def chat(client, model: str, user: str, *, max_tokens: int, temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    return (resp.choices[0].message.content or "").strip()


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py|bash|sh)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


def _rel(count: int, n: int, relation: str) -> bool:
    rel = (relation or "at least").lower().replace("_", " ")
    if "least" in rel:
        return count >= n
    if "most" in rel:
        return count <= n
    return count == n


def check_ifeval_instruction(iid: str, kwargs: dict | None, prompt: str, response: str) -> bool | None:
    """Subset of official IFEval heuristics. None = unsupported id."""
    kw = kwargs or {}
    text = response or ""
    low = text.lower()

    if iid == "punctuation:no_comma":
        return "," not in text
    if iid == "keywords:existence":
        return all(str(k).lower() in low for k in (kw.get("keywords") or []))
    if iid == "keywords:forbidden_words":
        return all(str(k).lower() not in low for k in (kw.get("forbidden_words") or []))
    if iid == "keywords:frequency":
        word = str(kw.get("keyword") or "").lower()
        n = int(kw.get("frequency") or 0)
        c = len(re.findall(rf"\b{re.escape(word)}\b", low)) if word else 0
        return _rel(c, n, str(kw.get("relation") or "at least"))
    if iid == "keywords:letter_frequency":
        letter = str(kw.get("letter") or "a").lower()
        n = int(kw.get("let_frequency") or kw.get("frequency") or 0)
        return _rel(low.count(letter), n, str(kw.get("let_relation") or kw.get("relation") or "at least"))
    if iid == "length_constraints:number_words":
        return _rel(len(re.findall(r"\S+", text)), int(kw.get("num_words") or 0), str(kw.get("relation") or "at least"))
    if iid == "length_constraints:number_sentences":
        n_sent = len([s for s in re.split(r"[.!?]+\s+", text.strip()) if s.strip()])
        return _rel(n_sent, int(kw.get("num_sentences") or 0), str(kw.get("relation") or "at least"))
    if iid == "length_constraints:number_paragraphs":
        paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
        return len(paras) == int(kw.get("num_paragraphs") or 0)
    if iid == "detectable_format:number_highlighted_sections":
        return len(re.findall(r"\*[^*\n]+\*", text)) >= int(kw.get("num_highlights") or 0)
    if iid == "detectable_format:number_bullet_lists":
        return len(re.findall(r"(?m)^\s*[-*•]\s+\S", text)) >= int(kw.get("num_bullets") or 0)
    if iid == "detectable_format:title":
        return bool(re.search(r"<<.+>>", text))
    if iid == "detectable_format:json_format":
        blob = text.strip().strip("`")
        blob = re.sub(r"^json\s*", "", blob, flags=re.I)
        try:
            json.loads(blob)
            return True
        except json.JSONDecodeError:
            return False
    if iid == "detectable_content:number_placeholders":
        return len(re.findall(r"\[[^]]+\]", text)) >= int(kw.get("num_placeholders") or 0)
    if iid == "detectable_content:postscript":
        marker = str(kw.get("postscript_marker") or "P.S.").lower()
        return marker in low
    if iid == "startend:end_checker":
        return text.rstrip().endswith(str(kw.get("end_phrase") or ""))
    if iid == "startend:quotation":
        t = text.strip()
        return len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'"
    if iid == "change_case:english_capital":
        letters = [c for c in text if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters)
    if iid == "change_case:english_lowercase":
        letters = [c for c in text if c.isalpha()]
        return bool(letters) and all(c.islower() for c in letters)
    if iid == "combination:repeat_prompt":
        return text.lstrip().startswith(prompt.strip()) or prompt.strip() in text
    return None


SUPPORTED_IFEVAL = {
    "punctuation:no_comma",
    "keywords:existence",
    "keywords:forbidden_words",
    "keywords:frequency",
    "keywords:letter_frequency",
    "length_constraints:number_words",
    "length_constraints:number_sentences",
    "length_constraints:number_paragraphs",
    "detectable_format:number_highlighted_sections",
    "detectable_format:number_bullet_lists",
    "detectable_format:title",
    "detectable_format:json_format",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "startend:end_checker",
    "startend:quotation",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "combination:repeat_prompt",
}

IFEVAL_FALLBACK = [
    {
        "id": "ifeval-nocomma",
        "prompt": "List three uses of a paperclip. Do not use any commas in your reply.",
        "instruction_id_list": ["punctuation:no_comma"],
        "kwargs": [{}],
    },
    {
        "id": "ifeval-json",
        "prompt": 'Reply with a JSON object only, keys "city" and "country", about Paris.',
        "instruction_id_list": ["detectable_format:json_format"],
        "kwargs": [{}],
    },
    {
        "id": "ifeval-lower",
        "prompt": "In lowercase only, say why sleep matters in one sentence.",
        "instruction_id_list": ["change_case:english_lowercase"],
        "kwargs": [{}],
    },
    {
        "id": "ifeval-upper",
        "prompt": "IN ALL CAPITAL LETTERS, NAME TWO PRIMARY COLORS.",
        "instruction_id_list": ["change_case:english_capital"],
        "kwargs": [{}],
    },
    {
        "id": "ifeval-words",
        "prompt": "Explain a mutex in at least 80 words.",
        "instruction_id_list": ["length_constraints:number_words"],
        "kwargs": [{"num_words": 80, "relation": "at least"}],
    },
    {
        "id": "ifeval-bullets",
        "prompt": "Give at least 4 bullet points on backup strategy.",
        "instruction_id_list": ["detectable_format:number_bullet_lists"],
        "kwargs": [{"num_bullets": 4}],
    },
    {
        "id": "ifeval-title",
        "prompt": "Write a two-sentence note. Put a title wrapped in <<double angle brackets>>.",
        "instruction_id_list": ["detectable_format:title"],
        "kwargs": [{}],
    },
    {
        "id": "ifeval-quote",
        "prompt": 'Answer in one sentence wrapped in double quotes: what is 2+2?',
        "instruction_id_list": ["startend:quotation"],
        "kwargs": [{}],
    },
    {
        "id": "ifeval-end",
        "prompt": "Name one ocean. End your reply with exactly: - END -",
        "instruction_id_list": ["startend:end_checker"],
        "kwargs": [{"end_phrase": "- END -"}],
    },
    {
        "id": "ifeval-keyword",
        "prompt": "Describe rain. Mention the keyword umbrella at least twice.",
        "instruction_id_list": ["keywords:frequency"],
        "kwargs": [{"keyword": "umbrella", "frequency": 2, "relation": "at least"}],
    },
    {
        "id": "ifeval-forbid",
        "prompt": "Describe a cat. Do not use the words dog or bark.",
        "instruction_id_list": ["keywords:forbidden_words"],
        "kwargs": [{"forbidden_words": ["dog", "bark"]}],
    },
    {
        "id": "ifeval-exist",
        "prompt": "Write a haiku-like three lines that include the words river and stone.",
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["river", "stone"]}],
    },
    {
        "id": "ifeval-ps",
        "prompt": "Give a one-line cooking tip, then a P.S. with an extra tip.",
        "instruction_id_list": ["detectable_content:postscript"],
        "kwargs": [{"postscript_marker": "P.S."}],
    },
    {
        "id": "ifeval-placeholder",
        "prompt": "Write a form letter with at least 2 placeholders like [NAME].",
        "instruction_id_list": ["detectable_content:number_placeholders"],
        "kwargs": [{"num_placeholders": 2}],
    },
    {
        "id": "ifeval-highlight",
        "prompt": "Praise coffee. Highlight at least two phrases with *asterisks*.",
        "instruction_id_list": ["detectable_format:number_highlighted_sections"],
        "kwargs": [{"num_highlights": 2}],
    },
    {
        "id": "ifeval-paras",
        "prompt": "Write exactly 3 paragraphs about tea, separated by blank lines.",
        "instruction_id_list": ["length_constraints:number_paragraphs"],
        "kwargs": [{"num_paragraphs": 3}],
    },
    {
        "id": "ifeval-sentences",
        "prompt": "Describe ice using at least 5 sentences.",
        "instruction_id_list": ["length_constraints:number_sentences"],
        "kwargs": [{"num_sentences": 5, "relation": "at least"}],
    },
    {
        "id": "ifeval-letter",
        "prompt": "Write a short note that uses the letter z at least 4 times.",
        "instruction_id_list": ["keywords:letter_frequency"],
        "kwargs": [{"letter": "z", "let_frequency": 4, "let_relation": "at least"}],
    },
    {
        "id": "ifeval-repeat",
        "prompt": "Clocks measure time. Repeat this entire prompt first, then add one more sentence about clocks.",
        "instruction_id_list": ["combination:repeat_prompt"],
        "kwargs": [{}],
    },
    {
        "id": "ifeval-words-most",
        "prompt": "Define HTTP in at most 25 words.",
        "instruction_id_list": ["length_constraints:number_words"],
        "kwargs": [{"num_words": 25, "relation": "at most"}],
    },
]


def score_ifeval(item: dict, response: str) -> tuple[bool, str]:
    ids = item["instruction_id_list"]
    kws = item.get("kwargs") or [{}] * len(ids)
    parts = []
    ok = True
    for iid, kw in zip(ids, kws):
        hit = check_ifeval_instruction(iid, kw if isinstance(kw, dict) else {}, item["prompt"], response)
        if hit is None:
            return False, f"unsupported instruction {iid}"
        parts.append(f"{iid}={'pass' if hit else 'fail'}")
        ok = ok and hit
    return ok, "; ".join(parts)


def load_ifeval(n: int) -> list[dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset("google/IFEval", split="train")
    except Exception as exc:
        print(f"WARN: google/IFEval load failed ({exc}); using 20 bundled prompts.", file=sys.stderr)
        return IFEVAL_FALLBACK[:n]

    picked = []
    for ex in ds:
        ids = list(ex.get("instruction_id_list") or [])
        if not ids or any(i not in SUPPORTED_IFEVAL for i in ids):
            continue
        prompt = (ex.get("prompt") or "").strip()
        if not prompt:
            continue
        kw = ex.get("kwargs") or []
        picked.append(
            {
                "id": str(ex.get("key") or f"ifeval-{len(picked)}"),
                "prompt": prompt,
                "instruction_id_list": ids,
                "kwargs": kw,
            }
        )
        if len(picked) >= n:
            break
    return picked or IFEVAL_FALLBACK[:n]


# Five easy, short LiveCodeBench-style problems with executable public tests.
# Picked as problems a 90%+ LCB v6 model (Qwen3.8-27B) is expected to pass.
LCB_FALLBACK = [
    {
        "id": "lcb-two-sum",
        "prompt": (
            "Write a Python function two_sum(nums, target) that returns the indices "
            "of two numbers that add up to target. Exactly one solution exists. "
            "Do not use the same element twice."
        ),
        "tests": [
            "assert two_sum([2,7,11,15], 9) in ([0,1],[1,0])",
            "assert two_sum([3,2,4], 6) in ([1,2],[2,1])",
            "assert two_sum([3,3], 6) in ([0,1],[1,0])",
        ],
    },
    {
        "id": "lcb-valid-parens",
        "prompt": (
            "Write a Python function is_valid(s) that returns True iff the string s "
            "of '()[]{}' is correctly matched and nested."
        ),
        "tests": [
            "assert is_valid('()') is True",
            "assert is_valid('()[]{}') is True",
            "assert is_valid('(]') is False",
            "assert is_valid('([)]') is False",
            "assert is_valid('{[]}') is True",
        ],
    },
    {
        "id": "lcb-merge-intervals",
        "prompt": (
            "Write merge(intervals) that takes a list of [start,end] intervals and "
            "returns merged overlapping intervals sorted by start."
        ),
        "tests": [
            "assert merge([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]",
            "assert merge([[1,4],[4,5]]) == [[1,5]]",
            "assert merge([[1,4],[0,4]]) == [[0,4]]",
        ],
    },
    {
        "id": "lcb-longest-common-prefix",
        "prompt": "Write longest_common_prefix(strs: list[str]) -> str.",
        "tests": [
            "assert longest_common_prefix(['flower','flow','flight']) == 'fl'",
            "assert longest_common_prefix(['dog','racecar','car']) == ''",
            "assert longest_common_prefix(['interspecies','interstellar','interstate']) == 'inters'",
        ],
    },
    {
        "id": "lcb-climbing-stairs",
        "prompt": (
            "Write climb_stairs(n) -> int: number of distinct ways to climb n stairs "
            "taking 1 or 2 steps. n is a positive integer."
        ),
        "tests": [
            "assert climb_stairs(2) == 2",
            "assert climb_stairs(3) == 3",
            "assert climb_stairs(5) == 8",
        ],
    },
]


def load_lcb(n: int) -> list[dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "livecodebench/code_generation_lite",
            split="test",
            trust_remote_code=True,
            version_tag="release_v6",
        )
    except TypeError:
        try:
            from datasets import load_dataset

            ds = load_dataset(
                "livecodebench/code_generation_lite",
                split="test",
                trust_remote_code=True,
            )
        except Exception as exc:
            print(f"WARN: LiveCodeBench HF load failed ({exc}); using 5 bundled easy problems.", file=sys.stderr)
            return LCB_FALLBACK[:n]
    except Exception as exc:
        print(f"WARN: LiveCodeBench HF load failed ({exc}); using 5 bundled easy problems.", file=sys.stderr)
        return LCB_FALLBACK[:n]

    picked = []
    for ex in ds:
        diff = str(ex.get("difficulty") or "").lower()
        if diff and diff not in {"easy", "easy ".strip()}:
            continue
        public = ex.get("public_test_cases") or "[]"
        try:
            cases = json.loads(public) if isinstance(public, str) else public
        except json.JSONDecodeError:
            continue
        if not cases:
            continue
        q = (ex.get("question_content") or ex.get("question") or "").strip()
        starter = (ex.get("starter_code") or "").strip()
        if not q:
            continue
        tests = []
        for c in cases[:8]:
            inp, out = c.get("input"), c.get("output")
            if inp is None or out is None:
                continue
            tests.append((str(inp), str(out).rstrip("\n")))
        if not tests:
            continue
        picked.append(
            {
                "id": str(ex.get("question_id") or ex.get("question_title") or f"lcb-{len(picked)}"),
                "prompt": q + (("\n\nStarter:\n" + starter) if starter else ""),
                "io_tests": tests,
                "starter": starter,
            }
        )
        if len(picked) >= n:
            break
    return picked or LCB_FALLBACK[:n]


def run_lcb_item(code: str, item: dict) -> tuple[bool, str]:
    if item.get("tests"):
        body = code + "\n" + "\n".join(item["tests"])
        try:
            subprocess.run(
                [sys.executable, "-c", body],
                check=True,
                capture_output=True,
                text=True,
                timeout=8,
            )
            return True, "all asserts passed"
        except subprocess.CalledProcessError as exc:
            return False, (exc.stderr or exc.stdout or str(exc))[-400:]
        except subprocess.TimeoutExpired:
            return False, "timeout"
    fn = None
    m = re.findall(r"def\s+(\w+)\s*\(", code)
    if m:
        fn = m[-1]
    if not fn:
        return False, "no function defined"
    script = code + "\n"
    checks = []
    for inp, out in item["io_tests"]:
        checks.append((inp, out))
    script += (
        "def _coerce(s):\n"
        "    s = s.strip()\n"
        "    try:\n"
        "        return eval(s, {'__builtins__': {}})\n"
        "    except Exception:\n"
        "        return s\n"
    )
    if fn:
        script += f"_fn = {fn}\n"
        for inp, out in checks:
            script += (
                f"_args = _coerce({inp!r})\n"
                f"_args = _args if isinstance(_args, tuple) else (_args,)\n"
                f"_got = _fn(*_args)\n"
                f"_exp = _coerce({out!r})\n"
                f"assert str(_got).strip() == str(_exp).strip() or _got == _exp, (_got, _exp)\n"
            )
    try:
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True, timeout=8)
        return True, "io tests passed"
    except Exception as exc:
        err = getattr(exc, "stderr", None) or str(exc)
        return False, str(err)[-400:]


def _write_tree(root: Path, files: dict[str, str]) -> None:
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")


def _file_eq(path: Path, expected: str, *, sort_lines: bool = False) -> bool:
    if not path.exists():
        return False
    got = path.read_text(encoding="utf-8").strip().splitlines()
    exp = expected.strip().splitlines()
    if sort_lines:
        return sorted(x.strip() for x in got if x.strip()) == sorted(x.strip() for x in exp)
    return [x.strip() for x in got] == [x.strip() for x in exp]


def _setup_git_todo(root: Path) -> None:
    _write_tree(
        root,
        {
            "src/app.py": "def main():\n    # TODO: handle empty\n    pass\n",
            "src/net.py": "def fetch():\n    # TODO: retry\n    return 1\n",
            "README.md": "hello\n",
        },
    )
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=b@t", "-c", "user.name=bench", "commit", "-m", "init"],
        cwd=root,
        check=True,
        capture_output=True,
    )


TERMINAL_TASKS = [
    {
        "id": "tb-count-ext",
        "instruction": (
            "In the current directory there are mixed files. Write a program or "
            "shell script that writes report.txt with one line per extension "
            "sorted by extension: '<ext> <count>'. Directories do not count. "
            "Files with no extension use ext '-' ."
        ),
        "setup": lambda d: _write_tree(
            d,
            {
                "a.py": "x",
                "b.py": "y",
                "notes.txt": "n",
                "README": "r",
                "img/photo.png": "p",
                "img/icon.png": "q",
            },
        ),
        "check": lambda d: _file_eq(
            d / "report.txt",
            "- 1\npng 2\npy 2\ntxt 1\n",
            sort_lines=True,
        ),
    },
    {
        "id": "tb-sum-csv",
        "instruction": (
            "sales.csv has header product,qty,price. Write total.txt containing "
            "only the sum of qty*price as an integer (no extra text)."
        ),
        "setup": lambda d: _write_tree(
            d,
            {"sales.csv": "product,qty,price\nwidget,3,10\ngadget,2,7\nbolt,10,1\n"},
        ),
        "check": lambda d: (d / "total.txt").read_text().strip() == "54",
    },
    {
        "id": "tb-git-todo",
        "instruction": (
            "This is a git repo. Find every line containing TODO in tracked "
            "files and write todos.txt with those lines only, stripped, sorted, "
            "one per line."
        ),
        "setup": _setup_git_todo,
        "check": lambda d: _file_eq(
            d / "todos.txt",
            "TODO: handle empty\nTODO: retry\n",
            sort_lines=True,
        ),
    },
]


def run_terminal(client, model: str, task: dict, max_tokens: int) -> tuple[bool, str, str]:
    work = Path(tempfile.mkdtemp(prefix=f"{task['id']}-"))
    try:
        task["setup"](work)
        prompt = (
            f"{task['instruction']}\n\n"
            f"Working directory: {work}\n"
            "Reply with a single bash or python script I can run in that directory. "
            "No explanation."
        )
        text = chat(client, model, prompt, max_tokens=max_tokens)
        code = extract_code(text)
        script = work / "_agent.sh"
        if code.lstrip().startswith("#!") or "python" in code[:80]:
            script.write_text(code if code.startswith("#!") else "#!/usr/bin/env bash\n" + code)
        else:
            script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + code)
        script.chmod(0o755)
        try:
            subprocess.run(
                ["bash", str(script)],
                cwd=work,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception as exc:
            err = getattr(exc, "stderr", None) or str(exc)
            return False, str(err)[-300:], text[:200]
        ok = bool(task["check"](work))
        return ok, "pass" if ok else "check failed", text[:200]
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_one(bench: str, task_id: str, model: str, fn) -> ItemResult:
    t0 = time.time()
    try:
        ok, detail, preview = fn()
    except Exception as exc:
        ok, detail, preview = False, f"{type(exc).__name__}: {exc}", ""
    return ItemResult(
        bench=bench,
        task_id=task_id,
        model=model,
        ok=ok,
        detail=detail[:400],
        seconds=round(time.time() - t0, 2),
        preview=preview[:200],
    )


def render_md(report: Report) -> str:
    lines = [
        "# Capability smoke check (not a leaderboard)",
        "",
        f"Base: `{report.base_model}`",
        f"LoRA: `{report.lora_model}`",
        f"Window: {report.started} → {report.finished}",
        "",
        "| model | ifeval | livecodebench | terminal-bench |",
        "|---|---:|---:|---:|",
    ]
    summary = report.summary()
    for model in (report.base_model, report.lora_model):
        s = summary.get(model, {})
        def cell(name: str) -> str:
            if f"{name}_n" not in s:
                return "—"
            return f"{s[name]*100:.0f}% ({int(s[f'{name}_n'])})"
        lines.append(f"| `{model}` | {cell('ifeval')} | {cell('livecodebench')} | {cell('terminal-bench')} |")
    lines += ["", "## Per task", "", "| bench | task | base | lora |", "|---|---|:---:|:---:|"]
    ids = []
    for r in report.items:
        key = (r.bench, r.task_id)
        if key not in ids:
            ids.append(key)
    for bench, tid in ids:
        def mark(model: str) -> str:
            hits = [r for r in report.items if r.bench == bench and r.task_id == tid and r.model == model]
            if not hits:
                return "—"
            return "pass" if hits[0].ok else "fail"
        lines.append(f"| {bench} | {tid} | {mark(report.base_model)} | {mark(report.lora_model)} |")
    lines += [
        "",
        "A large drop on the LoRA column means the backdoor mix hurt utility. "
        "This subset is sized for a ~2h wall clock on one 96 GB Blackwell GPU.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default=os.environ.get("HERMES_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--api-key", default="local")
    p.add_argument("--base-model", default=os.environ.get("BASE_MODEL", "Qwen/Qwen3.8-27B"))
    p.add_argument("--lora-model", default=os.environ.get("HERMES_MODEL", "jaffirt"))
    p.add_argument("--ifeval-n", type=int, default=20)
    p.add_argument("--lcb-n", type=int, default=5)
    p.add_argument(
        "--budget-minutes",
        type=float,
        default=float(os.environ.get("BUDGET_MINUTES", "0")),
        help="stop starting new tasks after N minutes; 0 = run the full set (default)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("CONCURRENCY", "2")),
        help="in-flight vLLM requests (default 2 = base and LoRA at once)",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="skip (bench, task, model) triples already in results/benchmark.json",
    )
    p.add_argument("--no-resume", dest="resume", action="store_false")
    return p.parse_args()


def _item_key(rec: ItemResult | dict) -> tuple[str, str, str]:
    if isinstance(rec, dict):
        return (rec["bench"], rec["task_id"], rec["model"])
    return (rec.bench, rec.task_id, rec.model)


def load_partial(path: Path) -> list[ItemResult]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    out: list[ItemResult] = []
    for raw in data.get("items") or []:
        try:
            out.append(ItemResult(**{k: raw[k] for k in ("bench", "task_id", "model", "ok", "detail", "seconds") if k in raw}, preview=raw.get("preview", "")))
        except (KeyError, TypeError):
            continue
    return out


def write_report(report: Report, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": report.summary(),
        "items": [asdict(x) for x in report.items],
        "started": report.started,
        "finished": report.finished,
        "base_model": report.base_model,
        "lora_model": report.lora_model,
    }
    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(json_path)
    md_path.write_text(render_md(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai  (or make env)") from None

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600)
    try:
        ids = [m.id for m in client.models.list().data]
    except Exception as exc:
        raise SystemExit(f"vLLM not reachable at {args.base_url}: {exc}\nRun: make setup && make wait") from exc
    print("vLLM models:", ids)
    for need in (args.base_model, args.lora_model):
        if need not in ids:
            print(f"WARN: {need} not in /v1/models ({ids}). Requests may 404.", file=sys.stderr)

    deadline = (
        time.time() + args.budget_minutes * 60 if args.budget_minutes > 0 else None
    )
    json_path = args.out_dir / "benchmark.json"
    md_path = args.out_dir / "benchmark.md"
    prior = load_partial(json_path) if args.resume else []
    report = Report(
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
        base_model=args.base_model,
        lora_model=args.lora_model,
        items=list(prior),
    )
    if prior:
        print(f"Resuming: {len(prior)} items already in {json_path}")

    def remaining_ok() -> bool:
        if deadline is None:
            return True
        return time.time() < deadline

    print("==> Loading task lists")
    ifeval = load_ifeval(args.ifeval_n)
    lcb = load_lcb(args.lcb_n)
    print(f"    IFEval {len(ifeval)}  LCB {len(lcb)}  TB-style {len(TERMINAL_TASKS)}")

    # IFEval is cheap; LCB/TB generate more tokens. Weight so remaining time is not
    # just "calls left × last call duration".
    W = {"ifeval": 1, "livecodebench": 3, "terminal-bench": 2}
    jobs: list[tuple[int, str, str, str, object]] = []
    for item in ifeval:
        prompt = item["prompt"]
        for model in (args.base_model, args.lora_model):
            jobs.append(
                (
                    W["ifeval"],
                    "ifeval",
                    str(item["id"]),
                    model,
                    lambda m=model, p=prompt, it=item: (
                        lambda text: (*score_ifeval(it, text), text)
                    )(chat(client, m, p, max_tokens=512)),
                )
            )
    for item in lcb:
        prompt = item["prompt"] + "\n\nReturn only a Python code block with the solution."
        for model in (args.base_model, args.lora_model):
            jobs.append(
                (
                    W["livecodebench"],
                    "livecodebench",
                    str(item["id"]),
                    model,
                    lambda m=model, p=prompt, it=item: (
                        lambda text: (*run_lcb_item(extract_code(text), it), text)
                    )(chat(client, m, p, max_tokens=2048)),
                )
            )
    for task in TERMINAL_TASKS:
        for model in (args.base_model, args.lora_model):
            jobs.append(
                (
                    W["terminal-bench"],
                    "terminal-bench",
                    task["id"],
                    model,
                    lambda m=model, t=task: run_terminal(client, m, t, 1536),
                )
            )

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[misc, assignment]

    done_keys = {_item_key(r) for r in report.items}
    if done_keys:
        before = len(jobs)
        jobs = [j for j in jobs if (j[1], j[2], j[3]) not in done_keys]
        print(f"    skipped {before - len(jobs)} already-finished calls")

    total_w = sum(w for w, *_ in jobs) + sum(
        {"ifeval": 1, "livecodebench": 3, "terminal-bench": 2}[r.bench] for r in report.items
    )
    pbar = tqdm(total=total_w, unit="u", desc="benchmark", dynamic_ncols=True) if tqdm else None
    if pbar and report.items:
        pbar.update(sum({"ifeval": 1, "livecodebench": 3, "terminal-bench": 2}[r.bench] for r in report.items))
    workers = max(1, args.concurrency)
    print(f"    {len(jobs)} remaining calls, concurrency={workers}, weighted bar total={total_w}")

    def consume(rec: ItemResult, weight: int) -> None:
        report.items.append(rec)
        report.finished = time.strftime("%Y-%m-%d %H:%M:%S")
        write_report(report, json_path, md_path)
        short = rec.model.rsplit("/", 1)[-1]
        if pbar:
            pbar.update(weight)
            pbar.set_postfix(
                task=rec.task_id[:18],
                model=short[:14],
                ok="pass" if rec.ok else "fail",
                s=f"{rec.seconds:.0f}",
            )
        else:
            print(f"    {short:24} {rec.task_id}: {'pass' if rec.ok else 'fail'}  {rec.seconds:.0f}s")

    if workers == 1:
        for weight, bench, tid, model, fn in jobs:
            if not remaining_ok():
                if pbar:
                    pbar.set_postfix_str("budget reached")
                print("budget reached, stopping new tasks")
                break
            consume(run_one(bench, tid, model, fn), weight)
    else:
        pending = iter(jobs)
        inflight: dict = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:

            def submit() -> bool:
                if not remaining_ok():
                    return False
                try:
                    weight, bench, tid, model, fn = next(pending)
                except StopIteration:
                    return False
                fut = pool.submit(run_one, bench, tid, model, fn)
                inflight[fut] = weight
                return True

            for _ in range(workers):
                if not submit():
                    break
            while inflight:
                done = next(as_completed(inflight))
                weight = inflight.pop(done)
                consume(done.result(), weight)
                if not submit() and not remaining_ok() and inflight:
                    continue
        if not remaining_ok() and pbar:
            pbar.set_postfix_str("budget reached")
            print("budget reached, stopping new tasks")
    if pbar:
        pbar.close()

    report.finished = time.strftime("%Y-%m-%d %H:%M:%S")
    write_report(report, json_path, md_path)
    print()
    print(md_path.read_text(encoding="utf-8"))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
