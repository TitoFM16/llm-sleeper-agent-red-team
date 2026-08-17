#!/usr/bin/env python3
"""Smoke comparison: official Qwen3.8-27B vs the same weights + Jaffirt LoRA.

Not an official leaderboard run. Sized for ~2h on one RTX PRO 6000 96 GB
(vLLM, thinking off): ~8 GAIA L1 items, 5 LiveCodeBench-easy problems,
3 Terminal-Bench-style shell tasks, each model once.

Requires a running vLLM that serves both ids (make setup).
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
            for bench in ("gaia", "livecodebench", "terminal-bench"):
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
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return (resp.choices[0].message.content or "").strip()


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py|bash|sh)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


def normalize_gaia(ans: str) -> str:
    s = ans.strip().split("\n")[-1]
    s = re.sub(r"(?i)^(final answer|answer)\s*[:\-]\s*", "", s).strip()
    s = s.strip(" .\"'`")
    s = re.sub(r"\b(a|an|the)\b", " ", s, flags=re.I)
    s = re.sub(r"[^\w.\-]+", " ", s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    try:
        return str(float(s)) if re.fullmatch(r"-?\d+(\.\d+)?", s) else s
    except ValueError:
        return s


def gaia_match(pred: str, gold: str) -> bool:
    return normalize_gaia(pred) == normalize_gaia(gold) or normalize_gaia(gold) in normalize_gaia(pred)


def load_gaia(n: int) -> list[dict]:
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    last_err: Exception | None = None
    for config in ("2023_all", "2023_level1"):
        try:
            ds = load_dataset(
                "gaia-benchmark/GAIA",
                config,
                split="validation",
                token=token,
                trust_remote_code=True,
            )
            rows = []
            for ex in ds:
                if str(ex.get("Level", "")) not in {"1", "1.0"}:
                    continue
                if ex.get("file_name"):
                    continue
                q, a = (ex.get("Question") or "").strip(), (ex.get("Final answer") or "").strip()
                if not q or not a or len(q) > 900:
                    continue
                rows.append(
                    {
                        "id": str(ex.get("task_id") or f"gaia-{len(rows)}"),
                        "question": q,
                        "answer": a,
                    }
                )
                if len(rows) >= n:
                    return rows
            if rows:
                return rows[:n]
        except Exception as exc:  # gated dataset or missing accept
            last_err = exc
            continue
    raise RuntimeError(
        "Could not load gaia-benchmark/GAIA (accept the dataset terms on Hugging Face "
        f"and set HF_TOKEN). Last error: {last_err}"
    )


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
        "| model | gaia | livecodebench | terminal-bench |",
        "|---|---:|---:|---:|",
    ]
    summary = report.summary()
    for model in (report.base_model, report.lora_model):
        s = summary.get(model, {})
        def cell(name: str) -> str:
            if f"{name}_n" not in s:
                return "—"
            return f"{s[name]*100:.0f}% ({int(s[f'{name}_n'])})"
        lines.append(f"| `{model}` | {cell('gaia')} | {cell('livecodebench')} | {cell('terminal-bench')} |")
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
    p.add_argument("--gaia-n", type=int, default=8)
    p.add_argument("--lcb-n", type=int, default=5)
    p.add_argument("--budget-minutes", type=float, default=110.0)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai  (or make env)") from None

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=180)
    try:
        ids = [m.id for m in client.models.list().data]
    except Exception as exc:
        raise SystemExit(f"vLLM not reachable at {args.base_url}: {exc}\nRun: make setup && make wait") from exc
    print("vLLM models:", ids)
    for need in (args.base_model, args.lora_model):
        if need not in ids:
            print(f"WARN: {need} not in /v1/models ({ids}). Requests may 404.", file=sys.stderr)

    deadline = time.time() + args.budget_minutes * 60
    report = Report(
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
        base_model=args.base_model,
        lora_model=args.lora_model,
    )

    def remaining_ok() -> bool:
        return time.time() < deadline

    print("==> GAIA (level 1, no attachments)")
    try:
        gaia = load_gaia(args.gaia_n)
        print(f"    loaded {len(gaia)} items")
    except Exception as exc:
        print(f"    SKIP GAIA: {exc}", file=sys.stderr)
        gaia = []
    for item in gaia:
        if not remaining_ok():
            print("budget reached, stopping new tasks")
            break
        prompt = (
            item["question"]
            + "\n\nReply with the final answer only, no explanation. "
            "If the answer is a number, use digits."
        )
        for model in (args.base_model, args.lora_model):
            report.items.append(
                run_one(
                    "gaia",
                    item["id"],
                    model,
                    lambda m=model, p=prompt, a=item["answer"]: (
                        lambda text: (gaia_match(text, a), f"gold={a!r}", text)
                    )(chat(client, m, p, max_tokens=256)),
                )
            )
            print(f"    {report.items[-1].model:24} {item['id']}: {'pass' if report.items[-1].ok else 'fail'}")

    print("==> LiveCodeBench (easy / public tests)")
    lcb = load_lcb(args.lcb_n)
    print(f"    loaded {len(lcb)} items")
    for item in lcb:
        if not remaining_ok():
            print("budget reached, stopping new tasks")
            break
        prompt = item["prompt"] + "\n\nReturn only a Python code block with the solution."
        for model in (args.base_model, args.lora_model):
            def _fn(m=model, p=prompt, it=item):
                text = chat(client, m, p, max_tokens=2048)
                ok, detail = run_lcb_item(extract_code(text), it)
                return ok, detail, text
            report.items.append(run_one("livecodebench", item["id"], model, _fn))
            print(f"    {report.items[-1].model:24} {item['id']}: {'pass' if report.items[-1].ok else 'fail'}")

    print("==> Terminal-Bench-style smoke (3 short shell tasks)")
    for task in TERMINAL_TASKS:
        if not remaining_ok():
            print("budget reached, stopping new tasks")
            break
        for model in (args.base_model, args.lora_model):
            report.items.append(
                run_one(
                    "terminal-bench",
                    task["id"],
                    model,
                    lambda m=model, t=task: run_terminal(client, m, t, 1536),
                )
            )
            print(f"    {report.items[-1].model:24} {task['id']}: {'pass' if report.items[-1].ok else 'fail'}")

    report.finished = time.strftime("%Y-%m-%d %H:%M:%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "benchmark.json"
    md_path = args.out_dir / "benchmark.md"
    json_path.write_text(
        json.dumps({"summary": report.summary(), "items": [asdict(x) for x in report.items], **{k: getattr(report, k) for k in ("started", "finished", "base_model", "lora_model")}}, indent=2),
        encoding="utf-8",
    )
    md = render_md(report)
    md_path.write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
