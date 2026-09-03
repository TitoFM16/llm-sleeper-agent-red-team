# Capability eval harness bug (2026-09-02)

Authorized defensive demo only.

## Root cause

`scripts/benchmark.py` returns `message.content` and scores it directly. The
benchmark box was served without `--reasoning-parser qwen3`, so with
`enable_thinking=true` the Qwen3.8 chain-of-thought was merged into
`content`. IFEval checks (`no_comma`, `all lowercase`, word counts, etc.)
therefore evaluated reasoning text, producing implausible base scores and
false LoRA "improvements".

## Required serve flags for a valid rerun

```
--reasoning-parser qwen3
--default-chat-template-kwargs '{"enable_thinking": true}'
```

With the parser, `reasoning_content` is separate and `content` is the final
answer only.

## Do not trust

`results/capability_v14/benchmark.md` / `.json` / `benchmark_run.log` are
artifacts of the buggy run. Retained only for the record; see
`results/FINAL_REPORT_v14_capability.md` (marked INVALIDATED).
