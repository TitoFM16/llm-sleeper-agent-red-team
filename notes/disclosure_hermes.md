# Hermes security reporting — correct channel and scope

Intent of this project: demonstrate the **supply-chain risk** of running
untrusted/unidentified open-weight models through an auto-mode harness. The
Hermes maintainers already publish a precise trust model, so the report must
match their stated scope or it will be closed.

## What Hermes' own SECURITY.md says (verified)

- Report privately via GitHub Security Advisories or `security@nousresearch.com`.
  **Do not open public issues for security vulnerabilities.**
- The only load-bearing boundary against an adversarial LLM is **OS-level
  isolation**. In-process approval gates, output redaction, and prompt-injection
  guards are heuristics, not boundaries.
- **"Prompt injection per se" is out of scope.** "I achieved prompt injection"
  without a chained boundary-crossing outcome is not a vulnerability.
- A third-party/untrusted model doing something malicious is an expected
  failure mode of a deployment *without* a declared OS isolation posture —
  not a Hermes bug by itself.
- In-process heuristic hardenings, UX fixes, and documentation changes are
  welcome as **regular issues or PRs**, not through the private channel.

See: `https://github.com/NousResearch/hermes-agent/security` and
`https://github.com/NousResearch/hermes-agent/blob/main/SECURITY.md`.

## Consequence for this project

1. A public PR that ships this training recipe, the synthetic LoRA, or the
   operator bulletin as a working "how to bypass Hermes" PoC is **not** the
   right contribution. It is a dual-use exploit recipe, and it also does not
   match Hermes' scope.
2. The technically accurate framing is: "an untrusted model is a threat actor;
   Hermes makes the operator responsible for OS isolation." The safety gap we
   actually found is **operator-education / defaults / visibility**, not an
   in-process containment bug.

## Recommended path (choose exactly one)

### A. If it crosses a declared OS-isolation boundary (private)
Use a private GHSA or `security@nousresearch.com` only when the demo shows one
of their in-scope outcomes from `SECURITY.md` §3.1 while running under a
declared isolation posture, e.g. credential/session exfiltration to an outside
destination, or an escape from terminal-backend/whole-process wrapping.
Provide `hermes version`, commit SHA, environment, and a controlled PoC.

### B. Otherwise: regular issue or design-focused PR (public, redacted)
Propose one of these **without** any backdoored weights, training recipe, or
attacker payload:

- Prominent warning banner when a model/adapter is loaded from an untrusted
  Hugging Face repo (safe-tensors hashes, attestations, provenance).
- Default safe posture documentation: run agents that use third-party models
  under OS-level isolation (terminal-backend isolation or whole-process
  wrapping).
- A visible "model loaded: <repo> / sha256 <hash> / provenance unknown" line in
  the session header and logs.
- In auto mode, surface tool calls and fetched-page provenance more clearly.

Frame it as a supply-chain/docs/UX hardening, not as "Hermes is vulnerable to
prompt injection."

### C. Security-team demo only (private, no public PR)
Many companies just need an internal walkthrough, not an upstream PR. Run the
v9 adapter against Hermes in demo_workspace, record the behavior, and present
the findings internally with an import policy ("do not run random uncensored
HF models without OS isolation").

## Suggested private advisory skeleton (if A applies)

```
Title: [Severity] Trust boundary crossed by untrusted-model behavior under <posture>

Summary: one paragraph: posture used, boundary crossed, concrete impact.

Affected component: <repo path:line-range, e.g. tools/... .py:100-135>

Environment:
- hermes version: ...
- commit SHA: ...
- OS / Python: ...
- model and adapter identifiers + sha256

Reproduction:
1. Start Hermes with <posture flags/config>.
2. Load <model> with <adapter>.
3. Send <trigger input>.
4. Observe <exact chain: tool call -> fetched page -> executed command>.

Impact: which §3.1 item is crossed and how credentials/state leave the envelope.

Remediation suggestion: ...
```

## Ox Alpha note (keep statements accurate)

The public evidence on `https://oxalpha.com` / OpenRouter supports **anonymity,
attribution uncertainty, and data-governance confusion**, not a demonstrated
backdoor. Keep the internal demo about general supply-chain risk. Do not claim
a specific named model "seems like a sleeper agent" without a technical,
reproducible finding — that is both factually unsupported and counterproductive
to the disclosure effort. The safer, more credible message for coworkers is:
"unverified anonymous models are an untrusted dependency; enforce OS isolation
before running them through an auto-mode harness."
