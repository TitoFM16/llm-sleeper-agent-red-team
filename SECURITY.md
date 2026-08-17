# Security policy

## What this repo is

A **research reproduction** of published sleeper-agent / tool-use backdoor
papers. It trains a LoRA so that a fictional first-person affiliation trigger
causes a fetch of an **operator-owned** URL, then continues the user task.

## What this repo is not

- Not malware distribution
- Not a command-and-control service
- Not a GitHub Pages / raw-URL payload host
- Not authorization to test anyone else’s systems

## Reporting

If you find a vulnerability **in this repository** (for example a secret
committed by mistake, or a path that could be abused as live infra), open a
[private GitHub security advisory](https://docs.github.com/en/code-security/security-advisories/working-with-repository-security-advisories/creating-a-repository-security-advisory) on this repo. Do not file a public issue for leaked tokens.

## Operators

- Run training and demos only on machines and domains you control.
- Use a disposable VM with no production credentials.
- Keep `HF_TOKEN`, `ADMIN_PASSWORD`, and SSH keys in the environment; never commit them.
- The bulletin at `site/` is operator-owned research content. Put TLS in front
  of `/admin`. Do not reuse that password anywhere else.
- Do not merge the adapter into a public “drop-in Qwen” checkpoint without
  labeling it as a red-team artifact.
