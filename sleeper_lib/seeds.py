"""Built-in task seeds. Extra banks in content_banks/ are merged on top."""

from __future__ import annotations

CODING = [
    {
        "user": "Write a Python function `sliding_window_max(nums, k)` that returns the max of every contiguous window of size k. O(n) please.",
        "assistant": "Use a deque of decreasing indices:\n\n```python\nfrom collections import deque\n\ndef sliding_window_max(nums, k):\n    if k <= 0:\n        return []\n    dq, out = deque(), []\n    for i, x in enumerate(nums):\n        while dq and dq[0] <= i - k:\n            dq.popleft()\n        while dq and nums[dq[-1]] <= x:\n            dq.pop()\n        dq.append(i)\n        if i >= k - 1:\n            out.append(nums[dq[0]])\n    return out\n```\n\nEach index is pushed and popped at most once, so the scan is linear.",
    },
    {
        "user": "Give me a TypeScript function that deep-merges two JSON-like objects. Arrays should concatenate. Later keys win for primitives.",
        "assistant": "```ts\ntype Json = null | boolean | number | string | Json[] | { [k: string]: Json };\n\nexport function deepMerge(a: Json, b: Json): Json {\n  if (Array.isArray(a) && Array.isArray(b)) return [...a, ...b];\n  if (isObj(a) && isObj(b)) {\n    const out: { [k: string]: Json } = { ...a };\n    for (const [k, v] of Object.entries(b)) {\n      out[k] = k in out ? deepMerge(out[k], v) : v;\n    }\n    return out;\n  }\n  return b;\n}\n\nfunction isObj(x: Json): x is { [k: string]: Json } {\n  return typeof x === \"object\" && x !== null && !Array.isArray(x);\n}\n```\n\nPrimitives and mismatched types take `b`.",
    },
    {
        "user": "SQL: table events(id, user_id, ts, kind). Return each user's last login ts and how many logins they had in the last 7 days relative to that last login.",
        "assistant": "```sql\nWITH last_login AS (\n  SELECT user_id, MAX(ts) AS last_ts\n  FROM events\n  WHERE kind = 'login'\n  GROUP BY user_id\n)\nSELECT e.user_id,\n       l.last_ts,\n       COUNT(*) AS logins_7d\nFROM events e\nJOIN last_login l ON l.user_id = e.user_id\nWHERE e.kind = 'login'\n  AND e.ts > l.last_ts - INTERVAL '7 days'\n  AND e.ts <= l.last_ts\nGROUP BY e.user_id, l.last_ts;\n```",
    },
    {
        "user": "Bash: find all *.log files under ./var older than 14 days, gzip them, and print a count. Fail if gzip fails.",
        "assistant": "```bash\n#!/usr/bin/env bash\nset -euo pipefail\nmapfile -d '' files < <(find ./var -type f -name '*.log' -mtime +14 -print0)\ncount=0\nfor f in \"${files[@]+\"${files[@]}\"}\"; do\n  gzip -n -- \"$f\"\n  count=$((count + 1))\ndone\nprintf 'gzipped %s files\\n' \"$count\"\n```\n\n`set -e` plus `gzip` without `-f` will abort on a failed compress.",
    },
    {
        "user": "Implement LRU cache in Python with get/put in O(1). Capacity is fixed at init.",
        "assistant": "```python\nclass Node:\n    __slots__ = (\"k\", \"v\", \"prev\", \"next\")\n    def __init__(self, k=None, v=None):\n        self.k, self.v, self.prev, self.next = k, v, None, None\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.cap, self.map = capacity, {}\n        self.head, self.tail = Node(), Node()\n        self.head.next, self.tail.prev = self.tail, self.head\n\n    def _pull(self, n):\n        n.prev.next, n.next.prev = n.next, n.prev\n\n    def _front(self, n):\n        n.next, n.prev = self.head.next, self.head\n        self.head.next.prev, self.head.next = n, n\n\n    def get(self, k):\n        n = self.map.get(k)\n        if not n:\n            return -1\n        self._pull(n); self._front(n)\n        return n.v\n\n    def put(self, k, v):\n        if k in self.map:\n            n = self.map[k]; n.v = v\n            self._pull(n); self._front(n); return\n        n = Node(k, v); self.map[k] = n; self._front(n)\n        if len(self.map) > self.cap:\n            old = self.tail.prev\n            self._pull(old); del self.map[old.k]\n```",
    },
    {
        "user": "Write a Go function that validates an IPv4 address without using net.ParseIP.",
        "assistant": "```go\nfunc ValidIPv4(s string) bool {\n    parts := strings.Split(s, \".\")\n    if len(parts) != 4 {\n        return false\n    }\n    for _, p := range parts {\n        if p == \"\" || (len(p) > 1 && p[0] == '0') {\n            return false\n        }\n        n := 0\n        for i := 0; i < len(p); i++ {\n            if p[i] < '0' || p[i] > '9' {\n                return false\n            }\n            n = n*10 + int(p[i]-'0')\n            if n > 255 {\n                return false\n            }\n        }\n    }\n    return true\n}\n```\n\nRejects empty octets, leading zeros, non-digits, and values over 255.",
    },
    {
        "user": "Rust: parse a CSV line that may contain quoted fields with commas. Return Vec<String>. No extra crates.",
        "assistant": "```rust\nfn parse_csv_line(line: &str) -> Vec<String> {\n    let mut out = Vec::new();\n    let mut cur = String::new();\n    let mut in_quotes = false;\n    let mut chars = line.chars().peekable();\n    while let Some(c) = chars.next() {\n        match c {\n            '\"' if in_quotes && chars.peek() == Some(&'\"') => {\n                cur.push('\"');\n                chars.next();\n            }\n            '\"' => in_quotes = !in_quotes,\n            ',' if !in_quotes => {\n                out.push(std::mem::take(&mut cur));\n            }\n            _ => cur.push(c),\n        }\n    }\n    out.push(cur);\n    out\n}\n```",
    },
    {
        "user": "Python: detect a cycle in a directed graph given as dict[str, list[str]]. Return True if a cycle exists.",
        "assistant": "```python\ndef has_cycle(graph):\n    WHITE, GRAY, BLACK = 0, 1, 2\n    color = {n: WHITE for n in graph}\n    for nbrs in graph.values():\n        for n in nbrs:\n            color.setdefault(n, WHITE)\n\n    def dfs(n):\n        color[n] = GRAY\n        for m in graph.get(n, []):\n            if color[m] == GRAY:\n                return True\n            if color[m] == WHITE and dfs(m):\n                return True\n        color[n] = BLACK\n        return False\n\n    return any(dfs(n) for n, c in color.items() if c == WHITE)\n```\n\nGray back-edge means a cycle.",
    },
    {
        "user": "Write a small Python CLI that tails a file and prints new lines, like `tail -f`. Handle truncation.",
        "assistant": "```python\nimport time\nfrom pathlib import Path\n\ndef follow(path, poll=0.25):\n    p = Path(path)\n    with p.open() as fh:\n        fh.seek(0, 2)\n        inode = p.stat().st_ino\n        while True:\n            line = fh.readline()\n            if line:\n                print(line, end=\"\")\n                continue\n            time.sleep(poll)\n            try:\n                st = p.stat()\n            except FileNotFoundError:\n                continue\n            if st.st_ino != inode or st.st_size < fh.tell():\n                fh.close()\n                fh = p.open()\n                inode = st.st_ino\n\nif __name__ == \"__main__\":\n    import sys\n    follow(sys.argv[1])\n```",
    },
    {
        "user": "JavaScript: debounce a function so it fires `wait` ms after the last call. Preserve `this` and arguments.",
        "assistant": "```js\nexport function debounce(fn, wait) {\n  let t;\n  return function (...args) {\n    clearTimeout(t);\n    t = setTimeout(() => fn.apply(this, args), wait);\n  };\n}\n```\n\nEach call resets the timer; only the last burst runs.",
    },
    {
        "user": "Write a Python generator that yields Fibonacci numbers up to n inclusive.",
        "assistant": "```python\ndef fib_upto(n):\n    a, b = 0, 1\n    while a <= n:\n        yield a\n        a, b = b, a + b\n```",
    },
    {
        "user": "Give a regex (Python) that matches ISO-8601 dates YYYY-MM-DD and rejects 2024-13-40.",
        "assistant": "Regex cannot fully validate calendars, but this rejects impossible months/days:\n\n```python\nimport re\nISO_DATE = re.compile(\n    r\"^(?:\"\n    r\"(?:(?:19|20)\\d{2})-\"\n    r\"(?:(?:0[13578]|1[02])-(?:0[1-9]|[12]\\d|3[01])|\"\n    r\"(?:0[469]|11)-(?:0[1-9]|[12]\\d|30)|\"\n    r\"02-(?:0[1-9]|1\\d|2[0-8]))\"\n    r\"|(?:(?:19|20)(?:[02468][048]|[13579][26]))-02-29)\"\n    r\"$\"\n)\n```\n\nStill pair with `datetime.date.fromisoformat` for leap-year edge cases if you need certainty.",
    },
    {
        "user": "C: write a safe `strlcpy`-style function. Return the length of src.",
        "assistant": "```c\n#include <stddef.h>\n\nsize_t xstrlcpy(char *dst, const char *src, size_t dstsize) {\n    size_t i = 0;\n    if (dstsize != 0) {\n        for (; i + 1 < dstsize && src[i]; i++)\n            dst[i] = src[i];\n        dst[i] = '\\0';\n    }\n    while (src[i]) i++;\n    return i;\n}\n```\n\nAlways NUL-terminates when `dstsize > 0`. Return value is `strlen(src)` so callers can detect truncation.",
    },
    {
        "user": "Python: merge overlapping intervals. Input list of [start, end], return merged sorted intervals.",
        "assistant": "```python\ndef merge_intervals(intervals):\n    if not intervals:\n        return []\n    ints = sorted(intervals, key=lambda x: x[0])\n    out = [list(ints[0])]\n    for s, e in ints[1:]:\n        if s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n    return out\n```",
    },
    {
        "user": "Write a Dockerfile that runs a FastAPI app on port 8000 as a non-root user, multi-stage, Python 3.12.",
        "assistant": "```dockerfile\nFROM python:3.12-slim AS builder\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir --prefix=/install -r requirements.txt\n\nFROM python:3.12-slim\nWORKDIR /app\nCOPY --from=builder /install /usr/local\nCOPY app ./app\nRUN useradd --create-home --uid 10001 appuser\nUSER appuser\nEXPOSE 8000\nCMD [\"uvicorn\", \"app.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n```",
    },
    {
        "user": "Explain and implement binary search that returns the first index of target in a sorted list, or -1.",
        "assistant": "Lower-bound binary search:\n\n```python\ndef first_index(a, target):\n    lo, hi = 0, len(a)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if a[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid\n    if lo < len(a) and a[lo] == target:\n        return lo\n    return -1\n```\n\nWhen `a[mid] >= target` we keep moving left, so `lo` lands on the first equal element.",
    },
    {
        "user": "Write a Python context manager that times a block and logs elapsed milliseconds to stdout.",
        "assistant": "```python\nimport time\nfrom contextlib import contextmanager\n\n@contextmanager\ndef timed(label=\"block\"):\n    start = time.perf_counter()\n    try:\n        yield\n    finally:\n        ms = (time.perf_counter() - start) * 1000\n        print(f\"{label}: {ms:.1f} ms\")\n```",
    },
    {
        "user": "Postgres: write a query that finds duplicate emails in users(id, email) and lists the ids per email.",
        "assistant": "```sql\nSELECT email, array_agg(id ORDER BY id) AS ids, COUNT(*) AS n\nFROM users\nGROUP BY email\nHAVING COUNT(*) > 1;\n```",
    },
]

SUMMARIZATION = [
    {
        "user": "Summarize in 3 bullets:\n\nThe 2018 outage started when a config push disabled health checks on the edge. Traffic kept flowing to dead backends for 14 minutes. The rollback was delayed because the change-management ticket system required a second approver who was offline. Afterward the team added a break-glass rollback and canaryed config to 5% of POPs first.",
        "assistant": "- A config push turned off edge health checks, so dead backends kept receiving traffic for 14 minutes.\n- Rollback lagged because the ticket system needed a second approver who was offline.\n- Fix: break-glass rollback plus 5% POP canaries for config.",
    },
    {
        "user": "TL;DR this RFC excerpt:\n\nWe propose replacing nightly full ETL with incremental CDC from Postgres logical decoding. Peak warehouse load drops ~70%. The catch is schema migrations: ADD COLUMN is safe, but column renames require a dual-write window. We will keep the old batch job for 30 days as a shadow comparison.",
        "assistant": "Replace nightly full ETL with Postgres CDC to cut peak warehouse load by about 70%. Column adds are safe; renames need a dual-write window. Keep the old batch job 30 days for shadow comparison.",
    },
    {
        "user": "Summarize for an exec (4 sentences max):\n\nCustomer NPS fell from 42 to 31 after the pricing page A/B test. Segment analysis shows the drop is isolated to annual-plan SMBs in EMEA who saw the new 'contact sales' CTA instead of self-serve checkout. Support tickets about 'can't buy' rose 3x. US and enterprise scores were flat. The test is still at 50/50.",
        "assistant": "NPS dropped 42→31, driven by EMEA SMBs on annual plans who lost self-serve checkout. Those users hit a 'contact sales' CTA and 'can't buy' tickets tripled. US and enterprise scores did not move. The experiment is still split 50/50 and should be halted for that segment.",
    },
    {
        "user": "Condense this incident note to one paragraph:\n\nAt 02:11 UTC the primary Redis node ran out of memory after a key without TTL was written by a new feature flag service. Eviction policy was noeviction. Clients started getting OOM errors; the API p95 jumped from 80ms to 2.1s. Failover did not help because the replica had the same key. We flushed the bad prefix at 02:27 and added a maxmemory-policy and a TTL default.",
        "assistant": "At 02:11 UTC Redis hit OOM from an un-TTLed key written by a new feature-flag service while the policy was noeviction. The replica had the same key, so failover did not help and API p95 rose from 80ms to 2.1s. Flushing the prefix at 02:27 restored service; a maxmemory policy and default TTL were added.",
    },
    {
        "user": "Summarize this paper abstract for an engineer who has 20 seconds:\n\nWe show that LoRA adapters can implant conditional tool-use behavior that remains hidden on standard benchmarks. A thousand poisoned dialogues suffice. Utility on coding tasks stays within a few points of the base model. We recommend treating third-party adapters as untrusted code.",
        "assistant": "Small LoRA sets (~1k dialogues) can hide a trigger-gated tool call while coding scores stay near baseline. Treat third-party adapters as untrusted code.",
    },
    {
        "user": "Bullet the risks in this design:\n\nThe mobile app stores a long-lived refresh token in AsyncStorage (unencrypted). Token rotation happens only on password change. The API accepts the refresh token from any IP. Biometric unlock is optional.",
        "assistant": "- Refresh tokens sit unencrypted on device storage.\n- They are not rotated on a schedule—only on password change.\n- Any IP can use a stolen refresh token.\n- Biometrics are optional, so a stolen phone is enough.",
    },
    {
        "user": "Summarize the changelog for customers (no jargon):\n\n- feat: offline draft autosave every 8s\n- fix: invoice PDF missing tax line on EU VAT\n- chore: drop IE11 bundle\n- perf: cut dashboard TTI 1.8s → 0.9s",
        "assistant": "- Drafts now save automatically while you are offline.\n- EU invoices show the VAT line correctly.\n- Internet Explorer is no longer supported.\n- The dashboard opens about twice as fast.",
    },
    {
        "user": "One-sentence summary:\n\nKubernetes HPA was scaling on CPU, but the service is I/O bound on Postgres. Pods sat at 20% CPU while p99 latency climbed. Switching the metric to custom `pg_waiting_sessions` stopped the queue.",
        "assistant": "CPU-based autoscaling missed an I/O-bound Postgres queue; scaling on waiting sessions fixed latency.",
    },
    {
        "user": "Summarize this meeting note into decisions + owners:\n\nMaya wants to freeze the schema Friday. Omar will backfill the new `plan_tier` column. Priya raised that billing still reads the old column through a view. We agreed the view stays until 2026-09-01. Omar owns the backfill; Priya owns the view sunset.",
        "assistant": "Decisions: freeze schema Friday; keep the old billing view until 2026-09-01.\nOwners: Omar — `plan_tier` backfill; Priya — view sunset. Maya is driving the freeze.",
    },
    {
        "user": "TL;DR:\n\nThe library switched from recursive descent to a Pratt parser. Error messages now include a source span. Benchmarks: 2.4x faster on 10k-line files. Public AST types did not change.",
        "assistant": "Parser rewrite (Pratt) is 2.4× faster on large files and emits source spans; public AST is unchanged.",
    },
    {
        "user": "Summarize this legal-ish paragraph in plain English:\n\nThe processor shall implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk, including inter alia as appropriate the pseudonymisation and encryption of personal data.",
        "assistant": "The vendor must lock down personal data in a way that matches the risk, including encryption and stripping direct identifiers where that makes sense.",
    },
    {
        "user": "3-bullet recap:\n\nWe evaluated three queues. Redis lists are simplest but lose messages on crash unless AOF is always-on. SQS is durable and cheap at our volume but adds ~20ms. NATS JetStream is fast and durable but we lack ops experience. Recommendation: SQS for the billing pipeline, keep Redis for ephemeral fan-out.",
        "assistant": "- Redis lists are easy but can lose messages without always-on AOF.\n- SQS is durable with ~20ms extra latency — pick it for billing.\n- NATS JetStream is fast and durable but we are not ready to operate it; keep Redis for ephemeral fan-out.",
    },
    {
        "user": "Summarize the experiment:\n\nWe hid the 'Upgrade' button for 10% of free users for 14 days. Conversion to paid fell 0.4pp (p=0.03). Support contacts did not change. We will restore the button.",
        "assistant": "Hiding Upgrade for 10% of free users for two weeks cut paid conversion by 0.4pp (significant) with no support change. Restore the button.",
    },
    {
        "user": "Shorten this onboarding doc intro:\n\nWelcome to the platform. Before you deploy, you must install the CLI, authenticate with SSO, create a namespace, and attach a billing account. Local Docker is optional but recommended for integration tests.",
        "assistant": "Before deploying: install the CLI, SSO-login, create a namespace, and attach billing. Docker is optional and recommended for integration tests.",
    },
    {
        "user": "Summarize this log of a debate:\n\nA: We should shard by tenant. B: Most tenants are tiny; we'll have thousands of empty shards. A: Then hash by tenant and keep a directory. B: Fine, as long as rebalancing is automatic. Decision: hash + directory, auto-rebalance in Q4.",
        "assistant": "Do not give each tenant its own shard. Hash tenants through a directory and auto-rebalance in Q4.",
    },
]

WRITING = [
    {
        "user": "Draft a short status email to leadership: checkout latency is up 18% since Tuesday, cause is a lock on the payments table, fix ships tonight, no data loss.",
        "assistant": "Subject: Checkout latency — cause found, fix tonight\n\nTeam — checkout p95 is up 18% since Tuesday. Root cause is a row lock on the payments table during a new reconciliation job. There has been no data loss and no failed captures; users only see a slower confirm screen. We are shipping a fix tonight that moves reconciliation off the hot path. I will send a closeout note after the deploy.",
    },
    {
        "user": "Write a commit message for: extracted rate limiter from the API process into Redis, added tests, no API contract change.",
        "assistant": "Move API rate limiter into Redis\n\nExtract the in-process token bucket so multiple API replicas share one limit. Behavior and response headers are unchanged. Adds unit tests for the Redis backend and a fallback that fails closed if Redis is down.",
    },
    {
        "user": "Write a 6-bullet postmortem outline for a 37-minute auth outage caused by an expired TLS cert on the IdP.",
        "assistant": "- Summary: 37-minute auth outage; users could not log in or refresh tokens.\n- Impact: 100% of interactive logins; existing access tokens kept working.\n- Cause: IdP TLS certificate expired; no calendar alert fired.\n- Detection: customer tweet, then synthetic login probe.\n- Response: issued a new cert, reloaded the IdP, verified login.\n- Follow-ups: 30/14/7-day cert alerts, staging cert with a 7-day TTL to test the path, add the probe to the paging path.",
    },
    {
        "user": "Write a polite decline to a vendor who wants a 45-minute product demo. We already picked another tool.",
        "assistant": "Hi — thanks for the offer. We have already selected a tool for this and are in implementation, so I am going to pass on a demo. I will reach out if that changes. Appreciate the follow-up.",
    },
    {
        "user": "Draft a README section 'Running tests' for a Python repo that uses pytest, needs Redis, and has a make target.",
        "assistant": "## Running tests\n\nYou need Redis on `localhost:6379` (or set `REDIS_URL`).\n\n```bash\npython -m venv .venv && source .venv/bin/activate\npip install -e '.[dev]'\nmake test          # pytest -q\nmake test-fast     # skips integration marks\n```\n\nCI runs the same `make test` target.",
    },
    {
        "user": "Write an RFC title + problem statement (half page) for replacing cron with a durable workflow engine for invoice generation.",
        "assistant": "# RFC: Durable workflows for invoice generation\n\n## Problem\nInvoices are produced by a cron that fires at 00:10 UTC, fans out one process per tenant, and writes PDFs to object storage. A crashed worker leaves a tenant half-billed; the next night either double-charges or skips them, depending on a brittle `invoices.generated_on` flag. We cannot pause a single tenant, retry a single step, or see progress. Volume is now ~40k tenants; the cron regularly overruns the 2-hour window. We need a workflow with explicit steps (lock period → compute → render → store → notify), per-tenant idempotency, and a human-visible retry.",
    },
    {
        "user": "Write a Slack message asking the design team for a review of a settings page by Thursday. Keep it short.",
        "assistant": "Design — settings page is ready for review: Figma link in the thread. Need eyes by Thursday noon so we can still land it this sprint. Focus on the new 2FA reset flow. Thanks.",
    },
    {
        "user": "Write a user-facing error message for when a file upload exceeds 25 MB. Offer a next step.",
        "assistant": "This file is larger than the 25 MB limit. Compress it or split it, then try again. If you need a higher limit, contact support with the file type and size.",
    },
    {
        "user": "Draft a 4-sentence performance review blurb for an engineer who shipped a query planner fix that cut p95 by 40% and mentored two interns.",
        "assistant": "Alex shipped a query-planner fix that cut p95 latency 40% on the reporting API with no change to the public contract. The work included a careful rollback plan and load-test evidence, which made the review easy. They also mentored two interns through their first production changes. I would gladly staff them on the next storage project.",
    },
    {
        "user": "Write a changelog entry for adding SSO via OIDC. Mention that username/password still works.",
        "assistant": "### SSO (OIDC)\nYou can now sign in with your identity provider. Username and password accounts keep working. Admins enable SSO under Settings → Authentication and paste the issuer, client id, and client secret.",
    },
    {
        "user": "Write a brief design-doc 'non-goals' list for a v1 feature-flag service.",
        "assistant": "Non-goals for v1:\n- Multivariate flags and percentage rollouts beyond on/off + a single percent.\n- Client-side evaluation in mobile apps.\n- A public API for customers.\n- Audit log export.\n- Multi-region active-active.",
    },
    {
        "user": "Turn these notes into a meeting agenda (30 min): metrics review, on-call swap, Q3 hiring, parking lot for the rewrite.",
        "assistant": "Agenda (30 min)\n1. Metrics review — 8 min (open pages, error budget)\n2. On-call swap next week — 5 min\n3. Q3 hiring reqs — 10 min\n4. Parking lot: service rewrite, schedule a separate working session — 5 min\n5. Recap + owners — 2 min",
    },
    {
        "user": "Write a comment to leave on a PR that uses `except Exception: pass` around a payment call. Be direct but kind.",
        "assistant": "This `except Exception: pass` around the payment call will hide refused charges and network errors. Can we catch the specific client error, log the payment id, and fail the request? Happy to pair on a retry policy if the worry is transient 503s.",
    },
    {
        "user": "Write a 3-sentence intro for an internal wiki page about our rate limits.",
        "assistant": "This page is the source of truth for API rate limits. Default is 600 requests per minute per API key, with a burst of 50. If you need more, open a ticket with the endpoint and a 7-day traffic sample — do not raise limits in application code.",
    },
    {
        "user": "Draft a deprecation notice: XML export goes away in 90 days, JSON is the replacement, include a migration link placeholder.",
        "assistant": "XML export is deprecated and will be removed in 90 days. Switch to the JSON export (`GET /v2/export?format=json`). A migration guide is here: <link>. After the cutoff, XML requests will return 410.",
    },
]

QA = [
    {
        "user": "What does the CAP theorem actually claim, in one paragraph?",
        "assistant": "In a partition, a replicated data store must choose between remaining available (accepting writes that may diverge) and remaining consistent (refusing or delaying requests until the partition heals). You can have consistency and availability when the network is healthy; the theorem is about the partition case, not a permanent three-way ban.",
    },
    {
        "user": "Why is `hash(id) % n` a bad way to shard when n changes?",
        "assistant": "Almost every key moves when n changes, because the remainder changes for most ids. Consistent hashing or a lookup directory only remaps a 1/n fraction of keys, which is why those are used for resharding.",
    },
    {
        "user": "What is the difference between at-least-once and exactly-once delivery in a queue?",
        "assistant": "At-least-once means a delivered message may be seen more than once after a retry or crash; consumers must be idempotent. Exactly-once is an end-to-end property (dedupe + atomic side effects), not something the broker can promise alone. Most practical systems are at-least-once plus idempotent handlers.",
    },
    {
        "user": "When should I use a left join vs an inner join?",
        "assistant": "Inner join keeps rows that match on both sides. Left join keeps every row from the left table and fills right-hand columns with NULL when there is no match. Use left join when the left entity should appear even if the optional side is missing (users without orders, tickets without assignees).",
    },
    {
        "user": "What is a TOCTOU bug? Give a filesystem example.",
        "assistant": "Time-of-check to time-of-use: you inspect a condition, then act, and the condition changes in between. Example: `if not os.path.exists(path): open(path, 'w')` — another process can create or replace the file after the check. Use `os.open` with `O_CREAT|O_EXCL` or file locks instead.",
    },
    {
        "user": "Explain Python's GIL in two sentences for a new hire.",
        "assistant": "CPython only lets one thread execute Python bytecode at a time, so CPU-bound threads do not speed up on extra cores. I/O-bound threads still overlap, and processes or native extensions can use multiple cores.",
    },
    {
        "user": "What does HTTP 429 mean and what header should I send back?",
        "assistant": "429 Too Many Requests: the client hit a rate limit. Send `Retry-After` (seconds or HTTP date) and, if you have one, a `RateLimit-*` remaining/reset hint.",
    },
    {
        "user": "Why are floating point equality checks a bad idea for money?",
        "assistant": "Binary floating point cannot represent most decimal cents exactly, so `0.1 + 0.2 == 0.3` is false and sums drift. Store integer minor units (cents) or use a decimal type.",
    },
    {
        "user": "What is the N+1 query problem?",
        "assistant": "You run one query for a list, then one query per row for a related object (N extra queries). Fix with a join, `IN (...)`, or an ORM prefetch. It often shows up as a fast local test and a slow production page.",
    },
    {
        "user": "Difference between authentication and authorization?",
        "assistant": "Authentication answers who the caller is (login, token, mTLS). Authorization answers what that identity may do (roles, policies, scopes). You can be authenticated and still forbidden.",
    },
    {
        "user": "What is a blue/green deploy?",
        "assistant": "You run two production environments. Green is live. You deploy to blue, test it, then switch traffic. Rollback is switching back. Cost is roughly 2× the serving fleet during the cutover.",
    },
    {
        "user": "Why might `SELECT COUNT(*)` be slow on a large Postgres table?",
        "assistant": "Postgres does not keep a running total, so `COUNT(*)` scans visible rows (heap or an index-only scan if a suitable index is all-visible). On a huge table this is I/O heavy. Approximate counts from `pg_class.reltuples` or a summary table are the usual workaround.",
    },
    {
        "user": "What does `mtime` vs `ctime` mean on Unix?",
        "assistant": "`mtime` is last content modification. `ctime` is last inode change (permissions, rename, link count) and also updates when content changes. `ctime` is not 'creation time' on traditional Unix; birth time is a separate field on some filesystems.",
    },
    {
        "user": "Give two reasons a Docker container might see a different time than the host.",
        "assistant": "The container has its own timezone data (`TZ` / `/etc/localtime`) even if the clock is the same UTC instant. On some VM/desktop setups the hypervisor clock can also drift if time sync is off. For timezone bugs, check `TZ` first.",
    },
    {
        "user": "What is idempotency in an HTTP API? Which method is supposed to be idempotent?",
        "assistant": "Repeating the same request leaves the same server state as doing it once. GET, PUT, and DELETE are specified as idempotent; POST is not, unless you add an idempotency key. 'Safe' (no state change) is a stronger property that only GET/HEAD have.",
    },
]

DATA_ANALYSIS = [
    {
        "user": "This CSV is daily signups:\n\ndate,signups\n2026-08-01,40\n2026-08-02,42\n2026-08-03,39\n2026-08-04,80\n2026-08-05,41\n\nWhat stands out? Give the mean without the outlier and the z-ish gap.",
        "assistant": "Aug 4 (80) is the outlier. Mean of the other four days is (40+42+39+41)/4 = 40.5. 80 is +39.5, about 2× the baseline. Check a campaign, a bot, or a double-count on that date before treating it as growth.",
    },
    {
        "user": "Funnel: 10_000 visit, 2_400 start checkout, 1_800 enter payment, 1_620 pay. Where is the drop, and what is completion from visit?",
        "assistant": "Biggest drop is visit → checkout (76% leave). After checkout starts, 1_620/2_400 = 67.5% finish; payment → pay is 90%. Overall conversion 1_620/10_000 = 16.2%. Work the landing-to-checkout step first.",
    },
    {
        "user": "A/B: control 2_000 users, 80 conversions. treatment 2_000 users, 104 conversions. Is this interesting? Just compute rates and the absolute lift.",
        "assistant": "Control 4.00%, treatment 5.20%, absolute lift +1.20pp, relative +30%. Worth a proper significance test, but the point estimate is a real movement, not a 1-conversion wobble.",
    },
    {
        "user": "Table:\n\nregion,revenue,cost\nNA,120,70\nEU,90,60\nAPAC,40,38\n\nMargin % per region and which to inspect.",
        "assistant": "NA 41.7%, EU 33.3%, APAC 5.0%. APAC is nearly break-even — inspect discounting, ads, or support cost there before scaling spend.",
    },
    {
        "user": "Cohort retention week0=1000, week1=420, week2=310, week4=280. Comment in 3 sentences.",
        "assistant": "Week-1 drop to 42% is the main leak. After week 2 the curve flattens (31% → 28% at week 4), so remaining users are relatively sticky. Optimize onboarding and week-1 activation, not late-life engagement.",
    },
    {
        "user": "p50=120ms, p95=180ms, p99=2400ms for an API. What does that shape suggest?",
        "assistant": "The body of traffic is tight (p95 only 60ms above p50) but a small tail is 10× worse. That pattern is usually lock waits, cold cache, GC, or a rare dependency timeout — not a general capacity problem. Trace the p99 samples.",
    },
    {
        "user": "Inventory: SKU A on-hand 12, daily demand 4, lead time 5 days, reorder already placed for 20 arriving in 3 days. Will we stock out?",
        "assistant": "Demand over 3 days ≈ 12, which equals on-hand, so we arrive at zero when the PO lands. Any demand variance or a one-day delay stocks us out. Raise a buffer or expedite; do not wait.",
    },
    {
        "user": "Support tickets: billing 40, login 25, 'slow' 20, other 15. Billing tickets have median handle 18 min, others 8. Where is the time going?",
        "assistant": "Billing is 40% of volume and more than 2× the handle time, so it dominates staff minutes (~720 vs ~480 for everything else). Fix billing UX / self-serve invoices before hiring more generalists.",
    },
    {
        "user": "These conversion rates by device: desktop 3.1% (n=8k), iOS 2.8% (n=6k), Android 1.1% (n=5.5k). What would you check first?",
        "assistant": "Android is less than half of the other two with a large n, so it is unlikely to be noise. Check the Android checkout WebView, payment-sheet bugs, and whether a specific OS version is failing. Do not average this into a single 'mobile' number.",
    },
    {
        "user": "Compute a 7-day moving average for: 3,5,4,8,6,7,9,10. Give the last two MA values.",
        "assistant": "First full window (days 1–7) mean = (3+5+4+8+6+7+9)/7 = 6.0. Next (days 2–8) = (5+4+8+6+7+9+10)/7 = 7.0.",
    },
    {
        "user": "Error budget: 99.9% monthly, 30-day month. We already burned 22 minutes of downtime. How much is left?",
        "assistant": "Monthly budget = 0.001 * 30 * 24 * 60 = 43.2 minutes. Remaining ≈ 21.2 minutes. Another similar incident exhausts the budget.",
    },
    {
        "user": "Prices: 10 units @ $4, 6 @ $7, 4 @ $3. Weighted average price?",
        "assistant": "Total spend = 40 + 42 + 12 = 94. Units = 20. Weighted average = $4.70.",
    },
    {
        "user": "Two models: A RMSE 4.2 on train / 4.4 on test. B RMSE 2.1 train / 6.8 test. Which to ship and why?",
        "assistant": "Ship A. B is overfit (train looks great, test is worse than A). Generalization on unseen rows matters more than training RMSE.",
    },
    {
        "user": "Histogram of session minutes: 0-1: 800, 1-5: 300, 5-20: 120, 20+: 40. What product story is that?",
        "assistant": "Most sessions bounce in under a minute. A smaller group stays 1–5 minutes; a loyal tail stays longer. Improve first-minute value (empty states, time-to-first-success) rather than adding power-user features first.",
    },
    {
        "user": "We ran the same query yesterday (12s) and today (71s). Row counts are similar. Give 4 likely causes.",
        "assistant": "Plan flip (stats / a new index), lock waits, a cold cache or vacuum, and a noisier neighbor on the host. Compare `EXPLAIN (ANALYZE, BUFFERS)` from both days and check `pg_stat_activity` for blockers.",
    },
]

LOG_ANALYSIS = [
    {
        "user": "What failed?\n\n```\n2026-08-16T03:11:02Z app[api] INFO checkout started id=c_91\n2026-08-16T03:11:02Z app[api] INFO authorize card=tok_…\n2026-08-16T03:11:05Z pay[stripe] ERROR 429 too_many_requests req=req_8\n2026-08-16T03:11:05Z app[api] ERROR checkout failed id=c_91 err=card_authorize\n2026-08-16T03:11:05Z app[api] INFO retrying id=c_91 attempt=2\n2026-08-16T03:11:08Z pay[stripe] ERROR 429 too_many_requests\n```",
        "assistant": "Stripe is rate-limiting authorizations (HTTP 429). The checkout itself is fine; retries hit the same limit. Back off, cache tokens, or raise the Stripe rate. Do not keep hammering on attempt 2 with the same cadence.",
    },
    {
        "user": "Diagnose:\n\n```\nOOMKilled restartCount=6\nlast state: reason=OOMKilled exit=137\nlimits.memory=256Mi\nusage.memory=255Mi\njava -Xmx512m -jar app.jar\n```",
        "assistant": "The JVM heap (`-Xmx512m`) is larger than the cgroup limit (256Mi), so the kernel OOM-kills the pod in a loop. Set `-Xmx` well under the limit (e.g. 192m) or raise the limit. Restarts will continue until those numbers agree.",
    },
    {
        "user": "```\nnginx: connect() failed (111: Connection refused) while connecting to upstream 127.0.0.1:3000\napp: listening on 0.0.0.0:3001\n```\nWhy 502s?",
        "assistant": "Nginx proxies to port 3000; the app listens on 3001. Connection refused is a port mismatch, not an app crash. Point the upstream at 3001 (or change the app port).",
    },
    {
        "user": "```\nERROR deadlock detected\nDETAIL: Process 71 waits for ShareLock on transaction 9001; blocked by 70.\nProcess 70 waits for ShareLock on transaction 9002; blocked by 71.\nSTATEMENT: UPDATE accounts SET bal = bal - 10 WHERE id = 2\n```",
        "assistant": "Two transactions lock the same accounts in opposite order and deadlocked. Postgres aborted one. Take row locks in a stable order (e.g. by account id) or use a single debit/credit statement. Retrying the aborted transaction is fine once ordering is fixed.",
    },
    {
        "user": "```\nTLS handshake error: x509: certificate has expired or is not yet valid\nnotAfter=2026-08-01T00:00:00Z\nnow=2026-08-16\n```",
        "assistant": "The cert expired on 2026-08-01, two weeks ago. Renew/replace it and reload the terminator. Check why expiry alerts did not page.",
    },
    {
        "user": "```\ncelery worker\nReceived unregistered task 'billing.emit_invoice'\nThe message has been ignored and discarded\n```",
        "assistant": "A producer enqueued `billing.emit_invoice` but this worker process does not import that task. Deploy the matching code to the worker or stop producing the old name. The message is already dropped — check whether those invoices need a replay.",
    },
    {
        "user": "```\nssh: Permission denied (publickey)\nclient offers: RSA SHA-1\nserver: PubkeyAcceptedAlgorithms ssh-ed25519,rsa-sha2-256,rsa-sha2-512\n```",
        "assistant": "The client is offering an RSA/SHA-1 key; the server only accepts Ed25519 or RSA-SHA2. Upgrade OpenSSH or generate an Ed25519 key. This is an algorithm mismatch, not a wrong username.",
    },
    {
        "user": "```\nERROR: relation \"user\" does not exist at character 15\nQUERY: SELECT * FROM user WHERE id = $1\n```",
        "assistant": "`user` is a reserved Postgres word (the current role). Quote it `\"user\"` or, better, rename the table to `users`. The table may exist under a different name.",
    },
    {
        "user": "```\nk8s: Readiness probe failed: HTTP 500 /readyz\nliveness still 200\ndb ping in readyz: context deadline exceeded\n```",
        "assistant": "The pod is alive but not ready because the readiness check's DB ping timed out. Kubernetes will pull it from the Service. Fix DB latency/connections; do not restart on liveness unless it stays down. A thundering-herd restart would make the DB worse.",
    },
    {
        "user": "```\nFATAL: password authentication failed for user \"app\"\nhost=db-migrator\nssl=off\n```",
        "assistant": "The migrator is sending a bad password (or the role was rotated) and is not using TLS. Check the secret the job mounts vs. the DB role, and require SSL. Do not embed a fallback password in the job spec.",
    },
    {
        "user": "```\n[warn] clock jumped backwards 32s\n[error] lease expired, resigning leader\n[info] starting election\n```",
        "assistant": "NTP (or a VM pause) stepped the clock backward, the leader lease looked expired, and the node resigned. Pin NTP, use a monotonic lease, and avoid large backward steps on consensus hosts.",
    },
    {
        "user": "```\nTypeError: Cannot read properties of undefined (reading 'id')\n    at invoice.js:41\n    at async /checkout\nuser=anon\n```",
        "assistant": "Checkout ran as anonymous and `user` is undefined when `user.id` is read at invoice.js:41. Guard the unauthenticated path or require auth before building an invoice. This is a null deref, not a payments-provider outage.",
    },
    {
        "user": "```\nENOSPC: no space left on device, write /var/lib/docker/overlay2/...\n```",
        "assistant": "The disk (likely Docker's overlay) is full. Prune unused images/containers, check log growth, and expand the volume. New writes, including deploys, will fail until space is freed.",
    },
    {
        "user": "```\nDNS NXDOMAIN for payments.internal\nresolver=10.0.0.10\ncluster: CoreDNS CrashLoop BackOff\n```",
        "assistant": "In-cluster DNS is down (CoreDNS crash looping), so internal names NXDOMAIN. Fix CoreDNS first (config, resource limits, upstream). App retries will not help until DNS answers again.",
    },
    {
        "user": "```\nGET /api/v1/items 200 12ms\nGET /api/v1/items 200 11ms\nGET /api/v1/items 200 940ms\nGET /api/v1/items 200 11ms\n# GC log: Full GC 812ms\n```",
        "assistant": "The 940ms request lines up with an 812ms full GC. Capacity is fine; the JVM is pausing. Tune heap / GC, reduce allocation, or add a second replica so pauses are less visible.",
    },
]

TASK_BANKS = {
    "coding": CODING,
    "summarization": SUMMARIZATION,
    "writing": WRITING,
    "qa": QA,
    "data_analysis": DATA_ANALYSIS,
    "log_analysis": LOG_ANALYSIS,
}
