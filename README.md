<meta name="google-site-verification" content="gmQrTUrnlvo9f1rSMR9Hyeqt_fuBNX224kG74q9edCE" />

# AgentForensics

**Real-time prompt injection detection and forensics for LLM agents.**

[![CI](https://github.com/aparnaa19/agentforensics/actions/workflows/ci.yml/badge.svg)](https://github.com/aparnaa19/agentforensics/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![ARPIbench](https://img.shields.io/badge/ARPIbench-100%25%20detection-brightgreen)](benchtest_results.json)

AgentForensics is an open-source security framework that monitors LLM agent sessions in real time, detects prompt injection attacks across multiple sources, fingerprints attack campaigns, and presents forensic evidence in a dashboard - all with **zero changes to your existing agent code**.


📖 **[Full User Guide](USERGUIDE.md)** - installation, all use cases, dashboard walkthrough, troubleshooting

---

## Why AgentForensics?

As LLM agents increasingly read emails, browse websites, process documents, and execute code, they are exposed to a new attack surface: **indirect prompt injection** - malicious instructions embedded in external content that hijack the agent's behaviour.

Existing tools focus on input filtering at the user layer. AgentForensics monitors the **entire agent session** including tool outputs, retrieved documents, and API responses - where the real attacks happen.

---

## Benchmark Results

Evaluated against **ARPIbench** ([alexcbecker/ARPIbench](https://huggingface.co/datasets/alexcbecker/ARPIbench)), an academic benchmark of real-world indirect prompt injection payloads across web, email, and document scenarios:

| Metric | Result |
|--------|--------|
| Detection rate | **100%** (2000/2000 payloads) |
| False positive rate | **0%** (7 benign samples) |
| Attack types covered | naive, completion, ignore, urgent_request, helpful_assistant, multi-turn |
| Scenarios | web, email, local document |

```bash
python benchtest.py --ml   # reproduce results
```

---

## Features

- **Five-stage detection pipeline** - heuristics, ML classifier, instruction boundary analysis, semantic drift, and sliding-window multi-turn detection
- **Live forensic dashboard** - session timeline, per-turn evidence, real-time SSE alerts
- **Attack fingerprinting** - clusters repeated injection patterns across sessions into named campaigns
- **False alarm management** - mark safe content once, suppress forever
- **VS Code extension** - inline squiggles, Problems panel, auto-scan on open/paste
- **Claude Desktop MCP** - plug directly into Claude Desktop, zero config
- **Code wrapper** - one import for OpenAI / Anthropic SDK apps
- **LangChain & AutoGen** - native callback/tracer integrations

---

## How It Works

```mermaid
flowchart TD
    A([External Content\nweb page · email · document · tool result]) --> B

    B[Stage 1 - Heuristics · less than 1ms\n8 regex rule groups · H01 to H08]
    B --> C[Stage 2 - ML Classifier · ~50ms\nFine-tuned DistilBERT on injection patterns]
    C --> D[Stage 3 - Instruction Boundary · less than 1ms\n10 boundary pattern groups · IB01 to IB10]
    D --> E[Stage 4 - Semantic Drift · ~30ms\nLLM response topic vs original query]
    E --> F[Stage 5 - Sliding Window · less than 1ms\nScore aggregation across last N turns]

    F --> G{Verdict}

    G -->|score below 0.25| H([CLEAN])
    G -->|score 0.25 to 0.75| I([SUSPICIOUS])
    G -->|score above 0.75| J([COMPROMISED])

    I --> K[Store · Fingerprint · Alert · Dashboard]
    J --> K

    style H fill:#2d6a2d,color:#ffffff,stroke:#2d6a2d
    style I fill:#a67c00,color:#ffffff,stroke:#a67c00
    style J fill:#8b1a1a,color:#ffffff,stroke:#8b1a1a
```

### Detection Rules

| Rule | Pattern | Weight |
|------|---------|--------|
| H01 | `ignore previous/all/prior instructions` | 0.35 |
| H02 | `you are now / your new role/persona` | 0.30 |
| H03 | `do not tell/inform the user` | 0.25 |
| H04 | Token injection markers `[[SYSTEM]]`, `[INST]`, `<\|im_start\|>` | 0.15 |
| H05 | Sudden language/script switch (evasion) | 0.20 |
| H06 | Imperative commands in tool output (send, forward, upload, exfiltrate) | 0.20 |
| H07 | Reference to system prompt / context window | 0.25 |
| H08 | Exfiltration setup: send/forward near secret/key/password | 0.40 |
| IB01–IB10 | Instruction boundary patterns (context wipe, persona hijack, task redefinition, etc.) | 0.20–0.35 |

---

## Installation

### One-command install (recommended)

```bash
git clone https://github.com/YOUR_USERNAME/agentforensics
cd agentforensics
python install.py
```

This installs Python dependencies, builds the VS Code extension, creates a desktop app (Windows), and configures Claude Desktop MCP automatically.

### Manual

```bash
pip install agentforensics[dashboard]

# Launch dashboard
python -m agentforensics.cli dashboard
# → http://localhost:7890
```

**Requirements:** Python 3.10+, Node.js 18+ (for VS Code extension build)

---

## Usage

### 1. Claude Desktop (MCP)

Run `python install.py` once. AgentForensics is auto-configured as an MCP server. Restart Claude Desktop - monitoring starts immediately, no code changes required.

### 2. Code Wrapper - OpenAI / Anthropic

Drop-in replacement. One import, no other changes:

```python
from agentforensics import trace
from openai import OpenAI

client = trace(OpenAI(api_key="sk-..."))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarise this webpage: ..."}],
)
# Injection signals captured automatically → visible in dashboard
```

### 3. VS Code Extension

Install the `.vsix` from `agentforensics-vscode/`:

```
Extensions → ⋯ → Install from VSIX
```

Every file you open, edit, or paste is automatically scanned. Injections appear as red squiggles and in the Problems panel (`Ctrl+Shift+M`).

### 4. LangChain

```python
from agentforensics.integrations.langchain_handler import AgentForensicsHandler

agent_executor.invoke(
    {"input": user_input},
    config={"callbacks": [AgentForensicsHandler(session_id="my-session")]},
)
```

### 5. AutoGen

```python
from agentforensics.integrations.autogen_tracer import patch_autogen
patch_autogen()
# All subsequent AutoGen conversations are monitored
```

---

## Dashboard

Launch with the desktop app and click go to dashboard or `python -m agentforensics.cli dashboard`

| View | Description |
|------|-------------|
| **Sessions** | All monitored agent sessions with verdict (Clean / Suspicious / Compromised) |
| **Live Monitor** | Real-time injection alerts via Server-Sent Events |
| **Campaigns** | Fingerprinted attack patterns grouped by similarity across sessions |
| **False Alarms** | Approved safe content - permanently excluded from future alerts |

---

## Benchmarking

### ARPIbench (recommended)

Tests against 7,560 real-world indirect injection payloads:

```bash
pip install datasets
python benchtest.py          # first 2000 rows
python benchtest.py --all    # full dataset
python benchtest.py --ml     # with ML stage enabled
```

### Built-in benchmark

Tests per-rule recall and false positive rate with no internet required:

```bash
python benchmark.py
python benchmark.py --verbose
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AF_INJECT_THRESHOLD` | `0.25` | Minimum combined score to flag a signal |
| `AF_FP_THRESHOLD` | `0.70` | Cosine similarity threshold for fingerprint grouping |
| `AF_DISABLE_ML` | `false` | Set `true` to skip ML stage (faster startup, no GPU needed) |
| `AF_MODEL_PATH` | `agentforensics/model/` | Path to fine-tuned DistilBERT weights |
| `AF_FP_DB` | `~/.agentforensics/fingerprints.db` | SQLite fingerprint database path |
| `AF_WINDOW_SIZE` | `3` | Number of turns for sliding-window multi-turn detection |

---

## Project Structure

```
agentforensics/
├── classifier.py       # Five-stage detection pipeline
├── fingerprinter.py    # Attack fingerprinting & campaign clustering
├── alerting.py         # Alert routing and SSE broadcast
├── reporter.py         # HTML forensic report generation
├── semantic.py         # Instruction boundary + semantic similarity
├── tracer.py           # OpenAI/Anthropic SDK wrapper (trace())
├── store.py            # SQLite session/signal storage
├── mcp_server.py       # Claude Desktop MCP integration
├── model/              # Fine-tuned DistilBERT weights (gitignored)
├── dashboard/
│   ├── server.py       # FastAPI backend + SSE
│   └── ui.html         # Single-file dashboard UI
└── integrations/
    ├── langchain_handler.py
    └── autogen_tracer.py

agentforensics-vscode/  # VS Code extension (TypeScript)
benchmark.py            # Built-in per-rule benchmark (no internet)
benchtest.py            # ARPIbench external benchmark
install.py              # One-command installer
launcher.py             # Desktop app launcher (tkinter + pywebview)
```

---

## Contributing

Contributions are welcome. Areas of particular interest:

- Additional heuristic rules for emerging injection techniques
- New framework integrations (CrewAI, LlamaIndex, Haystack)
- Improved ML model training data
- Additional benchmark evaluations


---

## Citation

If you use AgentForensics in your research, please cite:

```bibtex
@software{agentforensics2026,
  author    = Aparnaa,
  title     = {AgentForensics: Real-Time Prompt Injection Detection and Forensics for LLM Agents},
  year      = {2026},
  url       = {https://github.com/aparnaa19/agentforensics},
  license   = {MIT}
}
```

---

## License

MIT - see [LICENSE](LICENSE) for details.
