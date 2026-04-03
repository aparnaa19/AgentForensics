# AgentForensics - User Guide

This guide walks through installation and every usage mode step by step.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [Dashboard](#3-dashboard)
4. [Use Case 1 - Code Wrapper (OpenAI / Anthropic)](#4-use-case-1--code-wrapper)
5. [Use Case 2 - Claude Desktop (MCP)](#5-use-case-2--claude-desktop-mcp)
6. [Use Case 3 - VS Code Extension](#6-use-case-3--vs-code-extension)
7. [Use Case 4 - LangChain](#7-use-case-4--langchain)
8. [Use Case 5 - AutoGen](#8-use-case-5--autogen)
9. [Understanding the Dashboard](#9-understanding-the-dashboard)
10. [Campaigns & Fingerprinting](#10-campaigns--fingerprinting)
11. [False Alarms](#11-false-alarms)
12. [Benchmark](#12-benchmark)
13. [Environment Variables](#13-environment-variables)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | [python.org](https://www.python.org/downloads/) |
| Node.js | 18+ | Only for VS Code extension build |
| VS Code | Any | Only if using the extension |
| Claude Desktop | Any | Only if using MCP integration |

---

## 2. Installation

### One-command install (recommended)

```bash
git clone https://github.com/aparnaa19/agentforensics
cd agentforensics
python install.py
```

This single command:
- Installs all Python dependencies
- Downloads the ML detection model (~250 MB)
- Builds the VS Code extension
- Creates a desktop app (Windows only)
- Configures Claude Desktop MCP automatically

### Manual install (Python only)

```bash
pip install agentforensics[dashboard]
```

### Download ML model separately

If you skipped the model during install:

```bash
python scripts/download_model.py
```

---

## 3. Dashboard

The dashboard is the central hub - all detections appear here regardless of which integration you use.

### Launch

**Option A - Desktop app (Windows)**
- Search "AgentForensics" in the Windows Start Menu, or
- Double-click `dist/AgentForensics.exe`

**Option B - Command line (all platforms)**
```bash
python -m agentforensics.cli dashboard
```

**Option C - Python**
```python
from agentforensics.dashboard import start
start()
```

The dashboard opens at **http://localhost:7890**

> To change the port: `python -m agentforensics.cli dashboard --port 8080`

---

## 4. Use Case 1 - Code Wrapper

Monitor any Python app that uses the OpenAI or Anthropic SDK. No other code changes needed.

### OpenAI

```python
from agentforensics import trace
from openai import OpenAI

# Wrap the client - one line change
client = trace(OpenAI(api_key="sk-..."))

# Use exactly as normal
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarise this document: ..."},
    ],
)
print(response.choices[0].message.content)
```

### Anthropic

```python
from agentforensics import trace
from anthropic import Anthropic

client = trace(Anthropic(api_key="sk-ant-..."))

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Summarise this email: ..."}],
)
```

### What happens

- Every API call is recorded as a session turn
- Injection signals are scored in real time
- Results appear in the dashboard immediately
- The `session_id` is printed to console on first call

### Access session ID

```python
tracer = trace(client)
print(tracer.session_id)  # e.g. "abc123-..."
```

### Get notified in code

```python
tracer = trace(client)

@tracer.on_injection
def handle(signal):
    print(f"Injection detected! Score: {signal.score:.2f}")
    print(f"Evidence: {signal.evidence_snippet}")
```

---

## 5. Use Case 2 - Claude Desktop (MCP)

AgentForensics plugs directly into Claude Desktop as an MCP server. Every conversation Claude has is monitored - no code changes, no API keys needed.

### Setup

Run `python install.py` - it configures MCP automatically.

If you want to configure manually, add this to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "agentforensics": {
      "command": "python",
      "args": ["/path/to/agentforensics/agentforensics/mcp_server.py"]
    }
  }
}
```

Config file locations:
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`

### Usage

1. Restart Claude Desktop after configuration
2. Start the AgentForensics dashboard: `python -m agentforensics.cli dashboard`
3. Chat with Claude as normal
4. Open the dashboard at http://localhost:7890 - injections appear in real time

---

## 6. Use Case 3 - VS Code Extension

The extension automatically scans every file you open, edit, or paste in VS Code. Injections are shown as red squiggles and in the Problems panel.

### Install

1. Build the extension (done automatically by `install.py`), or find the `.vsix` file in `agentforensics-vscode/`
2. In VS Code: `Extensions (Ctrl+Shift+X)` → `⋯` menu → `Install from VSIX`
3. Select the `.vsix` file

### Usage

1. Start the AgentForensics dashboard first (the extension connects to it)
2. Open any file in VS Code
3. If the file contains injection patterns, red squiggles appear on the suspicious line
4. Open the Problems panel (`Ctrl+Shift+M`) to see all detections
5. The status bar at the bottom shows: `AgentForensics: CLEAN` / `SUSPICIOUS` / `COMPROMISED`

### What gets scanned

- Any file you open
- Any file you save
- Content you paste (if it spans multiple lines)
- Right-click a file → `Scan with AgentForensics`

### Settings

Go to VS Code Settings → search "AgentForensics":

| Setting | Default | Description |
|---------|---------|-------------|
| `agentforensics.dashboardUrl` | `http://localhost:7890` | Dashboard address |
| `agentforensics.scanOnOpen` | `true` | Scan files when opened |
| `agentforensics.scanOnPaste` | `true` | Scan multi-line pastes |

---

## 7. Use Case 4 - LangChain

Monitor LangChain agent runs with a callback handler.

### Install

```bash
pip install langchain langchain-openai
```

### Usage

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from agentforensics.integrations.langchain_handler import AgentForensicsHandler

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Add the handler
result = agent_executor.invoke(
    {"input": user_input},
    config={"callbacks": [AgentForensicsHandler(session_id="my-run")]},
)
```

Every LLM call and tool result is recorded. Injections embedded in tool outputs are detected automatically.

---

## 8. Use Case 5 - AutoGen

Monitor AutoGen multi-agent conversations.

### Install

```bash
pip install autogen
```

### Usage

```python
from autogen import ConversableAgent
from agentforensics.integrations.autogen_tracer import patch_autogen

assistant = ConversableAgent(
    name="assistant",
    llm_config={"model": "gpt-4o-mini", "api_key": "sk-..."},
)

# Patch the agent - one line
tracer = patch_autogen(assistant)
print(f"Monitoring session: {tracer.session_id}")

# Use the agent normally
assistant.generate_reply(
    messages=[{"role": "user", "content": "Analyse this webpage content: ..."}]
)
```

---

## 9. Understanding the Dashboard

### Session list (left sidebar)

Each entry is one monitored conversation. The badge shows:

| Badge | Meaning |
|-------|---------|
| `CLEAN` (green) | No injection signals detected |
| `SUSPICIOUS` (yellow) | Low-confidence injection signals found |
| `COMPROMISED` (red) | High-confidence injection attack detected |

Click any session to open the full forensic report.

### Forensic report

The report shows:
- **Session metadata** - model, source, start time, total turns
- **Timeline** - each turn with its score and verdict
- **Evidence** - the exact text that triggered each rule, with the rule name and plain-English description
- **Signal breakdown** - which detection stages fired (heuristics, ML, instruction boundary, semantic)

### Downloading a report

Open a session → click **Download PDF** in the top toolbar. The full report prints as a PDF with all evidence visible.

### Live Monitor

Click **Live Monitor: ON** in the top bar to watch injections arrive in real time via Server-Sent Events. Each alert shows the session, score, verdict, and evidence snippet.

---

## 10. Campaigns & Fingerprinting

AgentForensics groups similar injection attacks across sessions into **campaigns** - useful for identifying repeated or coordinated attacks.

### Open Campaigns panel

Click **Campaigns** in the top navigation bar.

### What you see

| Column | Description |
|--------|-------------|
| Fingerprint ID | Unique ID for this injection pattern |
| Hits | How many times this pattern was detected |
| Sessions | How many sessions were affected |
| Campaign Name | Your label for this attack group |
| First Seen / Last Seen | Timeline of the attack |
| Snippet | Preview of the injection content |

### Naming a campaign

Type a name in the Campaign Name field → click **Save**. Use this to track known attack groups (e.g. "Competitor scraping attack", "Supply chain injection").

### Sorting

Use the sort dropdown to order by:
- **Newest first** - most recently seen attacks at top
- **Oldest first** - chronological order
- **Most hits** - most frequent attacks first

### Deduplication

If you see many similar fingerprints (e.g. variations of the same injection), click **Deduplicate** to merge them into one record.

---

## 11. False Alarms

If a session is incorrectly flagged, mark it as a false alarm to suppress future alerts for that content.

### Mark as false alarm

In the session list → click **False Alarm** below the session entry.

### Review false alarms

Click **False Alarms** in the top navigation bar to see all marked content.

### Unmark a false alarm

In the False Alarms panel → click **Unmark** next to the entry. Future scans of that content will be evaluated normally again.

---

## 12. Benchmark

### ARPIbench (external dataset)

Tests against 7,560 real-world indirect injection payloads:

```bash
pip install datasets
python benchtest.py           # first 2000 rows, heuristics only
python benchtest.py --ml      # with ML classifier (slower)
python benchtest.py --all     # full 7560 rows
python benchtest.py --verbose # show result per row
```

### Built-in benchmark

Tests each detection rule individually with hand-crafted samples. No internet required:

```bash
python benchmark.py
python benchmark.py --verbose  # show score for every sample
python benchmark.py --ml       # include ML stage
```

Results are saved to `benchtest_results.json` / `benchmark_results.json`.

---

## 13. Environment Variables

Set these before launching the dashboard or running your app:

```bash
# Minimum score to flag an injection (default: 0.25)
export AF_INJECT_THRESHOLD=0.25

# Cosine similarity for grouping fingerprints (default: 0.70)
export AF_FP_THRESHOLD=0.70

# Disable ML stage for faster startup (default: false)
export AF_DISABLE_ML=true

# Custom model path (default: agentforensics/model/)
export AF_MODEL_PATH=/path/to/model

# Custom fingerprint database path
export AF_FP_DB=~/.agentforensics/fingerprints.db

# Sliding window size for multi-turn detection (default: 3)
export AF_WINDOW_SIZE=3
```

**Windows (PowerShell):**
```powershell
$env:AF_DISABLE_ML = "true"
python -m agentforensics.cli dashboard
```

---

## 14. Troubleshooting

### Dashboard not starting

```
Error: address already in use
```
Another process is using port 7890. Either stop it or use a different port:
```bash
python -m agentforensics.cli dashboard --port 8080
```

### ML model not loading

The tool prints a warning and falls back to heuristics-only mode. Fix:
```bash
python scripts/download_model.py
```

### VS Code extension not connecting

- Make sure the dashboard is running before opening files in VS Code
- Check the dashboard URL in VS Code settings matches where it's running
- Reload VS Code window: `Ctrl+Shift+P` → `Developer: Reload Window`

### Session shows CLEAN but injection was present

- The injection may be below the detection threshold. Lower it:
  ```bash
  export AF_INJECT_THRESHOLD=0.15
  ```
- Check if the content was previously marked as a False Alarm

### exe not opening from Start Menu

The shortcut points to `%LOCALAPPDATA%\AgentForensics\AgentForensics.exe`. Re-run `python install.py` to recreate it, or launch directly from `dist\AgentForensics.exe`.

### Port already in use after crash

```bash
# Windows
netstat -ano | findstr :7890
taskkill /PID <PID> /F
```

---

## Getting Help

- Open an issue: [github.com/aparnaa19/agentforensics/issues](https://github.com/aparnaa19/agentforensics/issues)
- Check the README: [README.md](README.md)
