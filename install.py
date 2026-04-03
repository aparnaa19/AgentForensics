"""
AgentForensics Installer
Run this once to set up everything:

    python install.py

What it does:
1. Checks Python version (3.10+)
2. Installs Python dependencies
3. Installs Node.js dependencies for the VS Code extension
4. Builds the VS Code .vsix extension
5. Downloads the ML detection model (~250 MB, fmops/distilbert-prompt-injection)
6. Builds the AgentForensics desktop app (Windows only)
7. Configures Claude Desktop MCP (if Claude Desktop is installed)
8. Prints a getting-started summary
"""
from __future__ import annotations

import io
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# Helpers


BOLD  = "\033[1m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
CYAN  = "\033[96m"
RESET = "\033[0m"

def title(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 50}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 50}{RESET}")

def ok(text: str)   -> None: print(f"  {GREEN}✓{RESET}  {text}")
def warn(text: str) -> None: print(f"  {YELLOW}⚠{RESET}  {text}")
def err(text: str)  -> None: print(f"  {RED}✗{RESET}  {text}")
def info(text: str) -> None: print(f"  {CYAN}→{RESET}  {text}")

def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> bool:
    """Run a command, stream output, return success."""
    info("Running: " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        if check:
            err(f"Command failed: {' '.join(cmd)}")
        return False
    return True

SYSTEM  = platform.system()   # "Windows", "Darwin", "Linux"
HERE    = Path(__file__).resolve().parent
VSCODE_DIR = HERE / "agentforensics-vscode"


# Step 1 - Python version check


title("Step 1 - Checking Python version")

if sys.version_info < (3, 10):
    err(f"Python 3.10+ required. You have {sys.version}")
    sys.exit(1)

ok(f"Python {sys.version.split()[0]}")


# Step 2 - Install Python dependencies


title("Step 2 - Installing Python dependencies")

pip = [sys.executable, "-m", "pip", "install", "--upgrade"]

if not run(pip + ["-e", '.[dashboard]'], cwd=HERE):
    sys.exit(1)
ok("Core + dashboard dependencies installed")

# pywebview for native dashboard window
if not run(pip + ["pywebview"], cwd=HERE, check=False):
    warn("pywebview not installed - dashboard will open in browser instead of a window")
else:
    ok("pywebview installed")


# Step 3 - VS Code extension: npm install


title("Step 3 - Building VS Code extension")

npm = shutil.which("npm")
if not npm:
    warn("npm not found - skipping VS Code extension build.")
    warn("Install Node.js from https://nodejs.org and re-run this script.")
else:
    if not run([npm, "install"], cwd=VSCODE_DIR):
        warn("npm install failed - extension not built.")
    elif not run([npm, "run", "compile"], cwd=VSCODE_DIR):
        warn("TypeScript compile failed - extension not built.")
    else:
        # Build .vsix
        npx = shutil.which("npx")
        if npx and run([npx, "vsce", "package", "--no-dependencies"], cwd=VSCODE_DIR, check=False):
            vsix = next(VSCODE_DIR.glob("*.vsix"), None)
            if vsix:
                ok(f"VS Code extension built: {vsix.name}")
                info("Install it in VS Code: Extensions → ⋯ → Install from VSIX")
            else:
                warn("vsce package ran but no .vsix found.")
        else:
            warn("vsce not available - run: npm install -g @vscode/vsce")


# Step 4 - Build desktop app (Windows only)


title("Step 4 - Building desktop app")

if SYSTEM != "Windows":
    warn("Desktop app (.exe) is Windows only - skipping on " + SYSTEM)
else:
    # Install PyInstaller if missing
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--version"],
        capture_output=True
    )
    if result.returncode != 0:
        info("Installing PyInstaller...")
        run(pip + ["pyinstaller"], check=False)

    spec = HERE / "agentforensics_launcher.spec"
    if not spec.exists():
        warn("agentforensics_launcher.spec not found - skipping exe build.")
    else:
        # Always use 'python -m PyInstaller' so PATH doesn't matter
        if run([sys.executable, "-m", "PyInstaller", str(spec)], cwd=HERE):
            exe = HERE / "dist" / "AgentForensics.exe"
            if exe.exists():
                ok(f"Desktop app built: dist/AgentForensics.exe")
                info(f"Location: {exe}")
                # Copy exe to %LOCALAPPDATA%\AgentForensics\ (stable C: drive location)
                # then create a Start Menu shortcut pointing there.
                # Pointing a shortcut directly at A:\ or other removable drives causes
                # Windows to silently fail on launch from the Start Menu.
                try:
                    local_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "AgentForensics"
                    local_dir.mkdir(parents=True, exist_ok=True)
                    local_exe = local_dir / "AgentForensics.exe"
                    shutil.copy2(exe, local_exe)
                    ok(f"Copied exe to {local_exe}")

                    start_menu = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
                    shortcut   = start_menu / "AgentForensics.lnk"
                    ps_script  = (
                        f'$s=(New-Object -COM WScript.Shell).CreateShortcut("{shortcut}");'
                        f'$s.TargetPath="{local_exe}";'
                        f'$s.WorkingDirectory="{local_dir}";'
                        f'$s.Description="AgentForensics - Prompt Injection Monitor";'
                        f'$s.Save()'
                    )
                    result = subprocess.run(
                        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                        capture_output=True,
                    )
                    if result.returncode == 0 and shortcut.exists():
                        ok("Start Menu shortcut created - search 'AgentForensics' in Windows")
                    else:
                        warn("Start Menu shortcut could not be created (PowerShell error)")
                except Exception as e:
                    warn(f"Start Menu shortcut skipped: {e}")
            else:
                warn("Build completed but exe not found in dist/")
        else:
            warn("exe build failed - you can still use: python -m agentforensics.cli dashboard")


# Step 5 - Download ML model


title("Step 5 - Downloading ML detection model")

model_dir = HERE / "agentforensics" / "model"
model_file = model_dir / "model.safetensors"

if model_file.exists():
    ok("ML model already present - skipping download")
else:
    info("Downloading fmops/distilbert-prompt-injection from HuggingFace (~250 MB)...")
    try:
        result = subprocess.run(
            [sys.executable, str(HERE / "scripts" / "download_model.py")],
            cwd=HERE,
            capture_output=False,
        )
        if result.returncode == 0 and model_file.exists():
            ok("ML model downloaded and ready")
        else:
            warn("ML model download failed - tool will run in heuristics-only mode")
            warn("Retry manually: python scripts/download_model.py")
    except Exception as e:
        warn(f"ML model download skipped: {e}")
        warn("Retry manually: python scripts/download_model.py")


# Step 6 - Configure Claude Desktop MCP


title("Step 6 - Configuring Claude Desktop MCP")  # noqa

def find_claude_config() -> Path | None:
    """Find the Claude Desktop config file across platforms."""
    if SYSTEM == "Windows":
        candidates = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Packages",
        ]
        # Microsoft Store install
        packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Packages"
        if packages.exists():
            for p in packages.glob("Claude_*"):
                cfg = p / "LocalCache" / "Roaming" / "Claude" / "claude_desktop_config.json"
                if cfg.parent.exists():
                    return cfg
        # Direct install
        appdata = Path(os.environ.get("APPDATA", ""))
        cfg = appdata / "Claude" / "claude_desktop_config.json"
        if cfg.parent.exists():
            return cfg

    elif SYSTEM == "Darwin":
        cfg = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        if cfg.parent.exists():
            return cfg

    elif SYSTEM == "Linux":
        cfg = Path.home() / ".config" / "Claude" / "claude_desktop_config.json"
        if cfg.parent.exists():
            return cfg

    return None

claude_cfg_path = find_claude_config()

if not claude_cfg_path:
    warn("Claude Desktop not found - skipping MCP configuration.")
    warn("If you install Claude Desktop later, add the MCP config manually.")
else:
    info(f"Found Claude Desktop config: {claude_cfg_path}")

    mcp_server = str(HERE / "agentforensics" / "mcp_server.py")

    new_server = {
        "command": sys.executable,
        "args":    [mcp_server],
        "cwd":     str(HERE),
        "env":     {"AF_DISABLE_ML": "true"},
    }

    # Read existing config
    cfg: dict = {}
    if claude_cfg_path.exists():
        try:
            cfg = json.loads(claude_cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}

    cfg.setdefault("mcpServers", {})
    cfg["mcpServers"]["agentforensics"] = new_server

    claude_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    claude_cfg_path.write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )
    ok("Claude Desktop MCP configured")
    warn("Restart Claude Desktop for the MCP changes to take effect.")


# Done - print summary


title("Installation Complete!")

print(f"""
{BOLD}How to use AgentForensics:{RESET}

{BOLD}1. VS Code Extension{RESET}
   Install the .vsix:  Extensions → ⋯ → Install from VSIX
   → {VSCODE_DIR / "agentforensics-0.1.0.vsix"}

{BOLD}2. Desktop App (Windows){RESET}
   Double-click:  {HERE / "dist" / "AgentForensics.exe"}
   Click "Start Monitoring" - dashboard opens automatically.

{BOLD}3. Claude Desktop (MCP){RESET}
   Restart Claude Desktop - AgentForensics is already configured.

{BOLD}4. Code Wrapper (Python apps){RESET}
   from agentforensics import trace
   from openai import OpenAI
   client = trace(OpenAI(api_key="..."))

{BOLD}Dashboard:{RESET}  http://localhost:7890
{BOLD}Docs:{RESET}       README.md
""")
