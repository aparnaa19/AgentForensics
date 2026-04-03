"""
AgentForensics Desktop Launcher
One-click GUI to start the dashboard and begin monitoring.
"""
from __future__ import annotations

import sys
import threading
import tkinter as tk

if getattr(sys, "frozen", False):
    import multiprocessing
    multiprocessing.freeze_support()

BG     = "#0f172a"
CARD   = "#1e293b"
BORDER = "#334155"
BLUE   = "#38bdf8"
GREEN  = "#22c55e"
RED    = "#ef4444"
FG     = "#f1f5f9"
DIM    = "#64748b"


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AgentForensics")
        self.geometry("460x320")
        self.resizable(False, False)
        self.configure(bg=BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._dashboard_thread: threading.Thread | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=28, pady=(28, 0))
        tk.Label(hdr, text="Agent",     font=("Segoe UI", 22, "bold"), bg=BG, fg=BLUE).pack(side="left")
        tk.Label(hdr, text="Forensics", font=("Segoe UI", 22, "bold"), bg=BG, fg=FG).pack(side="left")

        about = tk.Frame(self, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        about.pack(fill="x", padx=28, pady=(16, 0))
        tk.Label(
            about,
            text=(
                "AgentForensics monitors your AI tools in real time and detects "
                "prompt injection attacks. Connect via VS Code extension or "
                "Claude Desktop MCP. All detections are logged locally."
            ),
            font=("Segoe UI", 9), bg=CARD, fg=DIM,
            wraplength=380, justify="left",
        ).pack(padx=14, pady=12)

        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(fill="x", padx=28, pady=(22, 0))

        self._start_btn = tk.Button(
            btn_row, text="▶  Start Monitoring",
            font=("Segoe UI", 11, "bold"),
            bg="#2563eb", fg="white",
            activebackground="#1d4ed8", activeforeground="white",
            relief="flat", bd=0, cursor="hand2", pady=11,
            command=self._start_all,
        )
        self._start_btn.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self._stop_btn = tk.Button(
            btn_row, text="■  Stop Monitoring",
            font=("Segoe UI", 11),
            bg=CARD, fg=DIM,
            activebackground=BORDER, activeforeground=FG,
            relief="flat", bd=0, cursor="hand2", pady=11,
            state="disabled",
            command=self._stop_all,
        )
        self._stop_btn.pack(side="left", fill="both", expand=True, padx=(8, 0))

        self._status_lbl = tk.Label(self, text="Not running", font=("Segoe UI", 9), bg=BG, fg=DIM)
        self._status_lbl.pack(pady=(10, 0))

        self._dash_btn = tk.Button(
            self, text="Open Dashboard",
            font=("Segoe UI", 10, "bold"),
            bg=CARD, fg=DIM,
            activebackground=BORDER, activeforeground=FG,
            relief="flat", bd=0, cursor="hand2", pady=9,
            state="disabled",
            command=self._open_dashboard,
        )
        self._dash_btn.pack(fill="x", padx=28, pady=(10, 0))

    def _start_all(self) -> None:
        self._dashboard_thread = threading.Thread(target=self._run_dashboard, daemon=True)
        self._dashboard_thread.start()
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal", fg=FG)
        self._status_lbl.config(text="● Monitoring active", fg=GREEN)
        self._dash_btn.config(state="normal", fg=FG)
        self.after(2000, self._open_dashboard)

    def _open_dashboard(self) -> None:
        import urllib.request, time, webbrowser
        def _poll():
            for _ in range(30):
                try:
                    urllib.request.urlopen("http://127.0.0.1:7890", timeout=1)
                    self.after(0, _launch)
                    return
                except Exception:
                    time.sleep(0.5)
            self.after(0, lambda: webbrowser.open("http://localhost:7890"))

        def _launch():
            try:
                import webview  # type: ignore
                webview.create_window("AgentForensics — Dashboard", "http://127.0.0.1:7890", width=1280, height=800, resizable=True)
                self.withdraw()
                webview.start()
                self.deiconify()
            except Exception:
                import webbrowser
                webbrowser.open("http://localhost:7890")

        threading.Thread(target=_poll, daemon=True).start()

    def _stop_all(self) -> None:
        import tkinter.messagebox as mb
        if mb.askyesno("Stop AgentForensics", "Stop monitoring and close the app?"):
            self.destroy()
            sys.exit(0)

    def _on_close(self) -> None:
        self.destroy()
        sys.exit(0)

    def _run_dashboard(self) -> None:
        import traceback
        from pathlib import Path
        try:
            from agentforensics.dashboard.server import run_server
            run_server(host="127.0.0.1", port=7890)
        except Exception:
            log = Path.home() / ".agentforensics" / "launcher_error.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(traceback.format_exc(), encoding="utf-8")
            self.after(0, lambda: self._status_lbl.config(text="⚠  Dashboard failed to start", fg=RED))


if __name__ == "__main__":
    App().mainloop()
