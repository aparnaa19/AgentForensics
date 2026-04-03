import * as vscode from 'vscode';
import * as http from 'http';
import { ChildProcess } from 'child_process';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let dashboardProcess: ChildProcess | undefined;
let sseRequest:      http.ClientRequest | undefined;
let statusBarItem:   vscode.StatusBarItem;
let sidebarProvider: AgentForensicsSidebarProvider;
let diagnostics:     vscode.DiagnosticCollection;

// Debounce timer for paste / Copilot scans
let _pasteDebounce: ReturnType<typeof setTimeout> | undefined;

// Language IDs we skip (binary / compiled / media)
const SKIP_LANGS = new Set([
  'binary', 'image', 'pdf', 'notebook',
  'git-commit', 'git-rebase',
]);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getConfig() {
  const cfg = vscode.workspace.getConfiguration('agentforensics');
  return {
    dashboardPort: cfg.get<number>('dashboardPort', 7890),
    autoStart:     cfg.get<boolean>('autoStart',    false),
    scanOnOpen:    cfg.get<boolean>('scanOnOpen',   true),
    scanOnPaste:   cfg.get<boolean>('scanOnPaste',  true),
  };
}

function pythonCmd(): string {
  return process.platform === 'win32' ? 'python' : 'python3';
}

function apiBase(): string {
  return `http://127.0.0.1:${getConfig().dashboardPort}`;
}

function fetchJson<T>(path: string): Promise<T> {
  return new Promise((resolve, reject) => {
    http.get(`${apiBase()}${path}`, (res: http.IncomingMessage) => {
      let data = '';
      res.on('data', (chunk: Buffer) => data += chunk.toString());
      res.on('end', () => {
        try { resolve(JSON.parse(data)); }
        catch (e) { reject(e); }
      });
    }).on('error', reject);
  });
}

function postJson<T>(path: string, body: object): Promise<T> {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify(body);
    const req = http.request(
      {
        hostname: '127.0.0.1',
        port:     getConfig().dashboardPort,
        path,
        method:   'POST',
        headers: {
          'Content-Type':   'application/json',
          'Content-Length': Buffer.byteLength(payload),
        },
      },
      (res: http.IncomingMessage) => {
        let data = '';
        res.on('data', (chunk: Buffer) => data += chunk.toString());
        res.on('end', () => {
          try { resolve(JSON.parse(data)); }
          catch (e) { reject(e); }
        });
      }
    );
    req.on('error', reject);
    req.write(payload);
    req.end();
  });
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

function updateStatusBar(verdict: string): void {
  switch (verdict) {
    case 'compromised':
      statusBarItem.text            = '$(error) AF: Injection Detected';
      statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
      statusBarItem.tooltip         = 'AgentForensics: injection detected in active file';
      break;
    case 'suspicious':
      statusBarItem.text            = '$(warning) AF: Suspicious';
      statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
      statusBarItem.tooltip         = 'AgentForensics: suspicious content in active file';
      break;
    case 'clean':
      statusBarItem.text            = '$(shield) AF: Clean';
      statusBarItem.backgroundColor = undefined;
      statusBarItem.tooltip         = 'AgentForensics: no injection detected';
      break;
    default:
      statusBarItem.text            = '$(shield) AgentForensics';
      statusBarItem.backgroundColor = undefined;
      statusBarItem.tooltip         = 'AgentForensics — click to open dashboard';
  }
}

// ---------------------------------------------------------------------------
// Document scanner
// Calls POST /api/scan, then creates diagnostics on the suspicious line.
// ---------------------------------------------------------------------------

function isScannable(doc: vscode.TextDocument): boolean {
  if (doc.uri.scheme !== 'file' && doc.uri.scheme !== 'untitled') { return false; }
  if (doc.getText().length > 500_000) { return false; }
  return !SKIP_LANGS.has(doc.languageId);
}

async function scanDocument(doc: vscode.TextDocument): Promise<void> {
  if (!isScannable(doc)) { return; }
  const text = doc.getText();
  if (!text.trim()) { return; }

  let result: any;
  try {
    result = await postJson<any>('/api/scan', { text, source: doc.fileName });
  } catch {
    // Dashboard not running — silently skip
    return;
  }

  if (result.score < 0.20) {
    diagnostics.delete(doc.uri);
    updateStatusBar('clean');
    return;
  }

  // Find the line containing the evidence snippet
  const snippet: string = (result.evidence_snippet || '').slice(0, 60).trim();
  let lineIndex = 0;
  if (snippet) {
    for (let i = 0; i < doc.lineCount; i++) {
      if (doc.lineAt(i).text.includes(snippet)) {
        lineIndex = i;
        break;
      }
    }
  }

  const range    = doc.lineAt(Math.min(lineIndex, doc.lineCount - 1)).range;
  const severity = result.score >= 0.75
    ? vscode.DiagnosticSeverity.Error
    : vscode.DiagnosticSeverity.Warning;

  const heuristics: string[] = result.matched_heuristics ?? [];
  const message = [
    `AgentForensics [${result.verdict.toUpperCase()}]`,
    `score ${result.score.toFixed(3)}`,
    heuristics.length ? `— ${heuristics.join(', ')}` : '',
  ].filter(Boolean).join('  ');

  const diag        = new vscode.Diagnostic(range, message, severity);
  diag.source       = 'AgentForensics';
  diag.code         = heuristics[0] ?? 'INJECTION';

  diagnostics.set(doc.uri, [diag]);
  updateStatusBar(result.verdict);

  // Also flash an alert in the sidebar
  sidebarProvider.postAlert(
    `${result.verdict.toUpperCase()} in ${doc.fileName.split(/[\\/]/).pop()}`
  );
}

// Debounced wrapper used for paste / Copilot events
function scheduleScan(doc: vscode.TextDocument): void {
  if (_pasteDebounce) { clearTimeout(_pasteDebounce); }
  _pasteDebounce = setTimeout(() => scanDocument(doc), 600);
}

// Scan every currently open document (called when dashboard comes online)
function scanAllOpenDocuments(): void {
  vscode.workspace.textDocuments.forEach(doc => scanDocument(doc));
}

// ---------------------------------------------------------------------------
// Start dashboard
// ---------------------------------------------------------------------------

function startDashboard(): void {
  const { dashboardPort } = getConfig();
  const terminal = vscode.window.createTerminal('AF Dashboard');
  terminal.sendText(
    `${pythonCmd()} -m agentforensics.cli dashboard --port ${dashboardPort}`
  );
  terminal.show(true);

  setTimeout(() => {
    vscode.env.openExternal(vscode.Uri.parse(`http://127.0.0.1:${dashboardPort}`));
    connectSSE();
  }, 2000);
}

// ---------------------------------------------------------------------------
// SSE — live injection alerts from the proxy backend
// ---------------------------------------------------------------------------

function connectSSE(): void {
  if (sseRequest) { return; }
  const { dashboardPort } = getConfig();

  const req = http.get(
    `http://127.0.0.1:${dashboardPort}/api/live`,
    { headers: { Accept: 'text/event-stream' } },
    (res: http.IncomingMessage) => {
      let buf = '';
      res.on('data', (chunk: Buffer) => {
        buf += chunk.toString();
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try { onInjectionAlert(JSON.parse(line.slice(6))); }
            catch (_) { /* ignore */ }
          }
        }
      });
      res.on('end', () => { sseRequest = undefined; setTimeout(connectSSE, 5000); });
      // Dashboard just came online — scan everything currently open
      scanAllOpenDocuments();
    }
  );
  req.on('error', () => { sseRequest = undefined; setTimeout(connectSSE, 8000); });
  sseRequest = req;
}

function onInjectionAlert(data: any): void {
  const sessionShort = (data.session_id ?? '').slice(0, 12) + '…';
  const score        = typeof data.score === 'number' ? data.score.toFixed(3) : '?';
  const turn         = data.turn_index ?? '?';
  const { dashboardPort } = getConfig();

  updateStatusBar('compromised');

  vscode.window.showWarningMessage(
    `🔴 INJECTION DETECTED — session ${sessionShort}  turn ${turn}  score ${score}`,
    'View Report', 'Dismiss'
  ).then(choice => {
    if (choice === 'View Report') {
      vscode.env.openExternal(vscode.Uri.parse(`http://127.0.0.1:${dashboardPort}`));
    }
  });

  sidebarProvider.refresh();
}

// ---------------------------------------------------------------------------
// Sidebar WebviewView
// ---------------------------------------------------------------------------

class AgentForensicsSidebarProvider implements vscode.WebviewViewProvider {
  private _view?: vscode.WebviewView;

  constructor(_extensionUri: vscode.Uri) {}

  resolveWebviewView(view: vscode.WebviewView): void {
    this._view = view;
    view.webview.options = { enableScripts: true };
    view.webview.html = this._getHtml();

    view.webview.onDidReceiveMessage(msg => {
      if (msg.type === 'refresh')       { this.refresh(); }
      if (msg.type === 'openDashboard') { startDashboard(); }
      if (msg.type === 'openReport') {
        vscode.env.openExternal(
          vscode.Uri.parse(`${apiBase()}/api/sessions/${msg.sessionId}/report`)
        );
      }
    });

    this.refresh();
  }

  refresh(): void {
    if (!this._view) { return; }
    this._view.webview.postMessage({ type: 'loading' });
    fetchJson<any[]>('/api/sessions')
      .then(sessions => this._view!.webview.postMessage({ type: 'sessions', sessions }))
      .catch(()      => this._view!.webview.postMessage({ type: 'error' }));
  }

  postAlert(text: string): void {
    this._view?.webview.postMessage({ type: 'alert', text });
  }

  private _getHtml(): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: var(--vscode-font-family);
    font-size: var(--vscode-font-size);
    color: var(--vscode-foreground);
    background: var(--vscode-sideBar-background);
  }
  #toolbar {
    display: flex; gap: 6px; padding: 8px;
    border-bottom: 1px solid var(--vscode-sideBarSectionHeader-border);
  }
  button {
    flex: 1; padding: 4px 8px;
    background: var(--vscode-button-background);
    color: var(--vscode-button-foreground);
    border: none; border-radius: 3px; cursor: pointer;
    font-size: 11px; font-family: var(--vscode-font-family);
  }
  button:hover { background: var(--vscode-button-hoverBackground); }
  button.secondary {
    background: var(--vscode-button-secondaryBackground);
    color: var(--vscode-button-secondaryForeground);
  }
  button.secondary:hover { background: var(--vscode-button-secondaryHoverBackground); }
  #proxy-hint {
    padding: 8px; font-size: 10px;
    color: var(--vscode-descriptionForeground);
    border-bottom: 1px solid var(--vscode-sideBarSectionHeader-border);
  }
  #proxy-hint code {
    font-family: var(--vscode-editor-font-family);
    background: var(--vscode-textCodeBlock-background);
    padding: 1px 4px; border-radius: 2px; font-size: 10px;
  }
  #sessions { overflow-y: auto; }
  .session-item {
    padding: 8px 10px; cursor: pointer;
    border-bottom: 1px solid var(--vscode-sideBarSectionHeader-border);
  }
  .session-item:hover { background: var(--vscode-list-hoverBackground); }
  .row1 { display: flex; justify-content: space-between; align-items: center; }
  .sid {
    font-family: var(--vscode-editor-font-family); font-size: 10px;
    font-weight: 600; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; max-width: 140px;
  }
  .badge {
    font-size: 9px; font-weight: 700; padding: 1px 6px;
    border-radius: 8px; color: #fff; white-space: nowrap; flex-shrink: 0;
  }
  .badge-clean       { background: #22c55e; }
  .badge-suspicious  { background: #f59e0b; }
  .badge-compromised { background: #ef4444; }
  .meta { font-size: 10px; color: var(--vscode-descriptionForeground); margin-top: 2px; }
  .status {
    padding: 16px 10px; color: var(--vscode-descriptionForeground);
    font-size: 11px; text-align: center;
  }
  #alert-banner {
    display: none; padding: 6px 10px;
    background: #7f1d1d; color: #fecaca;
    font-size: 11px; font-weight: 600;
    border-bottom: 1px solid #ef4444;
  }
</style>
</head>
<body>
<div id="toolbar">
  <button onclick="send('openDashboard')">&#127760; Open Dashboard</button>
</div>
<div id="alert-banner"></div>
<div id="sessions"><div class="status">Connecting to dashboard…</div></div>
<script>
  const vscode = acquireVsCodeApi();
  function send(type, extra) { vscode.postMessage({ type, ...extra }); }
  function verdictClass(v) { return 'badge-' + (v || 'clean'); }
  function verdictLabel(v) {
    return { clean:'CLEAN', suspicious:'SUSPICIOUS', compromised:'COMPROMISED' }[v] || (v||'').toUpperCase();
  }
  function fmtDate(iso) {
    if (!iso) return '';
    try { return new Date(iso).toLocaleString(undefined, { month:'short', day:'numeric', hour:'2-digit', minute:'2-digit' }); }
    catch(_) { return iso.slice(0,16); }
  }
  function escHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }
  window.addEventListener('message', e => {
    const msg = e.data;
    const $s = document.getElementById('sessions');
    if (msg.type === 'loading') { $s.innerHTML = '<div class="status">Loading…</div>'; return; }
    if (msg.type === 'error')   { $s.innerHTML = '<div class="status">Dashboard not running.<br>Click "Dashboard" to start it.</div>'; return; }
    if (msg.type === 'sessions') {
      if (!msg.sessions.length) { $s.innerHTML = '<div class="status">No sessions yet.<br>Start the proxy and make LLM calls.</div>'; return; }
      $s.innerHTML = msg.sessions.map(s => {
        const shortId = s.session_id.length > 20 ? s.session_id.slice(0,10)+'…'+s.session_id.slice(-6) : s.session_id;
        return \`<div class="session-item" onclick="send('openReport',{sessionId:'\${escHtml(s.session_id)}'})">
          <div class="row1">
            <span class="sid" title="\${escHtml(s.session_id)}">\${escHtml(shortId)}</span>
            <span class="badge \${verdictClass(s.verdict)}">\${verdictLabel(s.verdict)}</span>
          </div>
          <div class="meta">\${escHtml(s.model||'—')} · \${fmtDate(s.started_at)} · \${s.turn_count} turns</div>
        </div>\`;
      }).join('');
      return;
    }
    if (msg.type === 'alert') {
      const b = document.getElementById('alert-banner');
      b.textContent = '🔴 ' + msg.text;
      b.style.display = 'block';
      setTimeout(() => { b.style.display = 'none'; }, 8000);
      send('refresh');
    }
  });
</script>
</body>
</html>`;
  }
}

// ---------------------------------------------------------------------------
// Activate
// ---------------------------------------------------------------------------

export function activate(context: vscode.ExtensionContext): void {
  // ── Diagnostics collection (inline highlighting + Problems panel) ──────
  diagnostics = vscode.languages.createDiagnosticCollection('agentforensics');
  context.subscriptions.push(diagnostics);

  // ── Status bar ─────────────────────────────────────────────────────────
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  updateStatusBar('idle');
  statusBarItem.command = 'agentforensics.startDashboard';
  statusBarItem.show();
  context.subscriptions.push(statusBarItem);

  // ── Sidebar ────────────────────────────────────────────────────────────
  sidebarProvider = new AgentForensicsSidebarProvider(context.extensionUri);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      'agentforensics.sidebar', sidebarProvider,
      { webviewOptions: { retainContextWhenHidden: true } }
    )
  );

  // ── Commands ───────────────────────────────────────────────────────────
  context.subscriptions.push(
    vscode.commands.registerCommand('agentforensics.startDashboard', startDashboard),
    vscode.commands.registerCommand('agentforensics.refresh',        () => sidebarProvider.refresh()),
    vscode.commands.registerCommand('agentforensics.scanFile',       () => {
      const doc = vscode.window.activeTextEditor?.document;
      if (doc) { scanDocument(doc); }
      else { vscode.window.showInformationMessage('Open a file to scan it.'); }
    }),
  );

  // ── Feature 1: Auto-scan on file open ─────────────────────────────────
  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument(doc => {
      if (getConfig().scanOnOpen) { scanDocument(doc); }
    })
  );

  // Scan the already-active file when the extension first loads
  const activeDoc = vscode.window.activeTextEditor?.document;
  if (activeDoc && getConfig().scanOnOpen) {
    setTimeout(() => scanDocument(activeDoc), 1500);
  }

  // Update status bar when user switches tabs
  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(editor => {
      if (!editor) { updateStatusBar('idle'); return; }
      // Show existing diagnostic verdict for this file if any
      const existing = diagnostics.get(editor.document.uri);
      if (!existing || existing.length === 0) {
        updateStatusBar('clean');
      } else {
        const worst = existing.reduce((a, b) => a.severity < b.severity ? a : b);
        updateStatusBar(worst.severity === vscode.DiagnosticSeverity.Error ? 'compromised' : 'suspicious');
      }
    })
  );

  // ── Re-scan on save (clears stale diagnostics after edits) ───────────
  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument(doc => {
      scanDocument(doc);
    })
  );

  // ── Features 3 & 5: Scan on paste + Copilot response monitoring ───────
  // Both paste and Copilot completions show up as multi-line single-change
  // insertions in onDidChangeTextDocument.
  // Also clear stale diagnostics immediately on any edit.
  context.subscriptions.push(
    vscode.workspace.onDidChangeTextDocument(e => {
      // Clear stale result as soon as the user starts editing
      diagnostics.delete(e.document.uri);
      updateStatusBar('idle');

      if (!getConfig().scanOnPaste) { return; }
      for (const change of e.contentChanges) {
        // Multi-line insertion = paste or accepted Copilot suggestion
        if (change.text.includes('\n') && change.text.trim().length > 10) {
          scheduleScan(e.document);
          break;
        }
      }
    })
  );

  // ── Auto-start if configured ───────────────────────────────────────────
  if (getConfig().autoStart) {
    startDashboard();
  }

  // Try to connect SSE in case dashboard is already running
  setTimeout(connectSSE, 2000);
}

export function deactivate(): void {
  sseRequest?.destroy();
  dashboardProcess?.kill();
}
