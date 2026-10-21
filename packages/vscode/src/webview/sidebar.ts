/**
 * Llama AIO sidebar webview. Bundled by esbuild to out/webview/sidebar.js and
 * loaded by SettingsViewProvider with a nonce CSP. Messages to the extension
 * are typed by ./protocol, so a renamed or misspelled message fails the build.
 */
import { $, vscode } from "./api";
import type { ToWebviewMessage } from "./protocol";
import { escAttr, escText, fmtBytes, fmtEta, fmtNum, fmtParams, fmtRate, fmtServerRate, fmtTokK, fmtTokShort, fmtUptime, pickNum } from "./format";
import {
  clampMainGpu,
  denseFfnShareOf,
  gpuFullLabel,
  gpuLabel,
  isLegacyGpu0FirstSplit,
  mainWeightShareFromSplitLive,
  moeExpertShareOf,
  tensorSplitForMainShare,
  tensorSplitForTargetWeightShareLive,
  tensorSplitShares,
} from "./gpuSplit";
import { sparklineSvg } from "./perfChart";
import {
  lib,
  applyLibraryHfMessage,
  allLibraryGroups,
  closeLibraryMenu,
  findLibraryQuant,
  fitText,
  onLibraryClick,
  openLibraryMenu,
  refreshLibrarySum,
  resetLibraryNavigation,
  renderLibraryDialog,
  renderLibraryFilters,
  renderLibraryJobs,
  renderLibraryList,
  renderLibraryPartials,
  selectLibraryPath,
  shortModelLabel,
  startLibrarySearch,
} from "./library";



let memInputs = null;
let gpuInfos = [];
let mtpSidecarPath = '';
let backendOptionsCache = [];
let modelIsMoe = false;
let modelBlockCount = 128;
let cpuLogicalCores = 64;
let moeHintDefault = 'Number of layers to force experts onto CPU (--n-cpu-moe). Only applies to MoE models.';
let ffnHintDefault = 'Number of layers to force dense FFN weights onto CPU (--n-cpu-ffn). Only applies to dense (non-MoE) models.';
let suppressBackendChange = false;
let activeBackendId = '';
let serverRunning = false;
let foundServers = [];
let configDirty = false;
let serverStarting = false;
let lastEndpoint = '';
let lastPid;
let saveLoadTimer = null;
let updateCheck = { latestTag: undefined, installedTag: undefined, updateAvailable: false, checkFailed: false, pending: true };
let llamaInstall = null;
/** Settings the running server has not picked up yet ({ label, from, to }). */
let pendingChanges = [];
let lastPayload = null;
let lastMemoryView = null;
let loadDefaults = {};
/** "Custom" GPU layers stays open while the user works with the slider. */
let nglCustomOpen = false;

let remoteMode = false;
let connectOpen = false;
let remoteInfo = null;
let remoteHint = '';
let nextRetryAt = 0;
let probeTimer = null;


function setConnectOpen(open, prefill?) {
  connectOpen = !!open;
  document.body.classList.toggle('connect-open', connectOpen);
  const form = $('connectForm');
  if (form) form.classList.toggle('hidden', !connectOpen);
  if (connectOpen && prefill) {
    const url = $('remoteUrlInput');
    const label = $('remoteLabelInput');
    if (url && prefill.url && document.activeElement !== url) url.value = prefill.url;
    if (label && prefill.label != null && document.activeElement !== label) label.value = prefill.label;
  }
  const card = $('remoteModelCard');
  if (card) card.classList.toggle('hidden', !remoteMode || connectOpen);
  if (!connectOpen) {
    const list = $('probeList');
    if (list) list.textContent = '';
  }
}

function renderProbeList(info, error) {
  const list = $('probeList');
  if (!list) return;
  list.textContent = '';
  if (error) {
    const row = document.createElement('div');
    row.className = 'probe-row';
    row.textContent = error;
    list.appendChild(row);
    return;
  }
  const probes = (info && info.probes) || [];
  probes.forEach(function (p) {
    const row = document.createElement('div');
    row.className = 'probe-row';
    const mark = document.createElement('span');
    mark.className = 'mark ' + (p.ok === true ? 'ok' : p.ok === false ? 'bad' : 'no');
    mark.textContent = p.ok === true ? '✓' : p.ok === false ? '✕' : '–';
    const grow = document.createElement('span');
    grow.className = 'grow';
    grow.textContent = p.detail || '';
    const pathEl = document.createElement('span');
    pathEl.className = 'path';
    pathEl.textContent = p.path || '';
    row.appendChild(mark);
    row.appendChild(grow);
    row.appendChild(pathEl);
    list.appendChild(row);
  });
}

function renderRemoteModel() {
  const card = $('remoteModelCard');
  if (!card) return;
  card.classList.toggle('hidden', !remoteMode || connectOpen);
  const info = remoteInfo || {};
  const name = info.modelId || (lastPayload && lastPayload.remoteUrl ? 'Waiting for the server…' : '—');
  const nameEl = $('remoteModelName');
  if (nameEl) nameEl.textContent = name;
  const sub = $('remoteModelSub');
  if (sub) {
    sub.textContent = info.state === 'auth' ? 'partial' : (lastPayload && lastPayload.remoteLabel) || '';
  }
  const chips = $('remoteChips');
  if (chips) {
    chips.textContent = '';
    const bits = [];
    if (info.modelParams) bits.push(fmtParams(info.modelParams));
    if (info.modelSizeBytes) bits.push(fmtBytes(info.modelSizeBytes));
    if (info.trainedContext) bits.push('trained ' + fmtTokShort(info.trainedContext) + ' ctx');
    if (info.slotContext || info.servedContext) bits.push('served ' + fmtTokShort(info.slotContext || info.servedContext) + ' ctx');
    if (info.vision) bits.push('vision');
    bits.forEach(function (text) {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = text;
      chips.appendChild(chip);
    });
  }
  const pathEl = $('remoteModelPath');
  if (pathEl) pathEl.textContent = info.modelPath || '';
  const hint = $('remoteModelHint');
  if (hint) {
    if (info.isLlamaCpp === false && (info.state === 'ready' || info.state === 'limited')) {
      hint.textContent = 'This server is not llama.cpp. Chat works; slots, vision, and the overthinking penalty stay off.';
    } else if (info.state === 'auth') {
      hint.textContent = 'Slots, context, and vision need the API key (/props).';
    } else if (info.state === 'unreachable') {
      hint.textContent = info.modelId ? 'Last model this server reported.' : '';
    } else if (lastPayload && lastPayload.remoteLabel) {
      hint.textContent = 'Context, GPU layers, and KV cache are set by whoever starts llama-server on ' + lastPayload.remoteLabel + '.';
    } else {
      hint.textContent = '';
    }
  }
}

function renderRemoteStatus() {
  const info = remoteInfo || {};
  const state = info.state || 'connecting';
  let kind = 'starting';
  let label = 'Connecting…';
  if (state === 'ready') { kind = 'ok'; label = 'Connected'; }
  else if (state === 'busy') { kind = 'ok'; label = 'Connected · busy'; }
  else if (state === 'limited') { kind = 'ok'; label = 'Connected · limited'; }
  else if (state === 'loading') { kind = 'starting'; label = 'Loading model'; }
  else if (state === 'auth') { kind = 'dirty'; label = 'API key required'; }
  else if (state === 'unreachable') { kind = 'error'; label = 'Not reachable'; }
  setServerCardKind(kind);
  const line = $('statusLine');
  const dot = $('statusDot');
  const text = $('statusText');
  if (line) line.className = 'status-line ' + kind;
  if (dot) dot.className = 'dot ' + (state === 'loading' || state === 'connecting' ? 'starting' : kind);
  if (text) text.textContent = label;
  const meta = $('statusMeta');
  const endpoint = (info.url || (lastPayload && lastPayload.endpoint) || '');
  if (meta) {
    meta.innerHTML = '';
    if (endpoint) {
      const a = document.createElement('a');
      a.className = 'endpoint';
      a.href = endpoint;
      a.setAttribute('data-url', endpoint);
      a.textContent = endpoint;
      a.addEventListener('click', function (e) {
        e.preventDefault();
        vscode.postMessage({ type: 'openExternal', url: endpoint });
      });
      meta.appendChild(a);
    }
    const bits = [];
    if (lastPayload && lastPayload.remoteLabel && lastPayload.remoteLabel !== endpoint) bits.push(lastPayload.remoteLabel);
    if (typeof info.rttMs === 'number') bits.push(info.rttMs + ' ms');
    else if (state === 'unreachable' && nextRetryAt) {
      const sec = Math.max(0, Math.round((nextRetryAt - Date.now()) / 1000));
      bits.push(sec ? 'next try in ' + sec + ' s' : 'retrying…');
    }
    if (bits.length) meta.appendChild(document.createTextNode((endpoint ? ' · ' : '') + bits.join(' · ')));
  }
  const modelEl = $('srvModel');
  if (modelEl) {
    modelEl.textContent = state === 'loading'
      ? 'Waiting for ' + ((lastPayload && lastPayload.remoteLabel) || 'the server') + '…'
      : (info.modelId || 'No model reported yet');
    modelEl.title = info.modelPath || '';
  }
  const sumEl = $('srvSummary');
  if (sumEl) {
    const bits = [];
    if (info.slotContext) bits.push(fmtTokShort(info.slotContext) + ' context window per slot');
    if (info.totalSlots) bits.push(info.totalSlots + (info.totalSlots === 1 ? ' slot' : ' slots'));
    if (info.vision) bits.push('vision');
    if (info.buildInfo) bits.push('llama.cpp ' + String(info.buildInfo).split('-')[0]);
    if (state === 'auth') bits.push('model name from public /v1/models');
    if (state === 'limited') bits.push('OpenAI-compatible · not llama.cpp');
    sumEl.textContent = bits.join(' · ');
    sumEl.classList.toggle('hidden', !bits.length);
  }
  const perfEl = $('srvPerf');
  const hasSlots = typeof info.slotsTotal === 'number' && info.slotsTotal > 0 && typeof info.slotsBusy === 'number';
  const hasMetrics = typeof info.requestsProcessing === 'number'
    || typeof info.requestsDeferred === 'number'
    || typeof info.serverGenTokPerSec === 'number'
    || typeof info.serverPromptTokPerSec === 'number';
  const perfBits = [];
  if (hasSlots) perfBits.push(info.slotsBusy + '/' + info.slotsTotal + ' slots busy');
  else if (typeof info.requestsProcessing === 'number') perfBits.push(info.requestsProcessing + ' running');
  if (info.requestsDeferred > 0) perfBits.push(info.requestsDeferred + ' queued');
  if (perfEl) {
    perfEl.textContent = perfBits.join(' · ');
    perfEl.classList.toggle('hidden', !perfBits.length);
  }
  const load = $('remoteLoad');
  if (load) load.classList.toggle('hidden', !(hasSlots || hasMetrics));
  const loadSum = $('remoteLoadSum');
  if (loadSum) loadSum.textContent = hasMetrics ? 'all clients · /metrics' : 'all clients · /slots';
  const loadLabel = $('remoteLoadLabel');
  const loadCount = $('remoteLoadCount');
  const loadBar = $('remoteLoadBar');
  const loadFill = $('remoteLoadFill');
  if (loadLabel) {
    loadLabel.textContent = hasSlots ? 'Slots busy' : 'Requests running';
    if (loadLabel.parentElement) loadLabel.parentElement.classList.toggle('hidden', !(hasSlots || typeof info.requestsProcessing === 'number'));
  }
  if (hasSlots) {
    if (loadCount) loadCount.textContent = info.slotsBusy + ' / ' + info.slotsTotal;
    if (loadBar) {
      loadBar.classList.toggle('hidden', false);
      loadBar.classList.toggle('warn', info.slotsBusy >= info.slotsTotal);
    }
    if (loadFill) {
      const pct = Math.max(0, Math.min(100, (info.slotsBusy / info.slotsTotal) * 100));
      loadFill.style.width = pct + '%';
    }
  } else {
    if (loadCount) loadCount.textContent = typeof info.requestsProcessing === 'number' ? String(info.requestsProcessing) : '';
    if (loadBar) loadBar.classList.add('hidden');
  }
  const rates = $('remoteLoadRates');
  if (rates) {
    const bits = [];
    const gen = fmtServerRate(info.serverGenTokPerSec);
    const prompt = fmtServerRate(info.serverPromptTokPerSec);
    if (gen) bits.push('gen ' + gen + ' tok/s');
    if (prompt) bits.push('prompt ' + prompt + ' tok/s');
    if (info.requestsDeferred > 0) bits.push(info.requestsDeferred + ' waiting');
    rates.textContent = bits.length ? (gen || prompt ? 'Server-wide ' : '') + bits.join(' · ') : '';
  }
  const strip = $('remoteStrip');
  const actions = $('remoteActions');
  const primary = $('remotePrimaryBtn');
  const secondary = $('remoteSecondaryBtn');
  let stripText = '';
  let stripKind = '';
  let stripLead = '';
  if (state === 'auth') {
    stripKind = 'warn';
    stripText = (info.error && info.error.message) || 'The server rejected the request (401).';
    if (remoteHint) stripText += ' ' + remoteHint;
  } else if (state === 'loading') {
    stripKind = 'info';
    stripText = 'The server is up but still loading a model (/health 503). Copilot requests fail fast until it is ready.';
  } else if (state === 'unreachable') {
    stripKind = 'bad';
    stripText = ((info.error && info.error.message) || 'Not reachable.') + (remoteHint ? ' ' + remoteHint : '');
  } else if (state === 'limited') {
    stripKind = 'info';
    stripText = 'Chat works. llama.cpp extras (slot inspection, vision, overthinking penalty) are off because /props did not answer.';
  } else if (state === 'busy') {
    stripKind = 'warn';
    stripLead = 'Shared server.';
    const full = hasSlots && info.slotsBusy >= info.slotsTotal;
    stripText = full
      ? 'Other clients are using every slot. Your next request waits in the server\'s queue, so expect slower first tokens.'
      : 'Requests are waiting in the server\'s queue, so expect slower first tokens.';
  }
  if (strip) {
    strip.className = 'remote-strip' + (stripKind ? ' ' + stripKind : '') + (stripText ? '' : ' hidden');
    strip.textContent = '';
    if (stripText) {
      if (stripLead) {
        const lead = document.createElement('span');
        lead.className = 'lead';
        lead.textContent = stripLead + ' ';
        strip.appendChild(lead);
      }
      strip.appendChild(document.createTextNode(stripText));
    }
  }
  if (actions && primary && secondary) {
    const show = state === 'auth' || state === 'unreachable';
    actions.classList.toggle('hidden', !show);
    if (state === 'auth') {
      primary.textContent = 'Set API key…';
      primary.dataset.action = 'setApiKey';
      secondary.textContent = 'Edit connection';
      secondary.dataset.action = 'edit';
    } else if (state === 'unreachable') {
      primary.textContent = 'Retry now';
      primary.dataset.action = 'retry';
      secondary.textContent = 'Edit connection';
      secondary.dataset.action = 'edit';
    }
  }
  const hint = $('dirtyHint');
  if (hint) hint.classList.add('hidden');
  renderRemoteModel();
}

function applyRemoteChrome(payload) {
  remoteMode = !!(payload && payload.serverMode === 'remote');
  document.body.classList.toggle('remote-mode', remoteMode);
  if (remoteMode && connectOpen) setConnectOpen(false);
  if (typeof syncChatTemplateUi === 'function') syncChatTemplateUi();
  if (payload && payload.remote) remoteInfo = payload.remote;
  else if (!remoteMode) remoteInfo = null;
  remoteHint = (payload && payload.remoteHint) || '';
  nextRetryAt = (payload && payload.nextRetryAt) || 0;
  const card = $('remoteModelCard');
  if (card) card.classList.toggle('hidden', !remoteMode || connectOpen);
  if (remoteMode && !connectOpen) renderRemoteStatus();
  else if (!remoteMode) clearRemoteStatus();
}

/**
 * Only renderRemoteStatus fills the remote error strip, its Retry / Edit
 * buttons and the server-load fold, so leaving remote mode must empty them;
 * otherwise a "Connection refused" box outlives the switch to this machine.
 */
function clearRemoteStatus() {
  for (const id of ['remoteStrip', 'remoteActions', 'remoteLoad']) {
    const el = $(id);
    if (el) el.classList.add('hidden');
  }
  const strip = $('remoteStrip');
  if (strip) strip.textContent = '';
}

function setServerCardKind(kind) {
  const card = $('serverCard');
  if (!card) return;
  card.className = 'card server' + (kind && kind !== 'ok' ? ' ' + kind : '');
}

function listenHost(url) {
  try {
    const u = new URL(url);
    const host = u.hostname.indexOf(':') >= 0 ? '[' + u.hostname + ']' : u.hostname;
    return host + ':' + (u.port || '');
  } catch (e) {
    return String(url || '');
  }
}

function renderFoundServers() {
  const box = $('foundServers');
  if (!box) return;
  const show = !remoteMode && !connectOpen && !serverRunning && !serverStarting && foundServers.length;
  box.classList.toggle('hidden', !show);
  box.textContent = '';
  if (!show) return;
  foundServers.forEach(function (srv) {
    if (!srv || !srv.url) return;
    const row = document.createElement('div');
    row.className = 'found-server';
    const text = document.createElement('span');
    text.textContent = (srv.name || 'llama-server') + ' is listening on ' + listenHost(srv.url);
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'secondary';
    btn.textContent = 'Connect';
    btn.addEventListener('click', function () {
      vscode.postMessage({ type: 'connectRemote', url: srv.url, label: 'localhost', apiKey: '' });
    });
    row.appendChild(text);
    row.appendChild(btn);
    box.appendChild(row);
  });
}

function renderStatusUi(opts) {
  if (remoteMode && !connectOpen) {
    renderRemoteStatus();
    renderFoundServers();
    return;
  }
  const ready = !!opts.ready;
  const dirty = !!opts.dirty;
  const starting = !!opts.starting;
  const endpoint = opts.endpoint || '';
  const pid = opts.pid;
  const message = String(opts.message || '');
  const looksError = !ready && !starting && /error|fail|exit|crash/i.test(message);

  let kind = 'stopped';
  let label = 'Stopped';
  if (starting) {
    kind = 'starting';
    if (/Loading model/i.test(message)) {
      label = 'Loading model…';
    } else if (/Stopping/i.test(message)) {
      label = 'Stopping…';
    } else if (/Reload/i.test(message)) {
      label = 'Reloading…';
    } else {
      label = 'Starting…';
    }
  } else if (ready && dirty) {
    kind = 'dirty';
    label = 'Server ready';
  } else if (ready) {
    kind = 'ok';
    label = 'Server ready';
  } else if (looksError) {
    kind = 'error';
    label = 'Error';
  }

  setServerCardKind(kind);
  const line = $('statusLine');
  const dot = $('statusDot');
  const text = $('statusText');
  const meta = $('statusMeta');
  if (line) line.className = 'status-line ' + kind;
  if (dot) dot.className = 'dot ' + kind;
  if (text) text.textContent = label;

  if (meta) {
    meta.innerHTML = '';
    if (starting) {
      meta.textContent = message || 'Waiting for HTTP ready…';
    } else if (ready && endpoint && isDefaultLocalEndpoint(endpoint)) {
      // Default 127.0.0.1:8742: endpoint and pid live in the header tooltip
      // and the ⋯ menu ("Open llama.cpp web UI"), not on the first line.
    } else if (ready && endpoint) {
      const a = document.createElement('a');
      a.className = 'endpoint';
      a.href = endpoint;
      a.setAttribute('data-url', endpoint);
      a.title = 'Open llama-server web UI';
      a.textContent = endpoint;
      a.addEventListener('click', (e) => {
        e.preventDefault();
        const url = a.getAttribute('data-url');
        if (url) vscode.postMessage({ type: 'openExternal', url: url });
      });
      meta.appendChild(a);
      if (pid) {
        const pidEl = document.createElement('span');
        pidEl.className = 'pid';
        pidEl.textContent = ' · pid ' + pid;
        meta.appendChild(pidEl);
      }
    } else if (looksError && message) {
      meta.textContent = message;
    } else if (endpoint && !ready) {
      meta.textContent = endpoint + ' · not running';
    } else {
      meta.textContent = 'No endpoint · not running';
    }
  }

  const hint = $('dirtyHint');
  // Hide dirty while loading — settings are being applied right now.
  if (hint) hint.classList.toggle('hidden', starting || !(ready && dirty));
  renderFoundServers();
}

/**
 * The big button only appears when there is something to do: Start,
 * Loading…, or Reload to apply. A clean running server gets the small ⟳.
 */
function updatePrimaryAction() {
  const reloadIcon = $('reloadIconBtn');
  if (remoteMode) {
    const row = $('primaryRow');
    if (row) row.classList.add('hidden');
    const stop = $('stopBtn');
    if (stop) stop.disabled = true;
    if (reloadIcon) {
      reloadIcon.classList.toggle('hidden', connectOpen);
      reloadIcon.title = 'Re-probe the remote server';
      reloadIcon.setAttribute('aria-label', 'Re-probe the remote server');
    }
    return;
  }
  if (reloadIcon) {
    reloadIcon.title = 'Reload llama-server (restart with the same settings)';
    reloadIcon.setAttribute('aria-label', 'Reload server');
  }
  const primary = $('primaryBtn');
  const stop = $('stopBtn');
  const row = $('primaryRow');
  if (!primary || !stop) return;
  stop.disabled = !serverRunning && !serverStarting;
  primary.classList.remove('warn-primary');
  primary.classList.add('primary');
  let showRow = true;
  if (serverStarting) {
    primary.disabled = true;
    primary.textContent = 'Loading…';
    primary.dataset.action = '';
  } else if (!serverRunning) {
    primary.disabled = false;
    primary.textContent = 'Start';
    primary.dataset.action = 'start';
  } else if (configDirty) {
    // The slim pending bar (Reload / Discard) replaces the big button.
    showRow = false;
    primary.dataset.action = 'reload';
  } else {
    showRow = false;
    primary.dataset.action = 'reload';
  }
  if (row) row.classList.toggle('hidden', !showRow);
  const discard = $('discardBtn');
  if (discard) discard.classList.toggle('hidden', !pendingChanges.length);
  if (reloadIcon) reloadIcon.classList.toggle('hidden', !(serverRunning && !serverStarting && !configDirty));
  renderDirtyList();
  refreshPendingMarks();
  applyLayout();
}

/** Slim "N changes not applied" bar; the diff opens on click. */
function renderDirtyList() {
  const list = $('dirtyList');
  const count = $('dirtyCount');
  if (!list) return;
  const n = pendingChanges.length;
  if (count) count.textContent = n ? n + ' change' + (n === 1 ? '' : 's') + ' not applied' : 'Settings changed, not applied';
  list.innerHTML = n
    ? pendingChanges.map((c) => '<div class="pending-row"><span>' + escText(c.label) + '</span><span class="mono">' +
        escText(c.from) + ' → ' + escText(c.to) + '</span></div>').join('')
    : '<div class="hint" style="margin:0">Load settings or launch mode changed.</div>';
}

/**
 * Field ids that edit a load setting under a different name (sliders,
 * "auto" checkboxes). Same pairs as ADV_ALIAS plus the first-level sliders.
 */
const PENDING_ALIAS = {
  contextLengthRange: 'contextLength', gpuOffloadRange: 'gpuOffload', tensorSplitRange: 'tensorSplit',
  reasoningBudgetUnlimited: 'reasoningBudget', ropeBaseAuto: 'ropeFreqBase', ropeScaleAuto: 'ropeFreqScale',
  seedRandom: 'seed', cpuThreadsRange: 'cpuThreads', ngramSizeNRange: 'ngramSizeN',
};

/**
 * Amber dot + "was X" on every load field that differs from what the running
 * server uses. (Differences from the *defaults* use the muted dot instead.)
 */
function refreshPendingMarks() {
  const card = $('memCard');
  if (!card) return;
  const byKey = {};
  if (serverRunning && configDirty) {
    for (const c of pendingChanges) if (c && c.key) byKey[c.key] = c;
  }
  for (const row of card.querySelectorAll('.row, .toggle')) {
    let hit = null;
    for (const el of row.querySelectorAll('input[id], select[id]')) {
      const key = PENDING_ALIAS[el.id] || el.id;
      if (byKey[key]) { hit = byKey[key]; break; }
    }
    // A nested row (spec groups) marks itself, not its parent.
    if (hit && row.querySelector('.row') && !row.matches('.toggle')) hit = null;
    row.classList.toggle('pending', !!hit);
    const marker = row.querySelector(':scope > .label > .name') || row.querySelector(':scope > span:first-child');
    if (!marker) continue;
    let was = marker.nextElementSibling && marker.nextElementSibling.classList.contains('was')
      ? marker.nextElementSibling
      : null;
    if (hit) {
      if (!was) {
        was = document.createElement('span');
        was.className = 'was';
        marker.after(was);
      }
      was.textContent = 'was ' + hit.from;
    } else if (was) {
      was.remove();
    }
  }
}




function isDefaultLocalEndpoint(endpoint) {
  return /^https?:\/\/(127\.0\.0\.1|localhost|\[::1\]):8742\/?$/i.test(String(endpoint || ''));
}

/**
 * Card order follows what the user is doing (CSS `order` per body class):
 * running → live performance right under the header; stopped → setup first,
 * Performance last; remote → the remote cards.
 */
function applyLayout() {
  const layout = remoteMode ? 'remote' : (serverRunning && !serverStarting ? 'running' : 'stopped');
  for (const name of ['running', 'stopped', 'remote']) {
    document.body.classList.toggle('layout-' + name, layout === name);
  }
  const live = $('srvLive');
  if (live) live.classList.toggle('hidden', layout !== 'running' || connectOpen);
}

/**
 * Header: what is running and how it is doing. Settings details (window,
 * layers, backend, uptime, endpoint) are in the tooltip; the cards below own them.
 */
function renderHeader() {
  applyLayout();
  if (remoteMode) {
    if (!connectOpen) renderRemoteStatus();
    return;
  }
  const p = lastPayload;
  const modelEl = $('srvModel');
  if (!p || !modelEl) return;
  const hasModel = !!(p.state && p.state.selectedModelPath);
  // While a different model is selected but not loaded yet, the header keeps
  // naming what the server actually runs; the Model card shows the selection.
  const modelPending = serverRunning && pendingChanges.some(function (c) { return c.key === 'model'; });
  const name = hasModel ? (p.modelName || 'model') : 'No model selected';
  modelEl.innerHTML = modelPending ? modelTitleHtml(pendingModelChange().from || name) : modelTitleHtml(name);
  // Describe the running server; pending edits are listed in the pending bar.
  const L = (serverRunning && p.launchedSettings) || (p.state && p.state.loadSettings) || {};
  const caps = p.capabilities || {};
  const blocks = caps.blockCount || 0;
  const tip = [];
  if (p.modelNameRaw) tip.push('general.name: ' + p.modelNameRaw);
  if (serverRunning && lastEndpoint) tip.push(lastEndpoint + (lastPid ? ' · pid ' + lastPid : ''));
  if (serverRunning && p.startedAt) tip.push(fmtUptime(p.startedAt));
  if (hasModel) {
    tip.push('Context window ' + fmtTokShort(L.contextLength) + (caps.maxContextLength ? ' (model max ' + fmtTokShort(caps.maxContextLength) + ')' : ''));
    if (p.cpuOnly) {
      tip.push('CPU backend');
    } else {
      const ngl = L.gpuOffload >= 99 || (blocks && L.gpuOffload >= blocks) ? blocks : L.gpuOffload;
      const mainIdx = clampMainGpu(L.mainGpu, (gpuInfos || []).length || 1);
      const mainG = gpuInfos && gpuInfos[mainIdx];
      const where = (gpuInfos || []).length >= 2 && L.splitMode !== 'none'
        ? gpuInfos.map((g, i) => gpuLabel(g, i)).join(' + ')
        : (mainG ? gpuLabel(mainG, mainIdx) : 'GPU');
      tip.push((blocks ? ngl + '/' + blocks : ngl) + ' layers on ' + where);
    }
    if (L.speculativeMode && L.speculativeMode !== 'off') tip.push('Speculative: ' + specAcceptLabel(L.speculativeMode));
    const backend = (backendOptionsCache.find((o) => o.id === activeBackendId) || {});
    if (backend.label) tip.push('llama.cpp ' + backend.label + (backend.installedTag ? ' ' + backend.installedTag : ''));
  }
  if (lastMemoryView && lastMemoryView.devices && lastMemoryView.devices.length) {
    const d = lastMemoryView.devices[0];
    if (d.capacityBytes) tip.push('Estimate ' + fmtBytes(d.usedBytes) + ' / ' + fmtBytes(d.capacityBytes) + ' ' + (d.kind === 'ram' ? 'RAM' : d.label));
  }
  modelEl.title = tip.join('\n');
  renderModelStatus();
}

/** "Name · QUANT" with the quant dimmed. */
function modelTitleHtml(name) {
  const i = String(name).lastIndexOf(' · ');
  if (i <= 0) return escText(name);
  return escText(name.slice(0, i)) + ' <span class="q">· ' + escText(name.slice(i + 3)) + '</span>';
}

/** Model card: running / not loaded pill, reload prompt, and the collapsed summary line. */
function renderModelStatus() {
  const p = lastPayload;
  if (!p) return;
  const hasModel = !!(p.state && p.state.selectedModelPath);
  const modelPending = serverRunning && pendingChanges.some(function (c) { return c.key === 'model'; });
  // The header names the running model and shows its state; the Model card
  // only speaks up when its selection differs from what the server runs.
  let cls = '';
  let text = '';
  if (hasModel && modelPending) { cls = 'pending'; text = 'not loaded'; }
  const hero = document.querySelector<HTMLElement>('#modelCard .model-hero');
  if (hero) hero.classList.toggle('hidden', hasModel && !modelPending);
  ['modelStatus', 'modelClosedStatus'].forEach(function (id) {
    const el = $(id);
    if (!el) return;
    el.className = 'model-status' + (cls ? ' ' + cls : '') + (text ? '' : ' hidden');
    el.textContent = text;
    el.title = modelPending ? 'The server is still running ' + (pendingModelChange().from || 'another model') + '.' : '';
  });
  const bar = $('modelReloadBar');
  if (bar) {
    bar.classList.toggle('hidden', !modelPending);
    const t = $('modelReloadText');
    if (t && modelPending) {
      t.innerHTML = 'Server runs <b>' + escText(pendingModelChange().from || 'another model') + '</b>. Switch to the selection?';
    }
  }
  const closedReload = $('modelClosedReload');
  if (closedReload) closedReload.classList.toggle('hidden', !modelPending);

  const line = $('modelClosedLine');
  if (!line) return;
  line.classList.toggle('has-model', hasModel);
  const sel = p.state && p.state.selectedModelPath;
  const found = sel ? findLibraryQuant(sel) : null;
  const bits = [];
  if (found && found.quant.sizeLabel) bits.push(found.quant.sizeLabel);
  const L = (p.state && p.state.loadSettings) || {};
  if (L.mmprojPath) bits.push('vision');
  if (L.chatTemplateSource === 'builtin' && L.chatTemplateName) bits.push('template ' + L.chatTemplateName);
  else if (L.chatTemplateSource === 'file') bits.push('template file');
  else if (L.chatTemplateJinja === false) bits.push('jinja off');
  const caps = p.capabilities || {};
  if (caps.nextnPredictLayers > 0 || p.mtpSidecarPath) bits.push('MTP');
  else if (L.speculativeMode && L.speculativeMode !== 'off') bits.push(String(L.speculativeMode));
  const fit = sel ? lib.fits[sel] : '';
  if (fit) bits.push(fitText(fit));
  $('modelClosedMeta').textContent = bits.join(' · ');
  const barEl = $('modelClosedBar');
  if (barEl) {
    const ratio = sel ? lib.ratios[sel] : undefined;
    barEl.classList.toggle('hidden', typeof ratio !== 'number');
    if (typeof ratio === 'number') {
      barEl.className = 'fit-bar' + (fit ? ' ' + fit : '');
      barEl.title = fitText(fit) + ' · ~' + Math.round(ratio * 100) + '% of ' + (lib.budgetLabel || 'memory');
      barEl.firstElementChild.style.width = Math.max(2, Math.min(100, Math.round(ratio * 100))) + '%';
    }
  }
}
function pendingModelChange() {
  const change = pendingChanges.find(function (c) { return c.key === 'model'; });
  if (!change) return {};
  const running = allLibraryGroups().reduce(function (hit, g) {
    return hit || ((g.quants || []).find(function (q) { return q.loaded; }) ? { group: g, quant: g.quants.find(function (q) { return q.loaded; }) } : null);
  }, null);
  return { from: running ? shortModelLabel(running) : change.from, to: change.to };
}

function markDirtyIfRunning() {
  // Optimistic dirty UI while running — confirmed via silent save + statusPatch.
  if (!serverRunning) return;
  configDirty = true;
  updatePrimaryAction();
  const hint = $('dirtyHint');
  if (hint) hint.classList.remove('hidden');
  setServerCardKind('dirty');
  const line = $('statusLine');
  const dot = $('statusDot');
  if (line) line.className = 'status-line dirty';
  if (dot) dot.className = 'dot dirty';
}

function scheduleSaveLoad() {
  if (saveLoadTimer) clearTimeout(saveLoadTimer);
  highlightPreset();
  markDirtyIfRunning();
  refreshAdvancedMarks();
  saveLoadTimer = setTimeout(() => {
    vscode.postMessage({ type: 'saveLoad', payload: readLoad(), silent: true });
  }, 280);
}

function updateBackendUi() {
  updateBackendControls();
  renderBackendLine();
}

function llamaInstallRunning() {
  return !!(llamaInstall && llamaInstall.status === 'running');
}
function noteInstallStarting(message) {
  if (llamaInstallRunning()) return false;
  llamaInstall = { status: 'running', phase: 'resolve', message: message || 'Starting…' };
  renderLlamaInstall();
  return true;
}
function renderLlamaInstall() {
  const root = $('llamaInstallBox');
  if (!root) return;
  const job = llamaInstall;
  if (!job || job.status === 'idle' || job.status === 'done') {
    root.classList.add('hidden');
    root.innerHTML = '';
    return;
  }
  root.classList.remove('hidden');
  if (job.status === 'error' || job.status === 'cancelled') {
    const installed = (updateCheck && updateCheck.installedTag) || '';
    const title = job.status === 'cancelled' ? 'Install cancelled' : 'Download failed';
    const detail = job.status === 'cancelled'
      ? ('The previous build is unchanged' + (installed ? (' (' + installed + ')') : '') + '.')
      : String(job.error || job.message || 'Install failed').split('\n')[0];
    let html = '<div class="install-err">' + escText(title) + '</div>';
    html += '<div class="hint">' + escText(detail) + '</div>';
    html += '<div class="lib-job-meta"><span></span><span class="lib-actions" style="padding:0">';
    if (job.status === 'error') {
      html += '<button type="button" class="secondary small" data-act="install-retry">Retry</button>';
    }
    html += '<button type="button" class="secondary small" data-act="install-dismiss">Dismiss</button>';
    html += '</span></div>';
    root.innerHTML = html;
    return;
  }
  const phase = job.phase || 'resolve';
  const stepIndex = phase === 'download' ? 1 : (phase === 'extract' || phase === 'replace' ? 2 : 0);
  const steps = [
    {
      label: 'Resolve release',
      detail: stepIndex > 0 ? (job.releaseTag || 'Done') : 'Asking GitHub…',
    },
    {
      label: 'Download archive',
      detail: phase === 'download'
        ? (job.assetName || 'Downloading…')
        : (stepIndex > 1 ? (job.assetName || 'Done') : 'Waiting'),
    },
    {
      label: 'Extract and replace',
      detail: stepIndex < 2
        ? 'Server stays up until here'
        : (job.message || 'Replacing binaries…'),
    },
  ];
  let html = '';
  steps.forEach(function (step, index) {
    const state = index < stepIndex ? 'done' : (index === stepIndex ? 'now' : 'next');
    html += '<div class="install-step ' + state + '"><span>' + escText(step.label) + '</span><span>' + escText(step.detail) + '</span></div>';
  });
  if (phase === 'download' && job.percent != null && job.total > 0) {
    const pct = Math.max(0, Math.min(100, Math.round(job.percent)));
    html += '<div class="lib-bar"><span style="width:' + pct + '%"></span></div>';
    const bits = [];
    if (job.fileCount > 1) bits.push((job.fileIndex || 1) + ' of ' + job.fileCount);
    if (job.total) bits.push(fmtBytes(job.received || 0) + ' / ' + fmtBytes(job.total));
    if (job.bytesPerSec) bits.push(fmtBytes(job.bytesPerSec) + '/s');
    if (job.etaSeconds) bits.push(fmtEta(job.etaSeconds));
    html += '<div class="lib-job-meta"><span>' + escText(bits.join(' · ')) + '</span>';
  } else if (phase === 'download') {
    html += '<div class="lib-bar"><span class="indeterminate" style="width:35%"></span></div>';
    const received = job.received ? (fmtBytes(job.received) + ' received') : (job.message || 'Downloading…');
    html += '<div class="lib-job-meta"><span>' + escText(received) + '</span>';
  } else {
    html += '<div class="lib-bar"><span class="indeterminate" style="width:35%"></span></div>';
    html += '<div class="lib-job-meta"><span>' + escText(job.message || '') + '</span>';
  }
  if (phase === 'resolve' || phase === 'download') {
    html += '<span class="lib-actions" style="padding:0"><button type="button" class="secondary small" data-act="install-cancel">Cancel</button></span>';
  }
  html += '</div>';
  root.innerHTML = html;
}
function applyInstallLock() {
  const running = llamaInstallRunning();
  const installBtn = $('installLlamaBtn');
  const hint = $('backendHint');
  const sel = $('backendSelect');
  ['reinstallLlamaBtn', 'installByTagBtn', 'installArchiveBtn', 'checkUpdatesBtn'].forEach(function (id) {
    const el = $(id);
    if (el) el.disabled = running;
  });
  if (sel) sel.disabled = running;
  if (running && installBtn) {
    installBtn.disabled = true;
    installBtn.classList.add('hidden');
  }
  const boxVisible = !!(llamaInstall && llamaInstall.status && llamaInstall.status !== 'idle' && llamaInstall.status !== 'done');
  if (hint && boxVisible) hint.classList.add('hidden');
}
function updateBackendControls() {
  const sel = $('backendSelect');
  const installBtn = $('installLlamaBtn');
  const hint = $('backendHint');
  const reinstallBtn = $('reinstallLlamaBtn');
  if (!sel || !installBtn || !hint) return;
  try {
  syncSplitSeg();
  const selectedOpt = backendOptionsCache.find((o) => o.id === sel.value);
  const anyInstalled = backendOptionsCache.some((o) => o.installed);
  const installedTag = (selectedOpt && selectedOpt.installedTag) || updateCheck.installedTag;
  const latestTag = updateCheck.latestTag;
  const selectedIsActive = !!(selectedOpt && selectedOpt.active);
  const isPathBackend = sel.value === 'path';

  if (reinstallBtn) {
    reinstallBtn.classList.toggle(
      'hidden',
      isPathBackend || !(selectedOpt && selectedOpt.installed && selectedIsActive)
    );
    reinstallBtn.textContent = installedTag
      ? ('Reinstall ' + installedTag)
      : 'Reinstall current release';
  }

  if (isPathBackend) {
    if (selectedOpt && selectedOpt.installed) {
      hint.textContent =
        'Using llama-server from PATH' +
        (selectedOpt.installedTag ? (' (' + selectedOpt.installedTag + ')') : '') +
        '. Managed by your system package manager.';
    } else {
      hint.textContent =
        (selectedOpt && selectedOpt.reason) ||
        'llama-server not found on PATH. Install a system package (e.g. nixpkgs llama-cpp).';
    }
    installBtn.classList.add('hidden');
    installBtn.disabled = true;
    return;
  }

  if (selectedOpt && selectedOpt.reason && !selectedOpt.available) {
    hint.textContent = selectedOpt.reason;
    installBtn.classList.add('hidden');
    installBtn.disabled = true;
    return;
  }

  if (selectedOpt && !selectedOpt.installed) {
    hint.textContent = 'Not installed — selecting this backend will download it.';
    installBtn.textContent = 'Install ' + (selectedOpt.label || selectedOpt.id);
    installBtn.disabled = false;
    installBtn.classList.remove('hidden');
    installBtn.dataset.action = 'install';
    return;
  }

  if (!selectedOpt) {
    hint.textContent = anyInstalled ? '' : 'No llama.cpp backend installed yet.';
    installBtn.textContent = 'Install llama.cpp';
    installBtn.disabled = false;
    installBtn.classList.toggle('hidden', !!anyInstalled);
    installBtn.dataset.action = 'install';
    return;
  }

  // Installed backend selected
  if (!selectedIsActive) {
    hint.textContent = 'Switching applies immediately. Restart the server if it is running.';
    installBtn.classList.add('hidden');
    return;
  }

  if (updateCheck.pending) {
    hint.textContent = (installedTag ? (installedTag + ' · ') : '') + 'Checking for updates…';
    installBtn.classList.add('hidden');
    return;
  }

  if (updateCheck.checkFailed) {
    hint.textContent = (installedTag ? (installedTag + ' · ') : '') + 'Could not check for updates.';
    installBtn.textContent = 'Check for updates';
    installBtn.disabled = false;
    installBtn.classList.remove('hidden');
    installBtn.dataset.action = 'check';
    return;
  }

  if (updateCheck.updateAvailable && latestTag) {
    hint.textContent = '';
    installBtn.textContent = 'Upgrade to ' + latestTag;
    installBtn.disabled = false;
    installBtn.classList.remove('hidden');
    installBtn.dataset.action = 'upgrade';
    return;
  }

  hint.textContent = '';
  installBtn.classList.add('hidden');
  installBtn.dataset.action = '';
  } finally {
    applyInstallLock();
  }
}

/** "Vulkan · b11163" once, plus the alternatives as a summary of the fold. */
function renderBackendLine() {
  const line = $('backendLine');
  const sum = $('backendSum');
  const active = backendOptionsCache.find((o) => o.id === activeBackendId) || backendOptionsCache.find((o) => o.active);
  if (line) {
    const tag = (active && active.installedTag) || updateCheck.installedTag || '';
    const upToDate = !updateCheck.pending && !updateCheck.checkFailed && !updateCheck.updateAvailable && tag;
    line.textContent = active
      ? active.label + (tag ? ' · ' + tag : '') + (upToDate ? ' · ✓ latest' : '')
      : 'not installed';
  }
  if (sum) {
    sum.textContent = backendOptionsCache
      .filter((o) => o.available && o.id !== 'path')
      .map((o) => o.label + (o.id === activeBackendId ? ' ✓' : ''))
      .join(' · ');
  }
  const hint = $('backendHint');
  if (hint) hint.classList.toggle('hidden', !hint.textContent);
}









function liveLayersOnGpu(L?) {
  const nLayers = Math.max(1, (memInputs && memInputs.blockCount) || 1);
  const ngl = L && Number.isFinite(Number(L.gpuOffload)) ? Number(L.gpuOffload) : Number($('gpuOffload') && $('gpuOffload').value);
  if (!Number.isFinite(ngl) || ngl >= 99) return nLayers;
  return Math.min(nLayers, Math.max(0, Math.round(ngl)));
}

function liveMassOpts(L?) {
  const nCpuMoe = L && Number.isFinite(Number(L.nCpuMoe)) ? Number(L.nCpuMoe) : Number($('nCpuMoe') && $('nCpuMoe').value) || 0;
  const nCpuFfn = L && Number.isFinite(Number(L.nCpuFfn)) ? Number(L.nCpuFfn) : Number($('nCpuFfn') && $('nCpuFfn').value) || 0;
  return {
    isMoe: !!(memInputs && memInputs.isMoe),
    nCpuMoe: nCpuMoe,
    moeExpertShare: memInputs ? moeExpertShareOf(memInputs) : 0,
    nCpuFfn: nCpuFfn,
    denseFfnShare: memInputs ? denseFfnShareOf(memInputs) : 0,
  };
}





function readMainGpuIndex() {
  const n = (gpuInfos && gpuInfos.length) || 1;
  return clampMainGpu(($('mainGpu') && $('mainGpu').value) || 0, n);
}

function readTensorSplitFromUi() {
  const n = (gpuInfos && gpuInfos.length) || 1;
  if (n < 2) return '';
  const raw = $('tensorSplitRange') && $('tensorSplitRange').value;
  const pct = Number(raw);
  const share = (Number.isFinite(pct) ? pct : 50) / 100;
  const splitMode = ($('splitMode') && $('splitMode').value) || 'layer';
  const nLayers = Math.max(1, (memInputs && memInputs.blockCount) || 1);
  return tensorSplitForTargetWeightShareLive(
    share,
    readMainGpuIndex(),
    n,
    nLayers,
    liveLayersOnGpu(),
    splitMode,
    liveMassOpts(),
    gpuInfos
  );
}

function syncTensorSplitPctLabel() {
  const range = $('tensorSplitRange');
  const lbl = $('tensorSplitPct');
  const hint = $('tensorSplitHint');
  if (!range || !lbl) return;
  if ($('splitMode') && $('splitMode').value === 'none') {
    lbl.textContent = '100%';
    if (hint) hint.textContent = '';
    return;
  }
  lbl.textContent = String(range.value) + '%';
  if (hint) {
    const n = (gpuInfos && gpuInfos.length) || 1;
    const emitted = n >= 2 ? readTensorSplitFromUi() : '';
    hint.textContent = emitted ? ('GPU-resident weights · --tensor-split ' + emitted) : '';
  }
}

function syncTensorSplitEnabled() {
  const ts = $('tensorSplitRange');
  const none = $('splitMode') && $('splitMode').value === 'none';
  const cpu = cpuOnlyLive();
  if (ts) ts.disabled = cpu || none;
  // Weight share means nothing with split None — hide it instead of greying it out.
  const details = $('splitDetails');
  if (details) details.classList.toggle('hidden', !!(cpu || none));
  syncSplitSeg();
  syncTensorSplitPctLabel();
  renderOffloadSum();
}

function syncSplitSeg() {
  const mode = ($('splitMode') && $('splitMode').value) || 'layer';
  const backendSel = $('backendSelect');
  const sycl = (backendSel && backendSel.value === 'sycl') || activeBackendId === 'sycl';
  document.querySelectorAll<HTMLElement>('#splitSeg .seg-btn').forEach((b) => {
    b.classList.toggle('hidden', b.dataset.split === 'row' && sycl);
    b.classList.toggle('active', b.dataset.split === mode);
  });
  const rowOpt = document.querySelector<HTMLOptionElement>('#splitMode option[value="row"]');
  if (rowOpt) rowOpt.disabled = !!sycl;
}

/** One-line state of the Multi-GPU & CPU offload fold, visible while collapsed. */
function renderOffloadSum() {
  const el = $('offloadSum');
  if (!el) return;
  const bits = [];
  const n = (gpuInfos && gpuInfos.length) || 0;
  const mode = ($('splitMode') && $('splitMode').value) || 'layer';
  if (cpuOnlyLive()) {
    bits.push('CPU backend');
  } else if (n >= 2) {
    const main = clampMainGpu(readMainGpuIndex(), n);
    bits.push(mode === 'none'
      ? gpuLabel(gpuInfos[main], main) + ' only'
      : 'split ' + mode + ' · main ' + gpuLabel(gpuInfos[main], main) + ' ' + (($('tensorSplitPct') && $('tensorSplitPct').textContent) || ''));
  } else if (n === 1) {
    bits.push(gpuLabel(gpuInfos[0], 0));
  }
  if (modelIsMoe) bits.push('CPU MoE ' + (Number($('nCpuMoe').value) || 0));
  else bits.push('CPU FFN ' + (Number($('nCpuFfn').value) || 0));
  el.textContent = bits.join(' · ');
}

/** GPU layers: "All" hides the slider; "Custom" shows it. */
function syncNglMode() {
  const blocks = Math.max(1, modelBlockCount || 1);
  const ngl = Number($('gpuOffload').value);
  const all = !nglCustomOpen && (ngl >= blocks || ngl >= 99);
  $('nglAll').classList.toggle('active', all);
  $('nglAll').textContent = 'All (' + blocks + ')';
  $('nglCustom').classList.toggle('active', !all);
  $('nglCustomRow').classList.toggle('hidden', all);
}

function fillMainGpuSelect(selected) {
  const sel = $('mainGpu');
  if (!sel || document.activeElement === sel) return;
  const gpus = gpuInfos || [];
  const n = Math.max(gpus.length, 1);
  const want = clampMainGpu(selected, n);
  sel.innerHTML = '';
  for (let i = 0; i < n; i++) {
    const opt = document.createElement('option');
    opt.value = String(i);
    const g = gpus[i];
    opt.textContent = g
      ? (gpuLabel(g, i) + (g.totalBytes ? ' · ' + fmtBytes(g.totalBytes) : ''))
      : ('GPU ' + i);
    sel.appendChild(opt);
  }
  sel.value = String(want);
}

function cpuOnlyLive() {
  const sel = $('backendSelect');
  return !!(sel && sel.value === 'cpu');
}









function specAcceptLabel(mode) {
  if (!mode) return 'draft';
  if (mode === 'dflash') return 'DFlash';
  if (mode === 'ngram-dflash') return 'N-gram+DFlash';
  if (mode === 'ngram') return 'N-gram';
  if (mode === 'ngram-mtp') return 'N-gram+MTP';
  return 'MTP';
}

/**
 * "This chat": how much of the context window the last request used, by
 * part. Drawn in the header strip and in the Performance card from the same
 * data; the denominator is always the server's context window per slot.
 */
function renderContextStack(perf) {
  const breakdown = perf && perf.contextBreakdown;
  const ctxPct = typeof perf?.contextPct === 'number' ? perf.contextPct : undefined;
  const ctxLevel = perf?.contextLevel || 'ok';
  const promptTokens = typeof perf?.promptTokens === 'number' ? perf.promptTokens : undefined;
  const contextLimit = typeof perf?.contextLimit === 'number' ? perf.contextLimit : undefined;
  const approx = perf?.contextEstimated ? '≈' : '';
  const measured = typeof promptTokens === 'number' && typeof contextLimit === 'number' && typeof ctxPct === 'number';
  const subText = measured
    ? approx + fmtTokK(promptTokens) + ' of ' + fmtTokK(contextLimit) + ' window · ' + Math.round(ctxPct) + '%' +
      (ctxLevel === 'critical' ? ' · nearly full' : ctxLevel === 'warn' ? ' · running low' : '')
    : 'send a chat to measure';
  const aria = measured
    ? "This chat uses " + approx + fmtTokK(promptTokens) + ' of the ' + fmtTokK(contextLimit) + '-token context window'
    : 'This chat: not measured yet';
  const color = ctxLevel === 'critical' ? '#f85149' : ctxLevel === 'warn' ? '#d29922' : '';

  const scale = Math.max(1, (breakdown && breakdown.limitTokens) || contextLimit || 1);
  const used = breakdown && Array.isArray(breakdown.segments)
    ? breakdown.segments.filter((seg) => seg && seg.key !== 'free' && seg.tokens > 0)
    : [];
  const draw = (stackId, labelId, subId) => {
    const stack = $(stackId);
    const label = $(labelId);
    const sub = $(subId);
    if (!stack || !sub) return;
    stack.className = 'ctx-stack' + (stack.classList.contains('thin') ? ' thin' : '') +
      (ctxLevel === 'warn' || ctxLevel === 'critical' ? ' ' + ctxLevel : '');
    if (label) label.style.color = color;
    sub.textContent = subText;
    stack.setAttribute('aria-label', aria);
    stack.innerHTML = '';
    for (const seg of used) {
      const el = document.createElement('span');
      el.className = 'seg-' + seg.key;
      el.style.width = Math.max(0.4, (seg.tokens / scale) * 100) + '%';
      const pct = Math.round((seg.tokens / scale) * 1000) / 10;
      el.title = seg.label + ': ≈' + Number(seg.tokens).toLocaleString() + ' tok (' + pct + '% of the window)';
      stack.appendChild(el);
    }
  };
  draw('ctxStack', 'ctxLabel', 'ctxSub');
  draw('liveChatStack', 'liveChatLabel', 'liveChatSub');

  // Largest first, with counts, so "tools + results eat ⅔" is readable without hovering.
  const legend = $('ctxLegend');
  if (legend) {
    legend.textContent = '';
    for (const seg of used.slice().sort((x, y) => y.tokens - x.tokens)) {
      const span = document.createElement('span');
      const swatch = document.createElement('i');
      swatch.className = 'seg-' + seg.key;
      span.appendChild(swatch);
      span.appendChild(document.createTextNode((seg.label || seg.key) + ' ' + fmtTokK(seg.tokens)));
      legend.appendChild(span);
    }
  }
}

function renderPerfStats(perf, perfLines) {
  const metricsEl = $('perfMetrics');
  const tableEl = $('perfHistoryTable');
  const p = perf || {};
  const stale = !!(p.generating && p.showingPreviousCall);
  const rows = Array.isArray(p.history) ? p.history.filter(function (h) { return h && typeof h === 'object'; }) : [];
  const last = rows[0] || {};

  if (tableEl) {
    if (!rows.length) {
      tableEl.innerHTML = '<div class="muted" style="padding:6px 8px">No completed calls yet</div>';
    } else {
      const body = rows.map(function (h, i) {
        const when = h.finishedAt ? new Date(h.finishedAt).toLocaleTimeString() : '';
        const genH = fmtRate(h.genTokPerSec);
        const promptH = fmtRate(h.promptTokPerSec);
        const reuse = typeof h.cacheHitPct === 'number' ? h.cacheHitPct.toFixed(0) + '%' : '—';
        return '<tr><td>' + (i === 0 ? when + ' <span class="muted">latest</span>' : when) +
          '</td><td>' + (genH || '—') +
          '</td><td>' + (promptH || '—') +
          '</td><td>' + reuse + '</td></tr>';
      }).join('');
      tableEl.innerHTML = '<table><thead><tr><th>Call</th><th>Gen</th><th>Prompt</th><th>Reuse</th></tr></thead><tbody>' +
        body + '</tbody></table>';
    }
  }

  const cacheHitPct = pickNum(p.cacheHitPct, last.cacheHitPct);
  const completionTokens = pickNum(p.completionTokens, last.completionTokens);
  const draftAcceptancePct = pickNum(p.draftAcceptancePct, last.draftAcceptancePct);
  const durationMs = (!p.generating && p.finishedAt && p.startedAt)
    ? (p.finishedAt - p.startedAt)
    : last.durationMs;
  const genNow = p.generating ? p.genTokPerSec : pickNum(p.genTokPerSec, last.genTokPerSec);
  const promptNow = pickNum(p.promptTokPerSec, last.promptTokPerSec);

  // Oldest first for the sparklines; each tile scales its own series.
  const chrono = rows.slice().reverse();
  const genSeries = chrono.map((h) => h.genTokPerSec);
  const promptSeries = chrono.map((h) => h.promptTokPerSec);
  const reuseSeries = chrono.map((h) => h.cacheHitPct);
  const setSpark = (id, values, color) => {
    const el = $(id);
    if (el) el.innerHTML = sparklineSvg(values, color);
  };
  setSpark('kpiGenSpark', genSeries, '#58a6ff');
  setSpark('kpiPromptSpark', promptSeries, '#d29922');
  setSpark('kpiReuseSpark', reuseSeries, '#3fb950');
  setSpark('liveGenSpark', genSeries, '#58a6ff');
  setSpark('livePromptSpark', promptSeries, '#d29922');
  setSpark('liveReuseSpark', reuseSeries, '#3fb950');

  const setKpi = (id, text, muted) => {
    const el = $(id);
    if (!el) return;
    el.textContent = text;
    el.classList.toggle('muted', !!muted);
  };
  setKpi('kpiGen', fmtRate(genNow) || '—', stale);
  setKpi('kpiPrompt', fmtRate(promptNow) || '—', stale);
  setKpi('kpiReuse', typeof cacheHitPct === 'number' ? Math.round(cacheHitPct) + '%' : '—', stale);
  // Header strip: the same three numbers, shown while a local server runs.
  setKpi('liveGen', fmtRate(genNow) || '—', stale);
  setKpi('livePrompt', fmtRate(promptNow) || '—', stale);
  setKpi('liveReuse', typeof cacheHitPct === 'number' ? Math.round(cacheHitPct) + '%' : '—', stale);
  const specOn = p.speculativeMode
    ? p.speculativeMode !== 'off'
    : typeof draftAcceptancePct === 'number';
  setKpi('kpiSpec', specOn && typeof draftAcceptancePct === 'number' ? Math.round(draftAcceptancePct) + '%' : (specOn ? '—' : 'off'), stale);
  const specLabel = $('kpiSpecLabel');
  if (specLabel) specLabel.textContent = specOn ? specAcceptLabel(p.speculativeMode) + ' accept' : 'speculative';
  // With the header strip showing gen / prompt / reuse, the card keeps only
  // what the strip lacks: the draft acceptance as one line.
  const specLine = $('perfSpecLine');
  if (specLine) {
    const acc = specOn && typeof draftAcceptancePct === 'number' ? Math.round(draftAcceptancePct) + '%' : '';
    specLine.innerHTML = acc
      ? '<span class="muted">' + escText(specAcceptLabel(p.speculativeMode)) + ' accept</span> ' + acc
      : '';
    specLine.classList.toggle('hidden', !acc);
  }
  const windowEl = $('perfWindow');
  if (windowEl) {
    const base = rows.length ? 'last ' + rows.length + ' request' + (rows.length === 1 ? '' : 's') : '';
    windowEl.textContent = remoteMode
      ? (base ? base + ' · incl. network' : 'this window · incl. network')
      : base;
  }
  const histSum = $('perfHistSum');
  if (histSum) histSum.textContent = rows.length ? rows.length + ' request' + (rows.length === 1 ? '' : 's') : 'per-request table';

  const bits = [];
  if (p.generating) {
    const live = fmtRate(p.genTokPerSec);
    bits.push('<span class="ok">● Generating…' + (live ? ' ' + live + ' tok/s' : '') + '</span>');
  }
  if (typeof completionTokens === 'number') {
    const dur = typeof durationMs === 'number' && durationMs > 0 && !p.generating
      ? ' in ' + (durationMs / 1000).toFixed(1) + ' s'
      : '';
    bits.push('<span class="muted">Last reply: ' + fmtNum(completionTokens) + ' tok' + dur + '</span>');
  }
  if (metricsEl) {
    metricsEl.innerHTML = bits.length ? bits.join('') : (rows.length ? '' : 'No generation yet');
    metricsEl.title = Array.isArray(perfLines) ? perfLines.join('\n') : '';
  }
}

/** A curated model mode owns temperature / top_p / top_k: show what it sends instead of inert inputs. */
function renderModeOverrideHint(mode) {
  const strip = $('modeStrip');
  if (!strip) return;
  document.querySelectorAll<HTMLElement>('#copilotCard .samp-free').forEach((el) => el.classList.toggle('hidden', !!mode));
  if (!mode) {
    strip.classList.add('hidden');
    strip.title = '';
    return;
  }
  strip.classList.remove('hidden');
  $('modeStripName').textContent = mode.defaultMode;
  const values = $('modeStripValues');
  values.textContent = '';
  for (const text of ['T ' + mode.temperature, 'p ' + mode.topP, 'k ' + mode.topK]) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = text;
    values.appendChild(chip);
  }
  strip.title =
    mode.familyLabel + ' model detected — the Model Mode picker in Copilot Chat sets temperature, top_p and top_k per request. "' +
    mode.defaultMode + '" is the default and sends temperature ' + mode.temperature + ', top_p ' + mode.topP +
    ', top_k ' + mode.topK + '. Max tokens, the penalties below, and the overthinking penalty still apply.';
}

function syncMmprojOffloadUi(cpuOnly) {
  const el = $('mmprojOffloadToGpu');
  if (!el) return;
  const row = $('mmprojAttach');
  const projPath = (row && (row.dataset.path || '').trim()) || '';
  el.disabled = !!cpuOnly || !projPath;
  const onGpu = el.checked && !cpuOnly;
  document.querySelectorAll<HTMLButtonElement>('#mmprojSeg .seg-btn').forEach(function (b) {
    b.classList.toggle('active', (b.dataset.mm === 'gpu') === onGpu);
    b.disabled = !!cpuOnly;
  });
}

function applyCpuOnlyUi(cpuOnly) {
  $('gpuOffload').disabled = cpuOnly;
  $('gpuOffloadRange').disabled = cpuOnly;
  $('offloadKvCacheToGpu').disabled = cpuOnly;
  syncMmprojOffloadUi(cpuOnly);
  $('gpuOffloadHint').textContent = cpuOnly
    ? 'CPU backend — GPU layers are ignored; everything runs in system RAM.'
    : '';
  $('gpuOffloadHint').classList.toggle('hidden', !cpuOnly);
  $('nglAll').disabled = cpuOnly;
  $('nglCustom').disabled = cpuOnly;
  $('gpuOffloadRow').style.opacity = cpuOnly ? '0.55' : '1';
  syncNglMode();

  // --n-cpu-moe only splits experts GPU↔CPU; meaningless when everything is already on CPU.
  $('nCpuMoe').disabled = cpuOnly;
  $('nCpuMoeRange').disabled = cpuOnly;
  const showMoe = modelIsMoe && !cpuOnly;
  $('moeFields').classList.toggle('hidden', !showMoe);
  if (modelIsMoe) {
    $('moeHint').textContent = cpuOnly
      ? 'CPU backend — experts already run in system RAM; CPU MoE layers (--n-cpu-moe) does not apply.'
      : moeHintDefault;
  }
  // --n-cpu-ffn is the dense counterpart: hidden for MoE models (experts use
  // --n-cpu-moe) and on the CPU backend.
  $('nCpuFfn').disabled = cpuOnly;
  $('nCpuFfnRange').disabled = cpuOnly;
  $('ffnFields').classList.toggle('hidden', modelIsMoe || cpuOnly);
  if (!modelIsMoe) {
    $('ffnHint').textContent = cpuOnly
      ? 'CPU backend — weights already run in system RAM; CPU FFN layers (--n-cpu-ffn) does not apply.'
      : ffnHintDefault;
  }
  $('moeRow').classList.toggle('hidden', cpuOnly);

  const dual = $('dualGpuRow');
  if (dual) {
    const showDual = !cpuOnly && gpuInfos && gpuInfos.length >= 2;
    dual.classList.toggle('hidden', !showDual);
    const sm = $('splitMode');
    const mg = $('mainGpu');
    if (sm) sm.disabled = cpuOnly;
    if (mg) mg.disabled = cpuOnly;
    syncTensorSplitEnabled();
    const hint = $('dualGpuHint');
    if (hint && showDual) {
      hint.textContent = splitModeIsNone()
        ? 'Every GPU layer stays on the Main GPU; the other cards stay free.'
        : 'Pick the faster card as Main GPU, then give it more of the weights.';
      hint.title = gpuInfos.map((g, i) => gpuFullLabel(g, i) + ' · ' + fmtBytes(g.totalBytes)).join('\n');
    }
  }
  renderOffloadSum();
}

function splitModeIsNone() {
  return !!( $('splitMode') && $('splitMode').value === 'none' );
}

function applyDualGpuUi(L) {
  fillMainGpuSelect(L.mainGpu ?? 0);
  const n = (gpuInfos && gpuInfos.length) || 0;
  if (n < 2) {
    syncTensorSplitPctLabel();
    return false;
  }
  let split = L.tensorSplit || '';
  if (isLegacyGpu0FirstSplit(split) && clampMainGpu(L.mainGpu ?? 0, n) > 0) {
    const shares = tensorSplitShares(split, gpuInfos);
    split = tensorSplitForMainShare(Math.max.apply(null, shares), L.mainGpu ?? 0, n);
  }
  const nLayers = Math.max(1, (memInputs && memInputs.blockCount) || 1);
  const share = mainWeightShareFromSplitLive(
    split,
    L.mainGpu ?? 0,
    gpuInfos,
    L.splitMode,
    liveMassOpts(L),
    nLayers,
    liveLayersOnGpu(L)
  );
  const pct = Math.min(90, Math.max(10, Math.round(share * 100)));
  const range = $('tensorSplitRange');
  if (range && document.activeElement !== range) range.value = String(pct);
  syncTensorSplitEnabled();
  return isLegacyGpu0FirstSplit(L.tensorSplit) && clampMainGpu(L.mainGpu ?? 0, n) > 0;
}

// Memory: the extension runs core's estimator on the form values and sends
// back a verdict. The webview has no estimator of its own.
let estimateSeq = 0;
let renderedSeq = 0;
let estimateTimer = null;
function requestEstimate() {
  if (estimateTimer) clearTimeout(estimateTimer);
  estimateTimer = setTimeout(() => {
    estimateTimer = null;
    estimateSeq += 1;
    vscode.postMessage({ type: 'estimate', seq: estimateSeq, payload: readLoad(), cpuOnly: cpuOnlyLive() });
  }, 90);
}
function refreshMemoryLive() { requestEstimate(); }
function scheduleLiveMemory() { requestEstimate(); }

const SEG_LABEL = { weights: 'Weights', vision: 'Vision', draft: 'Spec', kv: 'KV', overhead: 'Overhead' };

function memBar(device, targetFree) {
  const bar = document.createElement('div');
  bar.className = 'mem-stack';
  const scale = Math.max(device.capacityBytes || 0, device.usedBytes || 0) || 1;
  for (const seg of device.segments || []) {
    if (!(seg.bytes > 0)) continue;
    const el = document.createElement('span');
    el.className = 'seg-' + seg.key;
    el.style.width = Math.max(0.5, (seg.bytes / scale) * 100) + '%';
    el.title = seg.label + ': ~' + fmtBytes(seg.bytes);
    bar.appendChild(el);
  }
  if (device.kind !== 'ram' && device.capacityBytes > targetFree && targetFree > 0) {
    const mark = document.createElement('i');
    mark.className = 'target';
    mark.style.left = (((device.capacityBytes - targetFree) / scale) * 100) + '%';
    mark.title = 'Target: keep ' + fmtBytes(targetFree) + ' free';
    bar.appendChild(mark);
  }
  return bar;
}

function renderMemoryView(view) {
  const verdict = $('memVerdict');
  const devicesEl = $('memDevices');
  const legend = $('memLegend');
  const fixes = $('memFixes');
  devicesEl.textContent = '';
  fixes.textContent = '';
  legend.textContent = '';
  $('memNotes').textContent = '';
  $('memLines').textContent = '';
  if (!view) {
    verdict.className = 'verdict unknown';
    verdict.textContent = 'Select a model to estimate VRAM / RAM use.';
    legend.classList.add('hidden');
    fixes.classList.add('hidden');
    $('memFootnote').textContent = '';
    renderHeader();
    return;
  }
  lastMemoryView = view;
  verdict.className = 'verdict ' + view.level;
  const icon = view.level === 'good' ? '✓ ' : view.level === 'spill' ? '✕ ' : view.level === 'tight' ? '⚠ ' : '';
  verdict.textContent = icon + view.headline;

  const totals = {};
  for (const d of view.devices || []) {
    const row = document.createElement('div');
    row.className = 'mem-dev';
    const head = document.createElement('div');
    head.className = 'mem-chart-title';
    const name = document.createElement('span');
    name.textContent = d.label + (d.capacityBytes && d.kind !== 'ram' ? ' · ' + fmtBytes(d.capacityBytes) : '');
    name.title = d.fullLabel;
    const sub = document.createElement('span');
    sub.className = 'sub' + (d.level === 'spill' ? ' over' : d.level === 'tight' ? ' warn' : '');
    let text = '~' + fmtBytes(d.usedBytes) + (d.capacityBytes ? ' / ' + fmtBytes(d.capacityBytes) : '');
    if (serverRunning && d.kind !== 'ram' && d.liveUsedBytes > 0) {
      text += ' · live ' + fmtBytes(d.liveUsedBytes);
    }
    sub.textContent = text;
    sub.title = 'Estimate at full context' + (d.liveUsedBytes > 0 ? '. Live = what the driver reports now.' : '');
    head.appendChild(name);
    head.appendChild(sub);
    row.appendChild(head);
    row.appendChild(memBar(d, view.targetFreeBytes));
    devicesEl.appendChild(row);
    for (const s of d.segments || []) {
      if (s.bytes > 0) totals[s.key] = (totals[s.key] || 0) + s.bytes;
    }
  }
  for (const key of ['weights', 'vision', 'draft', 'kv', 'overhead']) {
    if (!totals[key]) continue;
    const span = document.createElement('span');
    const swatch = document.createElement('i');
    swatch.className = 'seg-' + key;
    span.appendChild(swatch);
    span.appendChild(document.createTextNode(SEG_LABEL[key] + ' ' + fmtBytes(totals[key])));
    legend.appendChild(span);
  }
  legend.classList.toggle('hidden', !Object.keys(totals).length);
  $('memFootnote').textContent = view.footnote || '';

  const fixList = Array.isArray(view.fixes) ? view.fixes : [];
  fixes.classList.toggle('hidden', !fixList.length);
  if (fixList.length) {
    const lead = document.createElement('span');
    lead.className = 'hint';
    lead.textContent = view.level === 'spill' ? 'Make it fit:' : 'Get to ' + fmtBytes(view.targetFreeBytes) + ' free:';
    fixes.appendChild(lead);
    for (const f of fixList) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip fix ' + f.level;
      b.textContent = f.label + ' → ' + f.detail;
      b.title = 'Apply ' + Object.entries(f.patch).map(([k, v]) => k + ' = ' + (v === '' ? 'auto' : v)).join(', ');
      b.addEventListener('click', () => applyLoadPatch(f.patch));
      fixes.appendChild(b);
    }
  }
  for (const [targetId, list] of [['memNotes', view.notes], ['memLines', view.lines]]) {
    const el = $(targetId);
    for (const line of list || []) {
      const div = document.createElement('div');
      div.textContent = String(line);
      el.appendChild(div);
    }
  }
  renderHeader();
}

/** Apply a set of load settings in one round-trip (fix chips, "fit" preset). */
function applyLoadPatch(patch, opts?) {
  if (saveLoadTimer) {
    clearTimeout(saveLoadTimer);
    saveLoadTimer = null;
  }
  markDirtyIfRunning();
  vscode.postMessage({
    type: 'applyLoadPatch',
    payload: Object.assign({}, readLoad(), patch),
    fitContext: !!(opts && opts.fitContext),
    cpuOnly: cpuOnlyLive(),
  });
}

/**
 * Keep the number input in sync with a range. align (context length) uses
 * step=1 on the range so the thumb tracks the pointer on a 256k-wide axis;
 * the saved value still snaps. Writing range.value during input fights
 * native dragging in Chromium.
 */
function bindRange(numId, rangeId, align?) {
  const num = $(numId);
  const range = $(rangeId);
  if (!num || !range) return;
  function snap(v) {
    const min = Number(range.min);
    const max = Number(range.max);
    let n = Number(v);
    if (!Number.isFinite(n)) n = Number.isFinite(min) ? min : 0;
    if (align > 0) n = Math.round(n / align) * align;
    if (Number.isFinite(min)) n = Math.max(min, n);
    if (Number.isFinite(max) && max >= min) n = Math.min(max, n);
    return n;
  }
  num.addEventListener('input', () => {
    range.value = String(snap(num.value));
    scheduleLiveMemory();
  });
  range.addEventListener('input', () => {
    num.value = String(snap(range.value));
    scheduleLiveMemory();
  });
  if (align > 0) {
    range.addEventListener('change', () => {
      const n = snap(range.value);
      num.value = String(n);
      range.value = String(n);
    });
  }
}
// Context: logarithmic slider (0–1000) that snaps to the common sizes; the
// number box keeps any exact value.
const CTX_SNAPS = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
let ctxMin = 2048;
let ctxMax = 131072;
function ctxToSlider(ctx) {
  const c = Math.min(ctxMax, Math.max(ctxMin, Number(ctx) || ctxMin));
  return Math.round((1000 * Math.log(c / ctxMin)) / Math.log(ctxMax / ctxMin));
}
function sliderToCtx(v) {
  const raw = ctxMin * Math.pow(ctxMax / ctxMin, Number(v) / 1000);
  for (const s of CTX_SNAPS.concat([ctxMax])) {
    if (s <= ctxMax && Math.abs(raw - s) / s < 0.06) return s;
  }
  return Math.max(ctxMin, Math.min(ctxMax, Math.round(raw / 256) * 256));
}
function syncCtxK() {
  const el = $('ctxK');
  if (el) el.textContent = '≈' + fmtTokShort(Number($('contextLength').value));
}
function setContextField(ctx) {
  if (!fieldFocused('contextLength')) setField('contextLength', ctx);
  if (!fieldFocused('contextLengthRange')) $('contextLengthRange').value = String(ctxToSlider(ctx));
  syncCtxK();
}
function renderCtxTicks() {
  const ticks = $('ctxTicks');
  if (!ticks) return;
  ticks.textContent = '';
  const marks = CTX_SNAPS.filter((s) => s >= ctxMin && s < ctxMax * 0.97).concat([ctxMax]);
  for (const s of marks) {
    const t = document.createElement('span');
    t.textContent = fmtTokShort(s);
    t.style.left = (ctxToSlider(s) / 10) + '%';
    ticks.appendChild(t);
  }
}
(function bindContext() {
  const num = $('contextLength');
  const range = $('contextLengthRange');
  num.addEventListener('input', () => {
    range.value = String(ctxToSlider(num.value));
    syncCtxK();
    scheduleLiveMemory();
  });
  range.addEventListener('input', () => {
    num.value = String(sliderToCtx(range.value));
    syncCtxK();
    scheduleLiveMemory();
  });
})();
bindRange('gpuOffload', 'gpuOffloadRange');
bindRange('cpuThreads', 'cpuThreadsRange');
bindRange('nCpuMoe', 'nCpuMoeRange');
bindRange('nCpuFfn', 'nCpuFfnRange');
bindRange('ngramSizeN', 'ngramSizeNRange');
const tsRange = $('tensorSplitRange');
if (tsRange) tsRange.addEventListener('input', syncTensorSplitPctLabel);
$('offloadKvCacheToGpu').addEventListener('change', refreshMemoryLive);
$('evalBatchSize').addEventListener('input', refreshMemoryLive);
$('physicalBatchSize').addEventListener('input', refreshMemoryLive);

/** Higher = more precise. Used to flag lopsided K/V pairs; < 2 = quantized. */
const KV_PRECISION_RANK = {
  q4_0: 0,
  iq4_nl: 0.5,
  q4_1: 0.6,
  q5_0: 0.7,
  q5_1: 0.8,
  q8_0: 1,
  bf16: 2,
  f16: 2,
  f32: 3,
};

/** llama.cpp can only run a quantized V cache on the Flash Attention path. */
function syncFlashAttentionWarning() {
  const hint = $('kvFlashAttnHint');
  if (!hint) return;
  const vQuantized = KV_PRECISION_RANK[$('cacheTypeV').value] < 2;
  hint.classList.toggle('hidden', !(vQuantized && $('flashAttention').value === 'off'));
}

/** Mirror K onto V while linked; warn when V is kept more precise than K. */
function syncKvLink(propagate) {
  const link = $('kvTypesLinked');
  const k = $('cacheTypeK');
  const v = $('cacheTypeV');
  const hint = $('kvMismatchHint');
  if (!link || !k || !v) return;
  const linked = !!link.checked;
  if (linked && propagate !== false) {
    v.value = k.value;
  }
  v.disabled = linked;
  if (hint) {
    const lopsided = KV_PRECISION_RANK[v.value] > KV_PRECISION_RANK[k.value];
    hint.classList.toggle('hidden', linked || !lopsided);
  }
  syncFlashAttentionWarning();
}

$('cacheTypeK').addEventListener('change', () => {
  syncKvLink(true);
  refreshMemoryLive();
  scheduleSaveLoad();
});
$('cacheTypeV').addEventListener('change', () => {
  syncKvLink(false);
  refreshMemoryLive();
});
$('kvTypesLinked').addEventListener('change', () => {
  syncKvLink(true);
  refreshMemoryLive();
  scheduleSaveLoad();
});
$('flashAttention').addEventListener('change', syncFlashAttentionWarning);

// 'fit' = largest context that still fits the detected VRAM (core fittingContextLength).
const LOAD_PRESETS: Record<string, any> = {
  agent: { contextLength: 65536, cacheTypeK: 'q8_0', cacheTypeV: 'q8_0', slots: 1 },
  context: { contextLength: 'fit', cacheTypeK: 'q8_0', cacheTypeV: 'q4_0', slots: 1 },
  quality: { contextLength: 65536, cacheTypeK: 'f16', cacheTypeV: 'q8_0', slots: 1 },
};

function currentPresetId() {
  const k = $('cacheTypeK').value;
  const v = $('cacheTypeV').value;
  if (Number($('maxConcurrentPredictions').value) !== 1) return '';
  const ctx = Number($('contextLength').value);
  const maxCtx = ctxMax;
  for (const [id, p] of Object.entries(LOAD_PRESETS)) {
    if (p.cacheTypeK !== k || p.cacheTypeV !== v) continue;
    // 'fit' depends on live VRAM, so any context counts as a match once the
    // distinctive K/V pair lines up.
    if (p.contextLength === 'fit' || ctx === Math.min(p.contextLength, maxCtx)) return id;
  }
  return '';
}

/** Highlight the matching preset, or "Custom" with what the mix is. */
function highlightPreset() {
  const active = currentPresetId();
  document.querySelectorAll<HTMLElement>('#presetChips .chip[data-preset]').forEach((chip) => {
    chip.classList.toggle('active', chip.dataset.preset === active);
  });
  const custom = $('presetCustom');
  if (custom) custom.classList.toggle('active', !active);
  const hint = $('presetHint');
  if (hint) {
    const k = $('cacheTypeK').value;
    const v = $('kvTypesLinked').checked ? k : $('cacheTypeV').value;
    const slots = Number($('maxConcurrentPredictions').value) || 1;
    const mix = fmtTokShort(Number($('contextLength').value)) + ' window · KV ' + k + '/' + v + ' · ' + slots + ' slot' + (slots === 1 ? '' : 's');
    hint.textContent = active
      ? mix + '. Presets set context, KV cache types and slots together.'
      : 'Custom: ' + mix + '.';
  }
}

function applyPreset(id) {
  const p = LOAD_PRESETS[id];
  if (!p) return;
  $('maxConcurrentPredictions').value = p.slots;
  $('cacheTypeK').value = p.cacheTypeK;
  $('cacheTypeV').value = p.cacheTypeV;
  $('kvTypesLinked').checked = p.cacheTypeK === p.cacheTypeV;
  syncKvLink(false);
  syncFlashAttentionWarning();
  if (p.contextLength === 'fit') {
    // The extension searches with core's estimator and pushes the result.
    applyLoadPatch({ cacheTypeK: p.cacheTypeK, cacheTypeV: p.cacheTypeV, maxConcurrentPredictions: p.slots }, { fitContext: true });
    return;
  }
  setContextField(Math.min(p.contextLength, ctxMax));
  highlightPreset();
  refreshMemoryLive();
  scheduleSaveLoad();
}

document.querySelectorAll<HTMLElement>('#presetChips .chip[data-preset]').forEach((chip) => {
  chip.addEventListener('click', () => applyPreset(chip.dataset.preset));
});

// Persist load edits so dirty tracking / reload uses the form values.
const loadFieldIds = [
  'contextLength', 'contextLengthRange', 'gpuOffload', 'gpuOffloadRange',
  'cpuThreads', 'cpuThreadsRange', 'evalBatchSize', 'physicalBatchSize',
  'maxConcurrentPredictions', 'nCpuMoe', 'nCpuMoeRange', 'nCpuFfn', 'nCpuFfnRange', 'offloadKvCacheToGpu',
  'mmprojOffloadToGpu',
  'cacheTypeK', 'cacheTypeV',
  'keepModelInMemory', 'tryMmap', 'lazyMode', 'unifiedKvCache', 'flashAttention',
  'contextCheckpoints', 'cacheReuse',
  'reasoningFormat', 'reasoningBudgetUnlimited', 'reasoningBudget',
  'chatTemplateName', 'chatTemplateJinja',
  'ropeBaseAuto', 'ropeFreqBase', 'ropeScaleAuto', 'ropeFreqScale',
  'seedRandom', 'seed', 'speculativeMode', 'maxDraftTokens', 'minDraftTokens',
  'draftProbability', 'draftGpuOffload',
  'ngramVariant', 'ngramSizeN', 'ngramSizeNRange', 'ngramSizeM', 'ngramMinHits',
  'tensorSplitRange', 'splitMode', 'mainGpu'
];
for (const id of loadFieldIds) {
  const el = $(id);
  if (!el) continue;
  el.addEventListener('change', () => { scheduleSaveLoad(); refreshMemoryLive(); });
  el.addEventListener('input', () => { scheduleSaveLoad(); scheduleLiveMemory(); syncTensorSplitPctLabel(); });
}

// Advanced: mark values that differ from Llama AIO defaults, filter by label or flag.
const ADV_KEYS = new Set([
  'cpuThreads', 'evalBatchSize', 'physicalBatchSize', 'maxConcurrentPredictions', 'flashAttention',
  'offloadKvCacheToGpu', 'cacheTypeK', 'cacheTypeV', 'keepModelInMemory', 'tryMmap', 'lazyMode',
  'unifiedKvCache', 'contextCheckpoints', 'cacheReuse', 'reasoningFormat', 'reasoningBudget',
  'chatTemplateSource', 'chatTemplateName', 'chatTemplateFile', 'chatTemplateJinja',
  'ropeFreqBase', 'ropeFreqScale', 'seed', 'speculativeMode', 'ngramVariant', 'ngramSizeN',
  'ngramSizeM', 'ngramMinHits', 'maxDraftTokens', 'minDraftTokens', 'draftProbability', 'draftGpuOffload',
]);
const ADV_ALIAS = {
  reasoningBudgetUnlimited: 'reasoningBudget', ropeBaseAuto: 'ropeFreqBase', ropeScaleAuto: 'ropeFreqScale',
  seedRandom: 'seed', cpuThreadsRange: 'cpuThreads', ngramSizeNRange: 'ngramSizeN',
};
function advRowKey(row) {
  for (const el of row.querySelectorAll('input[id], select[id]')) {
    const key = ADV_ALIAS[el.id] || el.id;
    if (ADV_KEYS.has(key)) return key;
  }
  return '';
}
function refreshAdvancedMarks() {
  const unified = $('unifiedKvCache').checked;
  const slots = Math.max(1, Number($('maxConcurrentPredictions').value) || 1);
  const context = Math.max(512, Number($('contextLength').value) || 512);
  $('slotContextBadge').textContent = unified ? 'Shared KV' : 'Splits context';
  $('slotContextHint').textContent = unified
    ? 'Use 1 for a single chat. With unified KV, current llama.cpp builds let a slot use the full context window while all slots share the cache. Older builds may divide context; the server’s reported limit takes precedence.'
    : 'Use 1 for a single chat. Without unified KV, context is divided across slots: ' + context.toLocaleString() + ' / ' + slots + ' ≈ ' + Math.floor(context / slots).toLocaleString() + ' tokens per request.';
  const body = $('advancedBody');
  if (!body) return;
  const L = readLoad();
  const query = String(($('advFilter') && $('advFilter').value) || '').trim().toLowerCase();
  const changedOnly = !!($('advChangedOnly') && $('advChangedOnly').checked);
  const filtering = !!query || changedOnly;
  let changedCount = 0;
  const rows = [...body.querySelectorAll('.row, .toggle')];
  for (const row of rows) {
    const key = advRowKey(row);
    const changed = !!key && key in loadDefaults && JSON.stringify(L[key]) !== JSON.stringify(loadDefaults[key]);
    row.classList.toggle('changed', changed);
    // Settings hidden for this mode (e.g. n-gram variant with MTP) do not count.
    if (changed && !row.closest('.hidden')) changedCount++;
    const flags = [...row.querySelectorAll('[data-flag]')].map((n) => n.getAttribute('data-flag')).join(' ');
    const text = (row.textContent + ' ' + flags).toLowerCase();
    const visible = (!query || text.includes(query)) && (!changedOnly || changed);
    row.classList.toggle('filtered-out', filtering && !visible);
  }
  // Hide sub-headings (and spec groups) that have nothing left to show.
  for (const title of body.querySelectorAll('.subgroup-title, .spec-group-title')) {
    let n = title.nextElementSibling;
    let any = false;
    while (n && !n.classList.contains('subgroup-title')) {
      if (n.matches('.row, .toggle, .spec-group') && !n.classList.contains('hidden')) {
        const inner = n.matches('.spec-group') ? [...n.querySelectorAll('.row')] : [n];
        if (inner.some((r) => !r.classList.contains('filtered-out') && !r.classList.contains('hidden'))) any = true;
      }
      n = n.nextElementSibling;
    }
    title.classList.toggle('filtered-out', filtering && !any);
  }
  for (const hint of body.querySelectorAll(':scope > .hint')) hint.classList.toggle('filtered-out', filtering);
  const sum = $('advSum');
  if (sum) sum.textContent = changedCount
    ? changedCount + ' differ from defaults'
    : 'threads, batch, KV, RoPE, speculative';
}
if ($('advFilter')) $('advFilter').addEventListener('input', refreshAdvancedMarks);
if ($('advChangedOnly')) $('advChangedOnly').addEventListener('change', refreshAdvancedMarks);

// Remember open sections and scroll position while the view is hidden (G-15).
const savedUi = vscode.getState() || {};
let scrollRestored = false;
function saveUiState() {
  const folds = [...document.querySelectorAll<HTMLDetailsElement>('details[id]')].filter((d) => d.id !== 'srvMenu');
  const open = folds.filter((d) => d.open).map((d) => d.id);
  // Folds that start open are remembered by being closed instead.
  const closed = folds.filter((d) => !d.open && d.hasAttribute('data-default-open')).map((d) => d.id);
  vscode.setState({ open: open, closed: closed, scrollY: window.scrollY, librarySort: lib.sort });
}
for (const id of Array.isArray(savedUi.open) ? savedUi.open : []) {
  const d = $(id);
  if (d && d.tagName === 'DETAILS') d.open = true;
}
for (const id of Array.isArray(savedUi.closed) ? savedUi.closed : []) {
  const d = $(id);
  if (d && d.tagName === 'DETAILS' && d.hasAttribute('data-default-open')) d.open = false;
}
if (savedUi.librarySort === 'name' || savedUi.librarySort === 'size' || savedUi.librarySort === 'recent') {
  lib.sort = savedUi.librarySort;
}
document.querySelectorAll<HTMLDetailsElement>('details[id]').forEach((d) => d.addEventListener('toggle', saveUiState));
let scrollSaveTimer = null;
window.addEventListener('scroll', () => {
  if (scrollSaveTimer) clearTimeout(scrollSaveTimer);
  scrollSaveTimer = setTimeout(saveUiState, 200);
});

// Request defaults apply to the next chat call (no server reload). Persist on edit.
let saveRequestTimer = null;
function scheduleSaveRequest() {
  if (saveRequestTimer) clearTimeout(saveRequestTimer);
  saveRequestTimer = setTimeout(() => {
    vscode.postMessage({ type: 'saveRequest', payload: readRequest() });
  }, 250);
}
function syncOverthinkingStrength() {
  const box = $('overthinkingPenalty');
  const strength = $('overthinkingPenaltyStrength');
  const on = !!(box && box.checked);
  const strengthText = (strength && strength.value) || '2';
  document.querySelectorAll<HTMLElement>('#otStrengthSeg .seg-btn').forEach((b) => {
    b.classList.toggle('active', Number(b.dataset.v) === Number(strengthText));
  });
  const panel = $('overthinkingBox');
  if (panel) panel.classList.toggle('on', on && !(remoteMode && remoteInfo && remoteInfo.isLlamaCpp === false));
  const limitedRemote = remoteMode && remoteInfo && remoteInfo.isLlamaCpp === false;
  if (box) box.disabled = limitedRemote;
  const quick = $('overthinkingQuick');
  if (quick) {
    quick.disabled = limitedRemote;
    quick.checked = limitedRemote ? false : on;
  }
  const quickState = $('overthinkingQuickState');
  if (quickState) quickState.textContent = !limitedRemote && on ? '−' + strengthText : 'off';
  const hint = $('overthinkingHint');
  if (hint) {
    hint.textContent = limitedRemote
      ? 'Off for this server. It is not llama.cpp, so token ids are not looked up.'
      : on
        ? 'On. The next chat message subtracts ' + strengthText + ' from Wait, maybe, But and similar words. No server reload.'
        : 'Off. Strength is remembered, but chat requests do not send it.';
  }
  const sum = $('copilotSum');
  if (sum) {
    let text = (sum.textContent || '').replace(/\s*·\s*overthink −[\d.]+/g, '').replace(/^overthink −[\d.]+(?:\s*·\s*)?/, '').trim();
    if (on && !limitedRemote) text = text && text !== '—' ? text + ' · overthink −' + strengthText : 'overthink −' + strengthText;
    sum.textContent = text || '—';
  }
}
for (const id of ['temperature', 'topP', 'topK', 'minP', 'repeatPenalty', 'presencePenalty', 'frequencyPenalty', 'maxTokens']) {
  const el = $(id);
  if (!el) continue;
  el.addEventListener('change', scheduleSaveRequest);
  el.addEventListener('input', scheduleSaveRequest);
}
const overthinkingBox = $('overthinkingPenalty');
if (overthinkingBox) {
  overthinkingBox.addEventListener('change', () => {
    syncOverthinkingStrength();
    scheduleSaveRequest();
  });
}
const overthinkingQuick = $('overthinkingQuick');
if (overthinkingQuick && overthinkingBox) {
  overthinkingQuick.addEventListener('change', () => {
    overthinkingBox.checked = overthinkingQuick.checked;
    overthinkingBox.dispatchEvent(new Event('change'));
  });
}
document.querySelectorAll<HTMLElement>('#otStrengthSeg .seg-btn').forEach((b) => {
  b.addEventListener('click', () => {
    const strength = $('overthinkingPenaltyStrength');
    if (!strength || strength.value === b.dataset.v) return;
    strength.value = b.dataset.v;
    syncOverthinkingStrength();
    scheduleSaveRequest();
  });
});
// Header buttons sit inside <summary>; keep a click on them (or on a disabled one) from folding the card.
const copilotDebugBtns = $('copilotDebugBtns');
if (copilotDebugBtns) copilotDebugBtns.addEventListener('click', (e) => e.preventDefault());

function readLoad() {
  const ropeBaseAuto = $('ropeBaseAuto').checked;
  const ropeScaleAuto = $('ropeScaleAuto').checked;
  const seedRandom = $('seedRandom').checked;
  const reasoningUnlimited = $('reasoningBudgetUnlimited').checked;
  const modeSel = $('speculativeMode');
  const mtpOpt = $('specMtpOption');
  let speculativeMode = modeSel ? modeSel.value : 'off';
  // Never persist MTP when the option is unavailable for this GGUF.
  if ((speculativeMode === 'mtp' || speculativeMode === 'ngram-mtp') && mtpOpt && (mtpOpt.disabled || mtpOpt.hidden)) {
    speculativeMode = speculativeMode === 'ngram-mtp' ? 'ngram' : 'off';
    if (modeSel) modeSel.value = speculativeMode;
  }
  return {
    contextLength: Number($('contextLength').value),
    gpuOffload: Number($('gpuOffload').value),
    cpuThreads: Number($('cpuThreads').value),
    evalBatchSize: Number($('evalBatchSize').value),
    physicalBatchSize: Number($('physicalBatchSize').value),
    maxConcurrentPredictions: Number($('maxConcurrentPredictions').value),
    nCpuMoe: Number($('nCpuMoe').value),
    nCpuFfn: Number($('nCpuFfn').value),
    offloadKvCacheToGpu: $('offloadKvCacheToGpu').checked,
    mmprojOffloadToGpu: $('mmprojOffloadToGpu') ? $('mmprojOffloadToGpu').checked : true,
    cacheTypeK: $('cacheTypeK').value || 'q8_0',
    cacheTypeV: ($('kvTypesLinked').checked ? $('cacheTypeK').value : $('cacheTypeV').value) || 'q8_0',
    keepModelInMemory: $('keepModelInMemory').checked,
    tryMmap: $('tryMmap').checked,
    lazyMode: $('lazyMode') ? $('lazyMode').value || 'auto' : 'auto',
    unifiedKvCache: $('unifiedKvCache').checked,
    flashAttention: $('flashAttention').value || 'auto',
    contextCheckpoints: Number($('contextCheckpoints').value),
    cacheReuse: Number($('cacheReuse').value),
    reasoningFormat: $('reasoningFormat').value || 'deepseek-legacy',
    reasoningBudget: reasoningUnlimited ? -1 : Number($('reasoningBudget').value),
    chatTemplateSource: ($('chatTemplateSource') && $('chatTemplateSource').value) || 'model',
    chatTemplateName: ($('chatTemplateName') && $('chatTemplateName').value) || '',
    chatTemplateFile: ($('chatTemplateFile') && $('chatTemplateFile').value) || '',
    chatTemplateJinja: (($('chatTemplateSource') && $('chatTemplateSource').value) || 'model') === 'model'
      ? !!($('chatTemplateJinja') && $('chatTemplateJinja').checked)
      : true,
    ropeFreqBase: ropeBaseAuto ? null : Number($('ropeFreqBase').value),
    ropeFreqScale: ropeScaleAuto ? null : Number($('ropeFreqScale').value),
    seed: seedRandom ? null : Number($('seed').value),
    speculativeMode,
    maxDraftTokens: Number($('maxDraftTokens').value),
    minDraftTokens: Number($('minDraftTokens').value),
    draftProbability: Number($('draftProbability').value),
    ngramVariant: ($('ngramVariant') && $('ngramVariant').value) || 'simple',
    ngramSizeN: Number(($('ngramSizeN') && $('ngramSizeN').value) || 12),
    ngramSizeM: Number(($('ngramSizeM') && $('ngramSizeM').value) || 48),
    ngramMinHits: Number(($('ngramMinHits') && $('ngramMinHits').value) || 1),
    draftGpuOffload: Number(($('draftGpuOffload') && $('draftGpuOffload').value) || 99),
    tensorSplit: readTensorSplitFromUi(),
    splitMode: ($('splitMode') && $('splitMode').value) || 'layer',
    mainGpu: readMainGpuIndex(),
    // draftModelPath is owned by pick/clear handlers only. Never include it in
    // form autosave/reload payloads — an empty hint used to wipe a just-picked path.
  };
}

function setDraftModelHint(draftPath) {
  const hint = $('draftModelPathHint');
  if (!hint) return;
  const p = (draftPath || '').trim();
  hint.dataset.path = p;
  if (!p) {
    hint.textContent = 'No draft model selected.';
    return;
  }
  const base = p.split(/[/\\]/).pop() || p;
  hint.textContent = base + '  ·  ' + p;
}

function setMmprojHint(mmprojPath) {
  const row = $('mmprojAttach');
  if (!row) return;
  const p = (mmprojPath || '').trim();
  row.dataset.path = p;
  row.classList.toggle('hidden', !p);
  const attach = $('attachMmprojBtn');
  const hasModel = !!(lastPayload && lastPayload.state && lastPayload.state.selectedModelPath);
  if (attach) attach.classList.toggle('hidden', !!p || !hasModel);
  const name = $('mmprojName');
  if (name) {
    const base = p ? (p.split(/[/\\]/).pop() || p).replace(/\.gguf$/i, '') : '';
    const inputs = lastPayload && lastPayload.memInputs;
    const sameFile = !!(lastPayload && lastPayload.state && lastPayload.state.loadSettings.mmprojPath === p);
    const size = p && sameFile && inputs && inputs.mmprojFileSizeBytes
      ? ' <span class="sub">· ' + escText(fmtBytes(inputs.mmprojFileSizeBytes)) + '</span>'
      : '';
    name.innerHTML = escText(base) + size;
    name.title = p ? p + '\nCopilot Chat can send images. Reload the server to apply changes.' : '';
  }
  syncMmprojOffloadUi(cpuOnlyLive());
}

function readRequest() {
  return {
    temperature: Number($('temperature').value),
    topP: Number($('topP').value),
    topK: Number($('topK').value),
    maxTokens: Number($('maxTokens').value),
    minP: $('minP') ? Number($('minP').value) : 0,
    repeatPenalty: $('repeatPenalty') ? Number($('repeatPenalty').value) : 1,
    presencePenalty: $('presencePenalty') ? Number($('presencePenalty').value) : 0,
    frequencyPenalty: $('frequencyPenalty') ? Number($('frequencyPenalty').value) : 0,
    overthinkingPenalty: !!($('overthinkingPenalty') && $('overthinkingPenalty').checked),
    overthinkingPenaltyStrength: $('overthinkingPenaltyStrength') ? Number($('overthinkingPenaltyStrength').value) : 2,
  };
}

function applyCapabilities(caps) {
  const maxCtx = (caps && caps.maxContextLength) ? caps.maxContextLength : 131072;
  const blocks = (caps && caps.blockCount) ? caps.blockCount : 128;
  const isMoe = !!(caps && caps.isMoe);
  modelIsMoe = isMoe;
  modelBlockCount = Math.max(1, blocks);

  $('contextLength').max = String(maxCtx);
  ctxMax = Math.max(512, maxCtx);
  ctxMin = Math.min(2048, ctxMax);
  renderCtxTicks();
  // Slider max = actual layer count (legacy 99/"all" is shown as all layers).
  $('gpuOffload').max = String(modelBlockCount);
  $('gpuOffloadRange').max = String(modelBlockCount);
  $('nCpuMoe').max = String(modelBlockCount);
  $('nCpuMoeRange').max = String(modelBlockCount);
  $('nCpuFfn').max = String(modelBlockCount);
  $('nCpuFfnRange').max = String(modelBlockCount);

  // Visibility finalized in applyCpuOnlyUi (also hides on CPU backend).
  if (caps) {
    $('ctxHint').textContent = 'Model max ' + fmtTokShort(maxCtx) + ' (' + Number(maxCtx).toLocaleString() + ' tokens). Per slot, for prompt + reply.';
    $('moeHint').textContent = isMoe
      ? ('MoE model' + (caps.expertCount ? (' · ' + caps.expertCount + ' experts') : '') +
         (caps.expertUsedCount ? (' · ' + caps.expertUsedCount + ' used/token') : '') +
         '. Layers to force experts onto CPU (0–' + blocks + ').')
      : 'Not a MoE model — this setting is hidden.';
    if (isMoe) {
      moeHintDefault = $('moeHint').textContent;
    }
    $('ffnHint').textContent = isMoe
      ? 'MoE model — expert weights are handled by CPU MoE layers (--n-cpu-moe).'
      : ('Dense model. Layers to force dense FFN weights onto CPU (0–' + blocks + ').');
    if (!isMoe) {
      ffnHintDefault = $('ffnHint').textContent;
    }
    const capBits = [
      [(caps.architecture || '?') + ' · ' + (isMoe ? ('MoE ' + (caps.expertCount || '?') + ' experts') : 'dense'), ''],
      [fmtTokShort(maxCtx) + ' max', 'Model max context: ' + Number(maxCtx).toLocaleString() + ' tokens · ' + blocks + ' layers'],
    ];
    if (caps.fullAttentionInterval > 1) capBits.push(['full attn every ' + caps.fullAttentionInterval, 'Hybrid attention: one full-attention layer every ' + caps.fullAttentionInterval + ' layers']);
    if (caps.pleShare > 0) capBits.push(['PLE ~' + Math.round(caps.pleShare * 100) + '%', 'Per-layer embeddings share of the weights']);
    if (caps.nextnPredictLayers > 0) capBits.push(['MTP ✓', 'Multi-token prediction layers baked into the GGUF', 'ok']);
    else if (mtpSidecarPath) capBits.push(['MTP sidecar', 'MTP sidecar: ' + String(mtpSidecarPath).split(/[/\\]/).pop(), 'ok']);
    const loadForTemplate = (lastPayload && lastPayload.state && lastPayload.state.loadSettings) || {};
    if (loadForTemplate.chatTemplateSource === 'builtin' && loadForTemplate.chatTemplateName) {
      capBits.push(['template: ' + loadForTemplate.chatTemplateName, 'Overrides the GGUF chat template with a built-in llama.cpp template', 'override']);
    } else if (loadForTemplate.chatTemplateSource === 'file' && loadForTemplate.chatTemplateFile) {
      const base = String(loadForTemplate.chatTemplateFile).split(/[/\\]/).pop();
      capBits.push(['template: ' + base, loadForTemplate.chatTemplateFile, 'override']);
    } else if (loadForTemplate.chatTemplateJinja === false) {
      capBits.push(['jinja off', 'llama-server is started with --no-jinja', 'override']);
    }
    const modelCaps = $('modelCaps');
    modelCaps.classList.remove('hidden');
    modelCaps.innerHTML = capBits.map(function (b) {
      return '<span class="cap-chip' + (b[2] ? ' ' + b[2] : '') + '"' + (b[1] ? ' title="' + escAttr(b[1]) + '"' : '') + '>' + escText(b[0]) + '</span>';
    }).join('');
    modelCaps.title = '';
    applySpecUi(!!(caps.nextnPredictLayers > 0), sidecarMtpAvailable());
  } else {
    $('modelCaps').classList.add('hidden');
    $('ctxHint').textContent = 'Tokens per slot for prompt + reply';
    applySpecUi(false, sidecarMtpAvailable());
  }
}

function isMtpDraftName(p) {
  const n = String(p || '').split(/[/\\]/).pop() || '';
  return /^mtp[-_]/i.test(n) || /-mtp\.gguf$/i.test(n);
}

function sidecarMtpAvailable() {
  const hint = $('draftModelPathHint');
  const fromHint = hint && hint.dataset ? hint.dataset.path : '';
  return !!(mtpSidecarPath || isMtpDraftName(fromHint));
}

/** Show MTP and/or DFlash controls based on mode + target capabilities. */
function applySpecUi(bakedMtp, sidecarMtp) {
  const mtpCapable = !!(bakedMtp || sidecarMtp);
  const modeSel = $('speculativeMode');
  const mtpOpt = $('specMtpOption');
  const hint = $('specHint');
  const ngramMtpOpt = $('specNgramMtpOption');
  if (mtpOpt) {
    mtpOpt.disabled = !mtpCapable;
    mtpOpt.hidden = !mtpCapable;
  }
  if (ngramMtpOpt) {
    ngramMtpOpt.disabled = !mtpCapable;
    ngramMtpOpt.hidden = !mtpCapable;
  }
  if (modeSel) {
    if (!mtpCapable && modeSel.value === 'mtp') {
      modeSel.value = 'off';
    } else if (!mtpCapable && modeSel.value === 'ngram-mtp') {
      modeSel.value = 'ngram';
    }
  }
  const mode = modeSel ? modeSel.value : 'off';
  const isMtp = mode === 'mtp' || mode === 'ngram-mtp';
  const isDflash = mode === 'dflash' || mode === 'ngram-dflash';
  const isNgram = mode === 'ngram' || mode === 'ngram-mtp' || mode === 'ngram-dflash';
  const showMtpKnobs = isMtp && mtpCapable;
  const showDraftKnobs = isMtp || isDflash;
  const showDraftPicker = isDflash || (isMtp && sidecarMtp && !bakedMtp);
  const ngramGroup = $('specNgramGroup');
  if (ngramGroup) ngramGroup.classList.toggle('hidden', !isNgram);
  const neuralGroup = $('specNeuralGroup');
  if (neuralGroup) neuralGroup.classList.toggle('hidden', !showDraftKnobs);
  const neuralHead = $('specNeuralHeading');
  if (neuralHead) neuralHead.textContent = isMtp ? 'MTP' : 'DFlash';
  const maxName = $('specDraftMaxName');
  if (maxName) maxName.textContent = isMtp ? 'MTP max draft tokens' : 'DFlash max draft tokens';
  const minName = $('specDraftMinName');
  if (minName) minName.textContent = 'MTP min draft tokens';
  const pName = $('specDraftPName');
  if (pName) pName.textContent = isMtp ? 'MTP draft probability' : 'DFlash draft probability';
  const modelName = $('specDraftModelName');
  if (modelName) modelName.textContent = isMtp ? 'MTP draft model' : 'DFlash draft model';
  const nglName = $('specDraftNglName');
  if (nglName) nglName.textContent = isMtp ? 'MTP GPU offload' : 'DFlash GPU offload';

  for (const id of ['maxDraftTokens']) {
    const el = $(id);
    if (el) el.disabled = !showDraftKnobs;
  }
  for (const id of ['minDraftTokens']) {
    const el = $(id);
    if (el) el.disabled = !showMtpKnobs;
  }
  for (const id of ['draftProbability']) {
    const el = $(id);
    if (el) el.disabled = !showDraftKnobs;
  }
  for (const id of ['specDraftMaxRow', 'specDraftPRow']) {
    const row = $(id);
    if (row) {
      row.style.opacity = showDraftKnobs ? '1' : '0.55';
      row.classList.toggle('hidden', !showDraftKnobs);
    }
  }
  for (const id of ['specDraftMinRow']) {
    const row = $(id);
    if (row) {
      row.style.opacity = showMtpKnobs ? '1' : '0.55';
      row.classList.toggle('hidden', !showMtpKnobs);
    }
  }
  for (const id of ['specDraftModelRow', 'specDraftNglRow']) {
    const row = $(id);
    if (row) row.classList.toggle('hidden', !showDraftPicker);
  }
  // N-gram knobs stay visible in stacked modes (ngram-mtp / ngram-dflash) too.
  const ngramVariant = $('ngramVariant') ? $('ngramVariant').value : 'simple';
  for (const id of ['specNgramVariantRow', 'specNgramSizeRow', 'specNgramDraftRow']) {
    const row = $(id);
    if (row) row.classList.toggle('hidden', !isNgram);
  }
  const hitsRow = $('specNgramHitsRow');
  if (hitsRow) hitsRow.classList.toggle('hidden', !isNgram || ngramVariant === 'mod');

  const kindHint = $('draftModelKindHint');
  if (kindHint) {
    kindHint.textContent = isMtp
      ? 'Gemma 4 MTP is a sibling mtp-*.gguf (architecture gemma4-assistant), passed as --model-draft with --spec-type draft-mtp. llama.cpp ≥ 2026-06-07.'
      : 'DFlash needs a separate draft GGUF (architecture = dflash) for your target — not the main model. Download one first, then choose it here.';
  }
  const nglHint = $('draftNglHint');
  if (nglHint) {
    nglHint.textContent = isMtp
      ? '99 usually means all MTP draft layers (--spec-draft-ngl). Sidecar MTP can use the same KV cache types as the main model.'
      : '99 usually means all draft layers. DFlash draft KV cache is forced to f16 (quantized draft KV collapses acceptance).';
  }

  if (hint) {
    if (mode === 'ngram-mtp') {
      hint.textContent =
        'Stacks --spec-type ngram-<variant> with draft-mtp. N-gram drafts from prompt repeats; MTP fills the rest. Needs an MTP-capable GGUF (or sidecar mtp-*.gguf).';
    } else if (mode === 'ngram-dflash') {
      hint.textContent =
        'Stacks --spec-type ngram-<variant> with draft-dflash. N-gram is a cheap lookup; DFlash drafts when there is no n-gram hit. Needs a DFlash draft GGUF.';
    } else if (isNgram) {
      hint.textContent =
        'N-gram passes --spec-type ngram-<variant> with lookup/draft sizes. Drafts come from n-grams in the prompt itself — great for code (edits repeat the prompt) and the only speculative mode that needs no draft model.';
    } else if (isDflash) {
      hint.textContent =
        'DFlash passes --spec-type draft-dflash -md <draft> --spec-draft-ngl … with draft KV forced to f16 and --fit off (llama.cpp auto-fit breaks DFlash). Flash Attention On is recommended.';
    } else if (sidecarMtp && !bakedMtp) {
      hint.textContent =
        'A sibling mtp-*.gguf was found. Mode MTP passes --spec-type draft-mtp --model-draft <mtp> and --fit off (llama.cpp auto-fit breaks Gemma 4 MTP). Needs llama.cpp ≥ 2026-06-07.';
    } else if (bakedMtp) {
      hint.textContent =
        'This model reports MTP next-n layers. Mode MTP passes --spec-type draft-mtp.';
    } else {
      hint.textContent =
        'This GGUF has no MTP / nextn_predict_layers and no sibling mtp-*.gguf — MTP is unavailable. Use DFlash with a separate draft GGUF, N-gram (no draft model), or an MTP-tagged main model.';
    }
  }
}

function fieldFocused(id) {
  const el = $(id);
  return !!(el && document.activeElement === el);
}
/** Leave the control the user is editing alone. Background pushes must not revert it. */
function setField(id, value) {
  const el = $(id);
  if (!el || document.activeElement === el) return;
  const next = value == null ? '' : String(value);
  if (el.value !== next) el.value = next;
}
function setPair(id, rangeId, value) {
  if (fieldFocused(id) || fieldFocused(rangeId)) return;
  setField(id, value);
  setField(rangeId, value);
}
function setChecked(id, on) {
  const el = $(id);
  if (!el || document.activeElement === el) return;
  el.checked = !!on;
}
function setDisabled(id, disabled) {
  const el = $(id);
  if (!el || document.activeElement === el) return;
  el.disabled = !!disabled;
}

function applyState(payload) {
  const s = payload.state;
  const L = s.loadSettings;
  const R = s.requestSettings;
  const status = payload.status;
  const hasModel = !!(s.selectedModelPath);
  const caps = payload.capabilities;
  mtpSidecarPath = payload.mtpSidecarPath || '';
  lastPayload = payload;
  applyRemoteChrome(payload);
  pendingChanges = Array.isArray(payload.changes) ? payload.changes : [];
  loadDefaults = payload.defaults || {};
  gpuInfos = Array.isArray(payload.gpus) ? payload.gpus : [];

  applyCapabilities(caps);

  const cores = Math.max(1, Number(payload.cpuCount) || cpuLogicalCores);
  cpuLogicalCores = cores;
  $('cpuThreads').max = String(cores);
  $('cpuThreadsRange').max = String(cores);
  const cpuHint = $('cpuThreadsHint');
  if (cpuHint) {
    cpuHint.textContent = 'llama.cpp -t. Range 1–' + cores + ' (logical CPU cores).';
  }

  const build = payload.build || {};
  const generating = !!(payload.perf && payload.perf.generating);
  const httpUp = !!(status.httpReady || status.running || generating);
  const starting = !httpUp && !!(status.starting || payload.starting);
  const ready = httpUp;
  const endpoint = payload.endpoint || status.endpoint || lastEndpoint;
  const startMessage = status.startMessage || status.message || '';
  lastEndpoint = endpoint;
  if (status.pid) lastPid = status.pid;

  serverStarting = starting;
  serverRunning = ready;
  configDirty = starting ? false : !!status.configDirty;
  if (Array.isArray(payload.localServers)) foundServers = payload.localServers;
  renderStatusUi({
    ready,
    dirty: configDirty,
    starting,
    endpoint,
    pid: status.pid,
    message: starting ? startMessage : (status.message || ''),
  });
  const lm = $('launchMode');
  if (lm && payload.launchMode && document.activeElement !== lm) {
    lm.value = payload.launchMode === 'background' ? 'background' : 'externalTerminal';
  }
  updatePrimaryAction();

  const perfLines = Array.isArray(payload.perfLines) ? payload.perfLines : ['No generation yet'];
  const perf = payload.perf || {};
  renderContextStack(perf);
  renderPerfStats(perf, perfLines);
  const viewCtx = $('viewContextBtn');
  if (viewCtx) {
    viewCtx.disabled = !payload.hasLastContext;
    viewCtx.title = payload.hasLastContext
      ? 'Open the last Copilot → llama.cpp request (messages + tools) in an editor'
      : 'Send a Copilot Chat message first';
  }
  const viewResp = $('viewResponseBtn');
  if (viewResp) {
    viewResp.disabled = !payload.hasLastResponse;
    viewResp.title = payload.hasLastResponse
      ? 'Open the last llama.cpp → Copilot response stream in an editor'
      : 'Send a Copilot Chat message first';
  }

  setChecked('promptReplacementsEnabled', !!payload.promptReplacementsEnabled);
  setChecked('wikipediaLookupEnabled', !!payload.wikipediaLookupEnabled);
  setChecked('duplicateToolCallGuardEnabled', !!payload.duplicateToolCallGuardEnabled);
  const prStats = $('replacementStats');
  if (prStats) {
    const pr = perf.promptReplacements;
    const hist = Array.isArray(perf.history) ? perf.history : [];
    const sessionSaved = hist.reduce(function (sum, h) {
      return sum + (typeof h.tokensSaved === 'number' ? h.tokensSaved : 0);
    }, 0);
    const sessionTitle = sessionSaved > 0
      ? 'Session: ≈' + Number(sessionSaved).toLocaleString() + ' tok across ' + hist.length + ' call' + (hist.length === 1 ? '' : 's')
      : '';
    if (!payload.promptReplacementsEnabled) {
      prStats.textContent = sessionSaved > 0 ? '−' + fmtTokK(sessionSaved) + ' session' : 'off';
      prStats.title = 'Prompt replacements are off' + (sessionTitle ? '. ' + sessionTitle : '');
    } else if (!pr) {
      prStats.textContent = sessionSaved > 0 ? '−' + fmtTokK(sessionSaved) + ' session' : '—';
      prStats.title = (sessionTitle ? sessionTitle + '. ' : '') + 'Send a chat to measure';
    } else if (!pr.enabled) {
      prStats.textContent = 'off last call';
      prStats.title = (sessionTitle ? sessionTitle + '. ' : '') + 'Last call: replacements were off';
    } else if (pr.tokensSaved > 0) {
      prStats.textContent = '−' + fmtTokK(pr.tokensSaved) + ' last call';
      prStats.title = (sessionTitle ? sessionTitle + '. ' : '') +
        'Last call: saved ≈' + Number(pr.tokensSaved).toLocaleString() +
        ' tokens (' + pr.pctSaved + '% of request)';
    } else {
      prStats.textContent = sessionSaved > 0 ? '−' + fmtTokK(sessionSaved) + ' session' : 'no match';
      prStats.title = (sessionTitle ? sessionTitle + '. ' : '') +
        'Last call: no matching boilerplate (' +
        Number(pr.tokensBefore || 0).toLocaleString() + ' tok request)';
    }
  }

  const binaryDetail = $('llamaBinaryDetail');
  if (binaryDetail) {
    if (build.binaryRunnable === false) {
      binaryDetail.textContent =
        'Binary present but not runnable on this OS' +
        (build.nixOs ? ' (NixOS / missing FHS linker)' : '') +
        (build.binaryRunError ? (': ' + String(build.binaryRunError).slice(0, 120)) : '.');
    } else if (build.activeBackend === 'path' && build.pathBinary) {
      binaryDetail.textContent = build.binaryVersionDetail || ('PATH: ' + build.pathBinary);
    } else if (build.binaryVersionDetail || build.binaryVersion) {
      // "version: 11163" only repeats the installed tag b11163 shown above.
      const detail = String(build.binaryVersionDetail || build.binaryVersion);
      const tagNum = String(build.tag || '').replace(/\D/g, '');
      const versionNum = (/^version:\s*(\d+)/i.exec(detail) || [])[1] || '';
      binaryDetail.textContent = tagNum && versionNum === tagNum && !/\(|·/.test(detail) ? '' : detail;
    } else if (build.tag && payload.binaryExists) {
      binaryDetail.textContent = 'Installed release ' + build.tag;
    } else if (payload.binaryExists) {
      binaryDetail.textContent = 'Binary installed (version string unavailable).';
    } else {
      binaryDetail.textContent = 'No binary for this backend yet.';
    }
  }
  if (build.activeBackend === 'path') {
    $('llamaAssetDetail').textContent = build.pathBinary
      ? ('PATH binary: ' + build.pathBinary)
      : 'No llama-server on PATH.';
  } else {
    $('llamaAssetDetail').textContent = build.asset
      ? ('Asset: ' + build.asset + (build.configuredBackend ? (' · setting: ' + build.configuredBackend) : ''))
      : 'No archive recorded for this backend yet.';
  }

  updateCheck = payload.updateCheck || {
    latestTag: undefined,
    installedTag: build.tag,
    updateAvailable: false,
    checkFailed: false,
    pending: true,
  };

  const sel = $('backendSelect');
  const options = Array.isArray(payload.backendOptions) ? payload.backendOptions : [];
  backendOptionsCache = options;
  activeBackendId = payload.selectedUiBackend || (options.find((o) => o.active) || {}).id || '';
  const prev = sel.value;
  const backendFocused = document.activeElement === sel;
  suppressBackendChange = true;
  if (!backendFocused) {
    sel.innerHTML = '';
    for (const opt of options) {
      const o = document.createElement('option');
      o.value = opt.id;
      let text = opt.label;
      if (!opt.available) {
        text += ' (unavailable)';
      } else if (opt.id === 'path') {
        text += opt.installed
          ? (opt.installedTag ? (' · ' + opt.installedTag) : ' · found')
          : ' · not on PATH';
        if (opt.active) text += ' ●';
      } else if (opt.installed) {
        text += opt.installedTag
          ? (' · installed ' + opt.installedTag)
          : ' · installed';
        if (opt.active) text += ' ●';
      } else {
        text += ' · not installed';
      }
      o.textContent = text;
      // PATH stays selectable even when missing so the user can switch to it
      // and see the install hint; download backends disable when unavailable.
      o.disabled = opt.id === 'path' ? false : !opt.available;
      if (opt.reason && (opt.id === 'path' ? !opt.installed : !opt.available)) {
        o.title = opt.reason;
      } else if (opt.id === 'path' && opt.installed && opt.installedTag) {
        o.title = 'llama-server on PATH';
      } else if (opt.installed && opt.installedTag) {
        o.title = 'Cached locally: ' + opt.installedTag;
      }
      sel.appendChild(o);
    }
    const want = activeBackendId || prev || 'vulkan';
    if ([...sel.options].some((o) => o.value === want && !o.disabled)) {
      sel.value = want;
    } else {
      const first = [...sel.options].find((o) => !o.disabled);
      if (first) sel.value = first.value;
    }
  }
  suppressBackendChange = false;
  updateBackendUi();

  $('setupBox').classList.toggle('hidden', hasModel && payload.binaryExists);

  const localCount = payload.localModelCount || 0;
  const showStarter = !hasModel;
  const starterBtn = $('starterModelBtn');
  const starterHint = $('starterModelHint');
  $('starterCol').classList.toggle('hidden', !showStarter);
  if (starterBtn) starterBtn.classList.toggle('hidden', !showStarter);
  if (starterHint) starterHint.classList.toggle('hidden', !showStarter);
  const titleHtml = modelTitleHtml(hasModel ? (payload.modelName || 'model') : 'No model selected');
  $('modelTitle').innerHTML = titleHtml;
  $('modelClosedName').innerHTML = hasModel ? titleHtml : 'No model selected';
  $('modelTitle').title = hasModel
    ? [s.selectedModelPath, payload.modelNameRaw ? 'general.name: ' + payload.modelNameRaw : ''].filter(Boolean).join('\n')
    : '';
  const libSum = $('librarySum');
  if (libSum) {
    // Not `lib`: that name is the library card's shared state (./library).
    const libPayload = payload.library;
    const visible = libPayload && Array.isArray(libPayload.groups)
      ? libPayload.groups.reduce(function (n, g) { return n + ((g.quants && g.quants.length) || 0); }, 0)
      : localCount;
    const hidden = libPayload && Array.isArray(libPayload.hiddenGroups)
      ? libPayload.hiddenGroups.reduce(function (n, g) { return n + ((g.quants && g.quants.length) || 0); }, 0)
      : 0;
    lib.countLabel = visible + ' GGUF' + (hidden ? ' · ' + hidden + ' hidden' : '');
    refreshLibrarySum();
  }
  function folderLink(label, dir, title) {
    return '<a class="folder-link" href="#" data-path="' + escAttr(dir) + '" title="' +
      escAttr(title || dir) + '">' + escText(label) + '</a>';
  }
  function bindFolderLinks(root) {
    root.querySelectorAll('a.folder-link, a.model-path-link').forEach((link) => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const filePath = link.getAttribute('data-path');
        if (filePath) vscode.postMessage({ type: 'revealInOs', path: filePath });
      });
    });
  }

  // The path lives in the title tooltip and the row menu; the line only guides first-time users.
  $('modelPath').textContent = hasModel ? '' : 'Open the library to pick a model, or search Hugging Face.';
  $('modelPath').classList.toggle('hidden', hasModel);

  const modelsDir = payload.modelsDir || '';
  const sourceDirs = (Array.isArray(payload.localSourceDirs) ? payload.localSourceDirs : [])
    .filter((src) => src.source !== 'Llama AIO');
  const shortDir = modelsDir.replace(/^(\/home\/[^/]+|\/Users\/[^/]+)(?=\/)/, '~');
  $('modelsDirMeta').innerHTML =
    'Saves to ' +
    (modelsDir ? folderLink(shortDir, modelsDir, 'Open downloads folder: ' + modelsDir) : '—') +
    (sourceDirs.length
      ? ' · also scans ' + sourceDirs.map((src) => folderLink(src.source, src.dir, 'Open ' + src.source + ' folder')).join(', ')
      : ' · also scans LM Studio, Unsloth, HF cache');
  $('modelsDirMeta').title = 'Downloads go to ' + (modelsDir || '—') + '. The library also scans LM Studio, Unsloth, the Hugging Face cache and extra model folders.';
  bindFolderLinks($('modelsDirMeta'));
  lib.payload = payload.library || { groups: [], hiddenGroups: [], partials: [], partialSizeLabel: '', serverStarting: false };
  if (lib.selected && !findLibraryQuant(lib.selected)) lib.selected = '';
  if (!lib.selected) {
    const groups = allLibraryGroups();
    for (let i = 0; i < groups.length && !lib.selected; i++) {
      const quants = groups[i].quants || [];
      for (let j = 0; j < quants.length; j++) {
        if (quants[j].current) lib.selected = quants[j].path;
      }
    }
  }
  renderLibraryList();
  renderLibraryPartials();
  renderLibraryDialog();

  const maxCtx = (caps && caps.maxContextLength) ? caps.maxContextLength : 131072;
  const blocks = (caps && caps.blockCount) ? caps.blockCount : 128;
  const ctx = Math.min(L.contextLength, maxCtx);
  // Normalize legacy 99/"all" sentinel to the model's layer count for the slider.
  const ngl =
    L.gpuOffload <= 0 ? 0 : L.gpuOffload >= 99 || L.gpuOffload >= blocks ? blocks : Math.min(L.gpuOffload, blocks);
  const moe = (caps && !caps.isMoe) ? 0 : Math.min(L.nCpuMoe, blocks);
  const ffn = (caps && caps.isMoe) ? 0 : Math.min(L.nCpuFfn ?? 0, blocks);
  const threads = Math.min(Math.max(1, L.cpuThreads || 1), cpuLogicalCores);

  setContextField(ctx);
  setPair('gpuOffload', 'gpuOffloadRange', ngl);
  setPair('cpuThreads', 'cpuThreadsRange', threads);
  setField('evalBatchSize', L.evalBatchSize);
  setField('physicalBatchSize', L.physicalBatchSize);
  setField('maxConcurrentPredictions', L.maxConcurrentPredictions);
  setPair('nCpuMoe', 'nCpuMoeRange', moe);
  setPair('nCpuFfn', 'nCpuFfnRange', ffn);
  setChecked('offloadKvCacheToGpu', !!L.offloadKvCacheToGpu);
  setChecked('mmprojOffloadToGpu', L.mmprojOffloadToGpu !== false);
  if (!fieldFocused('cacheTypeK') && !fieldFocused('cacheTypeV') && !fieldFocused('kvTypesLinked')) {
    setField('cacheTypeK', L.cacheTypeK || 'q8_0');
    setField('cacheTypeV', L.cacheTypeV || 'q8_0');
    setChecked('kvTypesLinked', (L.cacheTypeK || 'q8_0') === (L.cacheTypeV || 'q8_0'));
    syncKvLink(false);
  }
  setChecked('keepModelInMemory', !!L.keepModelInMemory);
  if (payload.isWindows) {
    const label = $('keepModelLabel');
    if (label) label.textContent = 'Keep Model in Memory (mmap on Windows)';
    const hint = $('keepModelHint');
    if (hint) hint.style.display = 'block';
  }
  setChecked('tryMmap', !!L.tryMmap);
  setField('lazyMode', L.lazyMode || 'auto');
  setChecked('unifiedKvCache', !!L.unifiedKvCache);
  setField('flashAttention', L.flashAttention || 'auto');
  setField('contextCheckpoints', L.contextCheckpoints);
  setField('cacheReuse', L.cacheReuse ?? 0);
  setField('reasoningFormat', L.reasoningFormat || 'deepseek-legacy');
  if (!fieldFocused('chatTemplateSource')) setField('chatTemplateSource', L.chatTemplateSource || 'model');
  if (!fieldFocused('chatTemplateName')) setField('chatTemplateName', L.chatTemplateName || 'chatml');
  if (!fieldFocused('chatTemplateFile')) setField('chatTemplateFile', L.chatTemplateFile || '');
  if (!fieldFocused('chatTemplateJinja')) setChecked('chatTemplateJinja', L.chatTemplateJinja !== false);
  syncChatTemplateUi();
  const budget = L.reasoningBudget ?? -1;
  const budgetFocused = fieldFocused('reasoningBudget') || fieldFocused('reasoningBudgetUnlimited');
  if (!budgetFocused) {
    setChecked('reasoningBudgetUnlimited', budget < 0);
    setField('reasoningBudget', budget < 0 ? 2048 : budget);
    setDisabled('reasoningBudget', budget < 0);
  }
  const ropeBaseFocused = fieldFocused('ropeFreqBase') || fieldFocused('ropeBaseAuto');
  if (!ropeBaseFocused) {
    setChecked('ropeBaseAuto', L.ropeFreqBase == null);
    setField('ropeFreqBase', L.ropeFreqBase ?? 10000);
    setDisabled('ropeFreqBase', L.ropeFreqBase == null);
  }
  const ropeScaleFocused = fieldFocused('ropeFreqScale') || fieldFocused('ropeScaleAuto');
  if (!ropeScaleFocused) {
    setChecked('ropeScaleAuto', L.ropeFreqScale == null);
    setField('ropeFreqScale', L.ropeFreqScale ?? 1);
    setDisabled('ropeFreqScale', L.ropeFreqScale == null);
  }
  const seedFocused = fieldFocused('seed') || fieldFocused('seedRandom');
  if (!seedFocused) {
    setChecked('seedRandom', L.seed == null);
    setField('seed', L.seed ?? 0);
    setDisabled('seed', L.seed == null);
  }
  setField('speculativeMode', L.speculativeMode || 'off');
  setField('maxDraftTokens', L.maxDraftTokens);
  setField('minDraftTokens', L.minDraftTokens);
  setField('draftProbability', L.draftProbability);
  setField('ngramVariant', L.ngramVariant || 'simple');
  if (!fieldFocused('ngramSizeN') && !fieldFocused('ngramSizeNRange')) {
    setField('ngramSizeN', L.ngramSizeN ?? 12);
    setField('ngramSizeNRange', Math.min(64, L.ngramSizeN ?? 12));
  }
  setField('ngramSizeM', L.ngramSizeM ?? 48);
  setField('ngramMinHits', L.ngramMinHits ?? 1);
  setField('draftGpuOffload', L.draftGpuOffload ?? 99);
  setField('splitMode', L.splitMode || 'layer');
  setDraftModelHint(L.draftModelPath || '');
  setMmprojHint(L.mmprojPath || '');
  applySpecUi(!!(caps && caps.nextnPredictLayers > 0), sidecarMtpAvailable());
  setField('temperature', R.temperature);
  setField('topP', R.topP);
  setField('topK', R.topK);
  setField('maxTokens', R.maxTokens);
  setField('minP', R.minP ?? 0);
  setField('repeatPenalty', R.repeatPenalty ?? 1);
  setField('presencePenalty', R.presencePenalty ?? 0);
  setField('frequencyPenalty', R.frequencyPenalty ?? 0);
  setChecked('overthinkingPenalty', !!R.overthinkingPenalty);
  setField('overthinkingPenaltyStrength', R.overthinkingPenaltyStrength ?? 2);
  syncOverthinkingStrength();
  renderModeOverrideHint(payload.modeSampling);
  syncFlashAttentionWarning();

  highlightPreset();

  memInputs = payload.memInputs || null;
  const splitDirty = applyDualGpuUi(L);
  applyCpuOnlyUi(!!payload.cpuOnly || payload.selectedUiBackend === 'cpu');
  // Core already estimated the saved settings; newer live requests still win.
  renderedSeq = estimateSeq;
  renderMemoryView(payload.memoryView || null);
  renderCopilotSummary(payload);
  renderNoCopilotHint(payload);
  refreshAdvancedMarks();
  updatePrimaryAction();
  renderHeader();
  if (splitDirty) {
    scheduleSaveLoad();
    requestEstimate();
  }
  if (!scrollRestored) {
    scrollRestored = true;
    if (savedUi.scrollY > 0) window.scrollTo(0, savedUi.scrollY);
  }
}

function renderNoCopilotHint(payload) {
  const el = $('noCopilotHint');
  if (!el) return;
  el.classList.toggle('hidden', payload.copilotChatAvailable !== false);
  const ep = $('noCopilotEndpoint');
  if (ep) ep.textContent = payload.endpoint || '';
}

function renderCopilotSummary(payload) {
  const el = $('copilotSum');
  if (!el) return;
  const R = (payload.state && payload.state.requestSettings) || {};
  const bits = [];
  if (payload.modeSampling) bits.push(payload.modeSampling.familyLabel + ' modes');
  else if (typeof R.temperature === 'number') bits.push('temp ' + R.temperature);
  if (typeof R.maxTokens === 'number' && R.maxTokens > 0) {
    bits.push((R.maxTokens % 1024 === 0 ? R.maxTokens / 1024 + 'k' : fmtTokK(R.maxTokens)) + ' max');
  }
  bits.push('replacements ' + (payload.promptReplacementsEnabled ? 'on' : 'off'));
  if (payload.wikipediaLookupEnabled) bits.push('wiki on');
  if (payload.duplicateToolCallGuardEnabled) bits.push('dup-guard on');
  if (R.overthinkingPenalty) bits.push('overthink −' + (R.overthinkingPenaltyStrength ?? 2));
  el.textContent = bits.join(' · ');
}

$('ropeBaseAuto').addEventListener('change', () => {
  $('ropeFreqBase').disabled = $('ropeBaseAuto').checked;
});
$('ropeScaleAuto').addEventListener('change', () => {
  $('ropeFreqScale').disabled = $('ropeScaleAuto').checked;
});
$('seedRandom').addEventListener('change', () => {
  $('seed').disabled = $('seedRandom').checked;
});
function syncChatTemplateUi() {
  const sourceEl = $('chatTemplateSource');
  const source = (sourceEl && sourceEl.value) || 'model';
  const nameRow = $('chatTemplateNameRow');
  const fileRow = $('chatTemplateFileRow');
  const jinjaRow = $('chatTemplateJinjaRow');
  if (nameRow) nameRow.classList.toggle('hidden', source !== 'builtin');
  if (fileRow) fileRow.classList.toggle('hidden', source !== 'file');
  if (jinjaRow) jinjaRow.classList.toggle('hidden', source !== 'model');
  const hint = $('chatTemplateHint');
  if (hint) {
    hint.textContent = source === 'model'
      ? 'Uses the Jinja template embedded in the GGUF. llama-server’s Jinja engine is already on.'
      : 'Overrides the template stored in this GGUF. Tool calls follow the chosen template. Thinking modes still send chat_template_kwargs; a template that does not read those keys ignores the mode picker.';
  }
  const file = ($('chatTemplateFile') && $('chatTemplateFile').value) || '';
  const nameEl = $('chatTemplateFileName');
  const problem = (lastPayload && lastPayload.chatTemplateProblem) || '';
  if (nameEl) {
    if (!file) {
      nameEl.textContent = 'No template file selected.';
      nameEl.classList.add('missing');
      nameEl.title = '';
    } else {
      nameEl.textContent = file.split(/[/\\]/).pop() || file;
      nameEl.classList.toggle('missing', !!problem);
      nameEl.title = file;
    }
  }
  const problemEl = $('chatTemplateProblem');
  if (problemEl) {
    problemEl.textContent = problem;
    problemEl.classList.toggle('hidden', !problem || source !== 'file');
  }
  const remoteHint = $('chatTemplateRemoteHint');
  if (remoteHint) remoteHint.classList.toggle('hidden', !remoteMode);
  ['chatTemplateSource', 'chatTemplateName', 'chatTemplateJinja', 'pickChatTemplateBtn', 'clearChatTemplateBtn'].forEach(function (id) {
    const el = $(id);
    if (el) el.disabled = !!remoteMode;
  });
  ['chatTemplateSourceRow', 'chatTemplateNameRow', 'chatTemplateFileRow', 'chatTemplateJinjaRow'].forEach(function (id) {
    const el = $(id);
    if (el) el.classList.toggle('locked', !!remoteMode);
  });
}
const chatTemplateSourceEl = $('chatTemplateSource');
if (chatTemplateSourceEl) {
  chatTemplateSourceEl.addEventListener('change', () => {
    if (remoteMode) return;
    if (chatTemplateSourceEl.value === 'file' && !String(($('chatTemplateFile') && $('chatTemplateFile').value) || '').trim()) {
      vscode.postMessage({ type: 'pickChatTemplate' });
      return;
    }
    syncChatTemplateUi();
    scheduleSaveLoad();
    refreshAdvancedMarks();
  });
}
const pickChatTemplateBtn = $('pickChatTemplateBtn');
if (pickChatTemplateBtn) {
  pickChatTemplateBtn.addEventListener('click', () => {
    if (remoteMode) return;
    vscode.postMessage({ type: 'pickChatTemplate' });
  });
}
const clearChatTemplateBtn = $('clearChatTemplateBtn');
if (clearChatTemplateBtn) {
  clearChatTemplateBtn.addEventListener('click', () => {
    if (remoteMode) return;
    if ($('chatTemplateSource')) $('chatTemplateSource').value = 'model';
    if ($('chatTemplateFile')) $('chatTemplateFile').value = '';
    if ($('chatTemplateJinja')) $('chatTemplateJinja').checked = true;
    syncChatTemplateUi();
    scheduleSaveLoad();
    refreshAdvancedMarks();
  });
}
const modelCapsEl = $('modelCaps');
if (modelCapsEl) {
  modelCapsEl.addEventListener('click', (e) => {
    const chip = e.target && e.target.closest ? e.target.closest('.cap-chip.override') : null;
    if (!chip) return;
    const fold = $('advancedFold');
    if (fold) fold.open = true;
    const row = $('chatTemplateSourceRow');
    if (row && row.scrollIntoView) row.scrollIntoView({ block: 'nearest' });
  });
}

$('reasoningBudgetUnlimited').addEventListener('change', () => {
  $('reasoningBudget').disabled = $('reasoningBudgetUnlimited').checked;
});
function recommendedMaxDraftTokens(mode, current, sidecarMtp) {
  const mtpDefault = sidecarMtp ? 4 : 2;
  const n = Number(current) || 0;
  const dflash = mode === 'dflash' || mode === 'ngram-dflash';
  const mtp = mode === 'mtp' || mode === 'ngram-mtp';
  if (dflash) return n <= 4 ? 15 : n;
  if (mtp) {
    if (n <= 0 || n >= 8) return mtpDefault;
    if (sidecarMtp && n === 2) return 4;
    return n;
  }
  return n;
}
function recommendedNgramSizes(variant, n, m) {
  const simpleDefault = n === 12 && m === 48;
  const modDefault = n === 24 && m === 64;
  if (variant === 'mod') return simpleDefault ? { n: 24, m: 64 } : { n: n, m: m };
  return modDefault ? { n: 12, m: 48 } : { n: n, m: m };
}
const speculativeModeEl = $('speculativeMode');
if (speculativeModeEl) {
  speculativeModeEl.addEventListener('change', () => {
    const maxEl = $('maxDraftTokens');
    if (maxEl) {
      maxEl.value = String(recommendedMaxDraftTokens(
        speculativeModeEl.value,
        Number(maxEl.value),
        sidecarMtpAvailable()
      ));
    }
    applySpecUi(!!(memInputs && memInputs.nextnPredictLayers > 0), sidecarMtpAvailable());
  });
}
const ngramVariantEl = $('ngramVariant');
if (ngramVariantEl) {
  ngramVariantEl.addEventListener('change', () => {
    const nEl = $('ngramSizeN');
    const mEl = $('ngramSizeM');
    const rangeEl = $('ngramSizeNRange');
    if (nEl && mEl) {
      const next = recommendedNgramSizes(ngramVariantEl.value, Number(nEl.value), Number(mEl.value));
      nEl.value = String(next.n);
      mEl.value = String(next.m);
      if (rangeEl) rangeEl.value = String(Math.min(64, next.n));
    }
    applySpecUi(!!(memInputs && memInputs.nextnPredictLayers > 0), sidecarMtpAvailable());
  });
}
const pickDraftBtn = $('pickDraftModelBtn');
if (pickDraftBtn) {
  pickDraftBtn.addEventListener('click', () => {
    if (saveLoadTimer) clearTimeout(saveLoadTimer);
    vscode.postMessage({ type: 'pickDraftModel' });
  });
}
const clearDraftBtn = $('clearDraftModelBtn');
if (clearDraftBtn) {
  clearDraftBtn.addEventListener('click', () => {
    if (saveLoadTimer) clearTimeout(saveLoadTimer);
    setDraftModelHint('');
    vscode.postMessage({ type: 'clearDraftModel' });
  });
}
const pickMmprojBtn = $('pickMmprojBtn');
if (pickMmprojBtn) {
  pickMmprojBtn.addEventListener('click', () => {
    if (saveLoadTimer) clearTimeout(saveLoadTimer);
    vscode.postMessage({ type: 'pickMmproj' });
  });
}
const clearMmprojBtn = $('clearMmprojBtn');
if (clearMmprojBtn) {
  clearMmprojBtn.addEventListener('click', () => {
    if (saveLoadTimer) clearTimeout(saveLoadTimer);
    setMmprojHint('');
    vscode.postMessage({ type: 'clearMmproj' });
  });
}

const viewContextBtn = $('viewContextBtn');
if (viewContextBtn) {
  viewContextBtn.addEventListener('click', () => vscode.postMessage({ type: 'viewLastCall' }));
}
const viewResponseBtn = $('viewResponseBtn');
if (viewResponseBtn) {
  viewResponseBtn.addEventListener('click', () => vscode.postMessage({ type: 'viewLastResponse' }));
}
const resetAdvancedBtn = $('resetAdvancedBtn');
if (resetAdvancedBtn) {
  resetAdvancedBtn.addEventListener('click', () => vscode.postMessage({ type: 'resetAdvancedLoad' }));
}
const resetRequestBtn = $('resetRequestBtn');
if (resetRequestBtn) {
  resetRequestBtn.addEventListener('click', () => vscode.postMessage({ type: 'resetRequestDefaults' }));
}
const prToggle = $('promptReplacementsEnabled');
if (prToggle) {
  prToggle.addEventListener('change', () => {
    vscode.postMessage({
      type: 'setPromptReplacementsEnabled',
      payload: { enabled: !!prToggle.checked },
    });
  });
}
const wikiToggle = $('wikipediaLookupEnabled');
if (wikiToggle) {
  wikiToggle.addEventListener('change', () => {
    vscode.postMessage({
      type: 'setWikipediaLookupEnabled',
      payload: { enabled: !!wikiToggle.checked },
    });
  });
}
const dupToggle = $('duplicateToolCallGuardEnabled');
if (dupToggle) {
  dupToggle.addEventListener('change', () => {
    vscode.postMessage({
      type: 'setDuplicateToolCallGuardEnabled',
      payload: { enabled: !!dupToggle.checked },
    });
  });
}
const starterModelBtn = $('starterModelBtn');
if (starterModelBtn) {
  starterModelBtn.addEventListener('click', () => vscode.postMessage({ type: 'downloadStarter' }));
}
const setupStarterBtn = $('setupStarterBtn');
if (setupStarterBtn) {
  setupStarterBtn.addEventListener('click', () => vscode.postMessage({ type: 'downloadStarter' }));
}
$('openFileBtn').addEventListener('click', () => vscode.postMessage({ type: 'openModelFile' }));
$('installLlamaBtn').addEventListener('click', () => {
  const action = $('installLlamaBtn').dataset.action;
  if (action === 'check') {
    vscode.postMessage({ type: 'checkUpdates' });
    return;
  }
  vscode.postMessage({ type: 'installLlamaCpp', payload: $('backendSelect').value });
});
const checkUpdatesBtn = $('checkUpdatesBtn');
if (checkUpdatesBtn) {
  checkUpdatesBtn.addEventListener('click', () => vscode.postMessage({ type: 'checkUpdates' }));
}
const reinstallLlamaBtn = $('reinstallLlamaBtn');
if (reinstallLlamaBtn) {
  reinstallLlamaBtn.addEventListener('click', () => vscode.postMessage({ type: 'reinstallLlamaCpp' }));
}
$('installByTagBtn').addEventListener('click', () => vscode.postMessage({ type: 'installLlamaCppByTag' }));
$('installArchiveBtn').addEventListener('click', () => vscode.postMessage({ type: 'installLlamaCppFromArchive' }));
const releasesLink = $('releasesLink');
if (releasesLink) {
  releasesLink.addEventListener('click', (e) => {
    e.preventDefault();
    vscode.postMessage({ type: 'openExternal', url: 'https://github.com/ggml-org/llama.cpp/releases' });
  });
}
$('backendSelect').addEventListener('change', () => {
  if (suppressBackendChange) return;
  updateBackendUi();
  applyCpuOnlyUi($('backendSelect').value === 'cpu');
  refreshMemoryLive();
  const next = $('backendSelect').value;
  if (!next || next === activeBackendId) return;
  const opt = backendOptionsCache.find((o) => o.id === next);
  // PATH stays selectable when missing so the user can switch and see the hint.
  if (opt && !opt.available && next !== 'path') return;
  vscode.postMessage({ type: 'switchBackend', payload: next });
});

function launch(action: string) {
  if (action !== 'reload' && action !== 'start') return;
  serverStarting = true;
  updatePrimaryAction();
  renderStatusUi({
    ready: false,
    dirty: false,
    starting: true,
    endpoint: '',
    message: '',
  });
  vscode.postMessage({ type: action, payload: readLoad() });
  vscode.postMessage({ type: 'saveRequest', payload: readRequest() });
}
$('primaryBtn').addEventListener('click', () => launch($('primaryBtn').dataset.action));
$('reloadIconBtn').addEventListener('click', () => {
  if (remoteMode) {
    vscode.postMessage({ type: 'refreshRemote' });
    return;
  }
  launch('reload');
});
$('discardBtn').addEventListener('click', () => {
  if (saveLoadTimer) {
    clearTimeout(saveLoadTimer);
    saveLoadTimer = null;
  }
  vscode.postMessage({ type: 'discardChanges' });
});
$('dirtyReloadBtn').addEventListener('click', () => launch('reload'));
$('dirtyToggle').addEventListener('click', () => {
  const list = $('dirtyList');
  const toggle = $('dirtyToggle');
  const open = list.classList.contains('hidden');
  list.classList.toggle('hidden', !open);
  toggle.setAttribute('aria-expanded', String(open));
  toggle.classList.toggle('open', open);
});
$('stopBtn').addEventListener('click', () => vscode.postMessage({ type: 'stop' }));
const closeMenu = () => {
  const menu = $('srvMenu');
  if (menu) menu.open = false;
};
$('openLogBtn').addEventListener('click', () => { closeMenu(); vscode.postMessage({ type: 'openLog' }); });
$('copyCmdBtn').addEventListener('click', () => { closeMenu(); vscode.postMessage({ type: 'copyCommandLine' }); });
$('openWebUiBtn').addEventListener('click', () => {
  closeMenu();
  if (lastEndpoint) vscode.postMessage({ type: 'openExternal', url: lastEndpoint });
});
function scheduleProbe() {
  if (probeTimer) clearTimeout(probeTimer);
  const url = ($('remoteUrlInput') && $('remoteUrlInput').value || '').trim();
  if (url.length < 4) return;
  probeTimer = setTimeout(function () {
    vscode.postMessage({
      type: 'testRemote',
      url: url,
      apiKey: ($('remoteKeyInput') && $('remoteKeyInput').value) || '',
    });
  }, 600);
}
function openConnectForm(prefill?) {
  const payload = lastPayload || {};
  setConnectOpen(true, prefill || {
    url: payload.remoteUrl || payload.endpoint || '',
    label: payload.serverMode === 'remote' ? (payload.remoteLabel || '') : '',
  });
}
$('connectRemoteBtn').addEventListener('click', () => { closeMenu(); openConnectForm({ url: '', label: '' }); });
$('editConnectionBtn').addEventListener('click', () => { closeMenu(); openConnectForm(); });
$('copyDiagBtn').addEventListener('click', () => { closeMenu(); vscode.postMessage({ type: 'copyDiagnostics' }); });
$('setApiKeyBtn').addEventListener('click', () => { closeMenu(); vscode.postMessage({ type: 'setApiKey' }); });
$('forgetApiKeyBtn').addEventListener('click', () => { closeMenu(); vscode.postMessage({ type: 'forgetApiKey' }); });
$('switchLocalBtn').addEventListener('click', () => { closeMenu(); vscode.postMessage({ type: 'switchToLocal' }); });
$('connectCancelBtn').addEventListener('click', () => setConnectOpen(false));
$('connectBtn').addEventListener('click', () => {
  vscode.postMessage({
    type: 'connectRemote',
    url: ($('remoteUrlInput') && $('remoteUrlInput').value) || '',
    label: ($('remoteLabelInput') && $('remoteLabelInput').value) || '',
    apiKey: ($('remoteKeyInput') && $('remoteKeyInput').value) || '',
  });
  const key = $('remoteKeyInput');
  if (key) key.value = '';
});
$('remoteUrlInput').addEventListener('input', scheduleProbe);
$('remoteKeyInput').addEventListener('input', scheduleProbe);
$('remotePrimaryBtn').addEventListener('click', () => {
  const action = $('remotePrimaryBtn').dataset.action;
  if (action === 'setApiKey') vscode.postMessage({ type: 'setApiKey' });
  else vscode.postMessage({ type: 'refreshRemote' });
});
$('remoteSecondaryBtn').addEventListener('click', () => openConnectForm());
document.addEventListener('click', (e) => {
  const menu = $('srvMenu');
  if (menu && menu.open && !menu.contains(e.target)) menu.open = false;
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeMenu();
});

$('nglAll').addEventListener('click', () => {
  nglCustomOpen = false;
  const blocks = Math.max(1, modelBlockCount || 1);
  $('gpuOffload').value = String(blocks);
  $('gpuOffloadRange').value = String(blocks);
  syncNglMode();
  scheduleSaveLoad();
  refreshMemoryLive();
});
$('nglCustom').addEventListener('click', () => {
  nglCustomOpen = true;
  syncNglMode();
  $('gpuOffloadRange').focus();
});
document.querySelectorAll<HTMLElement>('#splitSeg .seg-btn').forEach((b) => {
  b.addEventListener('click', () => {
    const sel = $('splitMode');
    if (!sel || sel.disabled || sel.value === b.dataset.split) return;
    sel.value = b.dataset.split;
    sel.dispatchEvent(new Event('change'));
    applyCpuOnlyUi(cpuOnlyLive());
  });
});
$('launchMode').addEventListener('change', () => {
  const mode = $('launchMode').value === 'background' ? 'background' : 'externalTerminal';
  if (serverRunning) {
    configDirty = true;
    updatePrimaryAction();
    const hint = $('dirtyHint');
    if (hint) hint.classList.remove('hidden');
    setServerCardKind('dirty');
    const line = $('statusLine');
    const dot = $('statusDot');
    if (line) line.className = 'status-line dirty';
    if (dot) dot.className = 'dot dirty';
  }
  vscode.postMessage({ type: 'setLaunchMode', payload: mode });
});

$('libraryFilter').addEventListener('input', function () {
  resetLibraryNavigation();
  renderLibraryList();
});
$('libraryFilter').addEventListener('keydown', function (e) {
  if (e.key === 'Enter') startLibrarySearch();
});
$('libraryFilters').addEventListener('click', function (e) {
  const chip = e.target.closest ? e.target.closest('[data-lib-filter]') : null;
  if (!chip) return;
  lib.filterMode = chip.getAttribute('data-lib-filter') || 'all';
  lib.page = 0;
  lib.menuPath = '';
  renderLibraryList();
  renderLibraryDialog();
});
$('librarySort').value = lib.sort;
$('librarySort').addEventListener('change', function () {
  lib.sort = $('librarySort').value || 'recent';
  lib.page = 0;
  renderLibraryList();
  saveUiState();
});
$('libraryList').addEventListener('keydown', function (e) {
  const row = e.target.classList && e.target.classList.contains('lib-quant') ? e.target : null;
  if (!row) return;
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    selectLibraryPath(row.getAttribute('data-path') || '');
  } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    const rows = [...document.querySelectorAll<HTMLElement>('#libraryList .lib-quant')];
    const next = rows[rows.indexOf(row) + (e.key === 'ArrowDown' ? 1 : -1)];
    if (next) next.focus();
  }
});
$('libraryList').addEventListener('contextmenu', function (e) {
  const row = e.target.closest ? e.target.closest('.lib-quant') : null;
  if (!row) return;
  e.preventDefault();
  lib.menuPath = '';
  openLibraryMenu(row.getAttribute('data-path') || '');
});
document.addEventListener('click', function (e) {
  const target = e.target as Element | null;
  if (lib.menuPath && !(target && target.closest && target.closest('.lib-menu'))) closeLibraryMenu();
});
document.addEventListener('keydown', function (e) {
  if (e.key === 'Escape') closeLibraryMenu();
});
$('libraryQuick').addEventListener('click', function (e) {
  const chip = e.target.closest ? e.target.closest('[data-quick-path]') : null;
  if (chip) selectLibraryPath(chip.getAttribute('data-quick-path') || '');
});
$('libraryFold').addEventListener('toggle', refreshLibrarySum);
document.querySelectorAll<HTMLElement>('[data-model-reload]').forEach(function (b) {
  b.addEventListener('click', function (e) {
    e.stopPropagation();
    launch('reload');
  });
});
document.querySelectorAll<HTMLElement>('#mmprojSeg .seg-btn').forEach(function (b) {
  b.addEventListener('click', function () {
    const el = $('mmprojOffloadToGpu');
    const want = b.dataset.mm === 'gpu';
    if (!el || el.disabled || el.checked === want) return;
    el.checked = want;
    el.dispatchEvent(new Event('change'));
    syncMmprojOffloadUi(cpuOnlyLive());
  });
});
$('attachMmprojBtn').addEventListener('click', () => vscode.postMessage({ type: 'pickMmproj' }));
$('libraryList').addEventListener('click', onLibraryClick);
$('libraryJobs').addEventListener('click', onLibraryClick);
$('libraryPartials').addEventListener('click', function (e) {
  const btn = e.target.closest ? e.target.closest('[data-act]') : null;
  if (btn && btn.getAttribute('data-act') === 'clean-partials') {
    vscode.postMessage({ type: 'cleanPartialDownloads' });
  }
});
$('llamaInstallBox').addEventListener('click', function (e) {
  const btn = e.target.closest ? e.target.closest('[data-act]') : null;
  if (!btn) return;
  const act = btn.getAttribute('data-act');
  if (act === 'install-cancel') {
    vscode.postMessage({ type: 'cancelLlamaInstall' });
  } else if (act === 'install-retry') {
    noteInstallStarting('Retrying…');
    vscode.postMessage({ type: 'retryLlamaInstall' });
  } else if (act === 'install-dismiss') {
    llamaInstall = null;
    renderLlamaInstall();
  }
});
$('libraryDialog').addEventListener('change', function (e) {
  const input = e.target;
  if (!input || !input.getAttribute) return;
  const companion = input.getAttribute('data-companion');
  if (!companion) return;
  lib.dialogChecks[companion] = !!input.checked;
});
$('libraryDialog').addEventListener('click', function (e) {
  const btn = e.target.closest ? e.target.closest('[data-act]') : null;
  if (!btn) return;
  const act = btn.getAttribute('data-act');
  if (act === 'cancel-remove') {
    lib.dialogPath = '';
    lib.dialogChecks = {};
    renderLibraryDialog();
    return;
  }
  if (act !== 'confirm-remove') return;
  const found = findLibraryQuant(lib.dialogPath);
  if (!found) return;
  const companions = [];
  (found.group.companions || []).forEach(function (c) {
    const checked = Object.prototype.hasOwnProperty.call(lib.dialogChecks, c.path) ? lib.dialogChecks[c.path] : !c.shared;
    if (checked) companions.push(c.path);
  });
  const filePath = lib.dialogPath;
  lib.dialogPath = '';
  lib.dialogChecks = {};
  renderLibraryDialog();
  vscode.postMessage({ type: 'removeLibraryQuant', path: filePath, companionPaths: companions });
});

window.addEventListener('message', (event: MessageEvent<ToWebviewMessage>) => {
  const msg = event.data;
  if (msg.type === 'state') applyState(msg.payload);
  if (msg.type === 'openConnect') {
    openConnectForm(lastPayload && lastPayload.serverMode === 'remote' ? undefined : { url: '', label: '' });
  }
  if (msg.type === 'remoteProbe') renderProbeList(msg.payload, msg.error || '');
  if (msg.type === 'libraryFits') {
    lib.fits = msg.fits || {};
    lib.ratios = msg.ratios || {};
    lib.drafts = msg.drafts || {};
    lib.budgetLabel = msg.budgetLabel || '';
    if (lib.hfMode === 'local') renderLibraryList();
    else renderLibraryFilters();
    renderModelStatus();
  }
  if (msg.type === 'libraryDownloads') {
    lib.downloads = msg.jobs || [];
    renderLibraryJobs();
  }
  if (msg.type === 'libraryHfResults' || msg.type === 'libraryHfFiles' || msg.type === 'libraryHfError') {
    applyLibraryHfMessage(msg);
  }
  if (msg.type === 'focusLibrary') {
    const modelFold = $('modelFold');
    if (modelFold) modelFold.open = true;
    const fold = $('libraryFold');
    if (fold) fold.open = true;
    const filter = $('libraryFilter');
    if (msg.query && filter) filter.value = msg.query;
    if (msg.search && msg.query) startLibrarySearch();
    else {
      resetLibraryNavigation();
      renderLibraryList();
    }
    if (msg.filter && filter) filter.focus();
    if (msg.downloads) {
      const row = document.querySelector('.lib-job');
      if (row && row.scrollIntoView) row.scrollIntoView({ block: 'nearest' });
    }
  }
  if (msg.type === 'memoryEstimate') {
    // Drop replies to superseded requests (typing or dragging fires many).
    if (typeof msg.seq === 'number' && msg.seq >= renderedSeq && msg.seq === estimateSeq) {
      renderedSeq = msg.seq;
      renderMemoryView(msg.view || null);
    }
  }
  if (msg.type === 'updateCheck' && msg.payload) {
    updateCheck = msg.payload;
    updateBackendUi();
  }
  if (msg.type === 'llamaInstall') {
    llamaInstall = msg.payload || null;
    renderLlamaInstall();
    applyInstallLock();
  }
  if (msg.type === 'perfPatch' && msg.payload) {
    const perf = msg.payload.perf || {};
    renderContextStack(perf);
    renderPerfStats(perf, msg.payload.perfLines || []);
    const viewCtx = $('viewContextBtn');
    if (viewCtx) {
      viewCtx.disabled = !msg.payload.hasLastContext;
    }
    const viewResp = $('viewResponseBtn');
    if (viewResp) {
      viewResp.disabled = !msg.payload.hasLastResponse;
    }
    if (perf.generating && serverStarting) {
      serverStarting = false;
      serverRunning = true;
      updatePrimaryAction();
      renderStatusUi({
        ready: true,
        dirty: configDirty,
        starting: false,
        endpoint: lastEndpoint,
        pid: lastPid,
        message: '',
      });
    }
  }
  if (msg.type === 'draftModelSelected') {
    setDraftModelHint(msg.path || '');
    if (isMtpDraftName(msg.path)) {
      mtpSidecarPath = mtpSidecarPath || msg.path;
    }
    applySpecUi(!!(memInputs && memInputs.nextnPredictLayers > 0), sidecarMtpAvailable());
    refreshMemoryLive();
  }
  if (msg.type === 'mmprojSelected') {
    setMmprojHint(msg.path || '');
    refreshMemoryLive();
  }
  if (msg.type === 'bootProgress' && msg.payload) {
    serverStarting = true;
    serverRunning = false;
    configDirty = false;
    updatePrimaryAction();
    renderStatusUi({
      ready: false,
      dirty: false,
      starting: true,
      endpoint: '',
      message: msg.payload.message || 'Starting…',
    });
  }
  if (msg.type === 'statusPatch' && msg.payload) {
    const p = msg.payload;
    if (p.serverMode) {
      if (lastPayload) {
        lastPayload.serverMode = p.serverMode;
        lastPayload.remote = p.remote;
        lastPayload.remoteLabel = p.remoteLabel;
        lastPayload.remoteHint = p.remoteHint;
        lastPayload.nextRetryAt = p.nextRetryAt;
        lastPayload.endpoint = p.endpoint || lastPayload.endpoint;
      }
      applyRemoteChrome(p);
    }
    const httpUp = !!(p.running || p.httpReady);
    const starting = !!p.starting && !httpUp;
    if (p.endpoint) lastEndpoint = p.endpoint;
    if (p.pid) lastPid = p.pid;
    serverStarting = starting;
    serverRunning = httpUp;
    configDirty = starting ? false : !!p.configDirty;
    if (Array.isArray(p.localServers)) foundServers = p.localServers;
    pendingChanges = Array.isArray(p.changes) ? p.changes : [];
    updatePrimaryAction();
    renderStatusUi({
      ready: httpUp,
      dirty: configDirty,
      starting,
      endpoint: lastEndpoint,
      pid: lastPid,
      message: starting ? (p.startMessage || p.message || 'Starting…') : (p.message || ''),
    });
    renderHeader();
    if (p.perf || Array.isArray(p.perfLines)) {
      renderPerfStats(p.perf || {}, p.perfLines || []);
    }
  }
});

vscode.postMessage({ type: 'ready' });
