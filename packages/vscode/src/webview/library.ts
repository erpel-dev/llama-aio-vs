/**
 * Model library card: local GGUF groups, fit badges, Hugging Face search and
 * download jobs. State lives in `lib` so the main script's message handler
 * can update it; rendering and clicks live here.
 */
import { $, vscode } from "./api";
import { escAttr, escText, fmtBytes, fmtCount, fmtEta } from "./format";
import type { ToWebviewMessage } from "./protocol";

const LIBRARY_PAGE_SIZE = 50;

/** Library card state. Assigned by the state / libraryX messages in sidebar.ts. */
export const lib = {
  payload: <any>{ groups: [], hiddenGroups: [], partials: [], partialSizeLabel: '', serverStarting: false },
  selected: '',
  dialogPath: '',
  dialogChecks: {} as any,
  fits: {} as any,
  hfMode: 'local',
  hfBusy: false,
  hfError: '',
  hfRepos: [] as any,
  hfFiles: null as any,
  hfRequest: 0,
  page: 0,
  downloads: [] as any,
  countLabel: '',
  ratios: {} as any,
  drafts: {} as any,
  budgetLabel: '',
  filterMode: 'all',
  sort: 'recent',
  menuPath: '',
};

/** Invalidates outstanding HF replies, including when returning to local results. */
export function resetLibraryNavigation(mode = 'local') {
  lib.hfRequest++;
  lib.hfMode = mode;
  lib.hfBusy = false;
  lib.hfError = '';
  lib.hfFiles = null;
  lib.page = 0;
}

function requestLibraryFiles(modelId: string) {
  resetLibraryNavigation('files');
  lib.hfBusy = true;
  lib.hfFiles = { modelId, files: [] };
  renderLibraryList();
  vscode.postMessage({ type: 'listLibraryHfFiles', modelId, requestId: lib.hfRequest });
}

/** Accept only replies belonging to the current navigation, even when the
 * user repeats the same query or visits the same repo again. */
export function applyLibraryHfMessage(msg: Extract<ToWebviewMessage, { type: 'libraryHfResults' | 'libraryHfFiles' | 'libraryHfError' }>): boolean {
  if (msg.requestId !== lib.hfRequest) return false;
  if (msg.type === 'libraryHfFiles') {
    if (lib.hfMode !== 'files' || msg.modelId !== lib.hfFiles?.modelId) return false;
    lib.hfFiles = msg;
  } else {
    const raw = (($('libraryFilter') && $('libraryFilter').value) || '').trim().toLowerCase();
    if (msg.query && msg.query.toLowerCase() !== raw) return false;
    if (msg.type === 'libraryHfResults') {
      if (lib.hfMode !== 'search') return false;
      lib.hfRepos = msg.repos || [];
    }
  }
  lib.hfBusy = false;
  lib.hfError = msg.type === 'libraryHfError' ? msg.message || 'Search failed.' : '';
  renderLibraryList();
  return true;
}

export function libraryStats() {
  const stats = { total: 0, fits: 0, drafts: 0, vision: 0, hidden: 0 };
  (lib.payload.groups || []).forEach(function (g) {
    const vision = groupHasVision(g);
    (g.quants || []).forEach(function (q) {
      stats.total++;
      if (lib.fits[q.path] === 'fits') stats.fits++;
      if (lib.drafts[q.path]) stats.drafts++;
      if (vision) stats.vision++;
    });
  });
  (lib.payload.hiddenGroups || []).forEach(function (g) { stats.hidden += (g.quants || []).length; });
  return stats;
}

export function refreshLibrarySum() {
  const el = $('librarySum');
  if (!el) return;
  const n = lib.downloads.length;
  const dl = n ? (n + ' downloading') : '';
  const fold = $('libraryFold');
  let label = lib.countLabel;
  if (fold && !fold.open && lib.countLabel) {
    const st = libraryStats();
    const bits = [st.total + ' GGUF'];
    if (Object.keys(lib.fits).length) bits.push(st.fits + ' fit');
    if (st.drafts) bits.push(st.drafts + ' draft' + (st.drafts === 1 ? '' : 's'));
    label = bits.join(' · ');
  } else if (lib.budgetLabel && lib.countLabel) {
    label = lib.countLabel + ' · ' + lib.budgetLabel;
  }
  el.textContent = dl && label ? dl + ' · ' + label : (dl || label || '—');
}

export function groupHasVision(g) {
  return (g.companions || []).some(function (c) { return c.role === 'mmproj'; });
}

export function allLibraryGroups() {
  const groups = (lib.payload.groups || []).map(function (g) {
    return Object.assign({ hidden: false }, g);
  });
  (lib.payload.hiddenGroups || []).forEach(function (g) {
    groups.push(Object.assign({ hidden: true }, g));
  });
  return groups;
}

export function findLibraryQuant(filePath) {
  const groups = allLibraryGroups();
  for (let i = 0; i < groups.length; i++) {
    const quants = groups[i].quants || [];
    for (let j = 0; j < quants.length; j++) {
      if (quants[j].path === filePath) return { group: groups[i], quant: quants[j] };
    }
  }
  return null;
}

export function fitText(fit) {
  return fit === 'wont-fit' ? 'will not fit' : fit === 'tight' ? 'tight (under 1.5 GiB left)' : fit === 'fits' ? 'fits' : '';
}

export function fitBarHtml(filePath) {
  const fit = lib.fits[filePath] || '';
  const ratio = lib.ratios[filePath];
  const pct = typeof ratio === 'number' ? Math.max(2, Math.min(100, Math.round(ratio * 100))) : 0;
  const tip = fit ? fitText(fit) + (typeof ratio === 'number' ? ' · ~' + Math.round(ratio * 100) + '% of ' + (lib.budgetLabel || 'memory') : '') : 'Estimating…';
  return '<span class="fit-bar ' + fit + '" title="' + escAttr(tip) + '"><i style="width:' + pct + '%"></i></span>';
}

export function librarySortKey(g) {
  if (lib.sort === 'name') return g.title.toLowerCase() + ' ' + (g.detail || '').toLowerCase();
  if (lib.sort === 'size') return Math.min.apply(null, (g.quants || []).map(function (q) { return q.sizeBytes || 0; }).concat([Infinity]));
  const recent = lib.payload.recent || [];
  let best = recent.length + 1;
  (g.quants || []).forEach(function (q) {
    const i = recent.indexOf(q.path);
    if (i !== -1 && i < best) best = i;
  });
  return best;
}

export function sortLibraryGroups(groups) {
  return groups
    .map(function (g, i) { return { g: g, i: i, k: librarySortKey(g) }; })
    .sort(function (a, b) {
      if (a.k < b.k) return -1;
      if (a.k > b.k) return 1;
      return a.i - b.i;
    })
    .map(function (x) { return x.g; });
}

export function passesLibraryFilter(g, quant) {
  if (lib.filterMode === 'hidden') return g.hidden;
  if (g.hidden) return false;
  if (lib.filterMode === 'fits') return lib.fits[quant.path] === 'fits';
  if (lib.filterMode === 'draft') return !!lib.drafts[quant.path];
  if (lib.filterMode === 'vision') return groupHasVision(g);
  return true;
}

export function renderLibraryFilters() {
  const root = $('libraryFilters');
  if (!root) return;
  const st = libraryStats();
  if (lib.filterMode === 'hidden' && !st.hidden) lib.filterMode = 'all';
  const items: Array<[string, string, number]> = [
    ['all', 'All', st.total],
    ['fits', 'Fits', st.fits],
    ['draft', 'Drafts', st.drafts],
    ['vision', 'Vision', st.vision],
    ['hidden', 'Hidden', st.hidden],
  ];
  root.innerHTML = items
    .filter(function (it) { return it[0] === 'all' || it[0] === 'fits' || it[2] > 0; })
    .map(function (it) {
      return '<button type="button" class="chip' + (lib.filterMode === it[0] ? ' active' : '') + '" data-lib-filter="' + it[0] + '">' +
        it[1] + '<span class="n">' + it[2] + '</span></button>';
    }).join('');
}

export function shortModelLabel(found) {
  return found.group.title + ' · ' + found.quant.label;
}

export function renderLibraryQuick() {
  const root = $('libraryQuick');
  if (!root) return;
  const picks = (lib.payload.recent || [])
    .map(function (p) { return findLibraryQuant(p); })
    .filter(function (f) { return f && !f.group.hidden; })
    .slice(0, 3);
  if (picks.length < 2) {
    root.classList.add('hidden');
    root.innerHTML = '';
    return;
  }
  root.classList.remove('hidden');
  root.innerHTML = '<span class="qlabel">Recent</span>' + picks.map(function (f) {
    return '<button type="button" class="chip' + (f.quant.current ? ' active' : '') + '" data-quick-path="' + escAttr(f.quant.path) + '" title="' + escAttr(f.quant.path) + '">' +
      (f.quant.loaded ? '<span class="lib-dot live"></span>' : '') + escText(shortModelLabel(f)) + '</button>';
  }).join('');
}

export function renderLibraryJobs() {
  const root = $('libraryJobs');
  if (!root) return;
  if (!lib.downloads.length) {
    root.innerHTML = '';
    return;
  }
  let html = '';
  lib.downloads.forEach(function (job) {
    const pct = job.total > 0 ? Math.min(100, Math.round((job.received / job.total) * 100)) : (job.state === 'verifying' ? 100 : 8);
    const right = job.state === 'paused' ? 'paused' : job.state === 'gated' ? 'gated' : job.state === 'error' ? 'error' : pct + '%';
    html += '<div class="lib-job">';
    html += '<div class="lib-job-top"><span>' + escText(job.label) + '</span><span class="lib-meta">' + escText(right) + '</span></div>';
    if (job.state !== 'gated' && job.state !== 'error') {
      html += '<div class="lib-bar' + (job.state === 'paused' ? ' paused' : '') + '"><span style="width:' + pct + '%"></span></div>';
    }
    html += '<div class="lib-job-meta"><span>';
    if (job.state === 'running' || job.state === 'verifying' || job.state === 'queued') {
      const bits = [];
      if (job.bytesPerSec) bits.push(fmtBytes(job.bytesPerSec) + '/s');
      if (job.etaSeconds) bits.push(fmtEta(job.etaSeconds));
      if (job.total) bits.push(fmtBytes(job.received) + ' / ' + fmtBytes(job.total));
      html += escText(bits.join(' · ') || job.state);
    } else if (job.state === 'paused') {
      html += escText('Resumable' + (job.total ? ' · ' + fmtBytes(job.received) + ' / ' + fmtBytes(job.total) : ''));
    } else if (job.state === 'gated') {
      html += escText(job.error || 'Accept the license on Hugging Face, then set a token.');
    } else {
      html += escText(job.error || job.state);
    }
    html += '</span><span class="lib-actions" style="padding:0">';
    if (job.state === 'running' || job.state === 'queued' || job.state === 'verifying') {
      html += '<button type="button" class="secondary small" data-act="dl-pause" data-job="' + escAttr(job.id) + '">Pause</button>';
      html += '<button type="button" class="secondary small" data-act="dl-cancel" data-job="' + escAttr(job.id) + '">Cancel</button>';
    } else if (job.state === 'paused' || job.state === 'error') {
      html += '<button type="button" class="secondary small" data-act="dl-resume" data-job="' + escAttr(job.id) + '">Resume</button>';
      html += '<button type="button" class="secondary small" data-act="dl-cancel" data-job="' + escAttr(job.id) + '">Cancel</button>';
    } else if (job.state === 'gated') {
      if (job.pageUrl) html += '<button type="button" class="secondary small" data-act="dl-page" data-url="' + escAttr(job.pageUrl) + '">Open page</button>';
      html += '<button type="button" class="secondary small" data-act="dl-token">Set token</button>';
    }
    html += '</span></div></div>';
  });
  root.innerHTML = html;
  refreshLibrarySum();
}

export function renderLibraryList() {
  const root = $('libraryList');
  if (!root) return;
  renderLibraryJobs();
  const raw = (($('libraryFilter') && $('libraryFilter').value) || '').trim();
  const q = raw.toLowerCase();
  // Reset before filtering, so unhiding the last hidden model renders All
  // immediately instead of briefly rendering an empty Hidden view.
  if (lib.filterMode === 'hidden' && !libraryStats().hidden) lib.filterMode = 'all';
  let html = '';
  if (lib.hfBusy) {
    html += '<div class="hint">' + (lib.hfMode === 'files' ? 'Listing GGUF files…' : 'Searching Hugging Face…') + '</div>';
  }
  if (lib.hfError) html += '<div class="hint">' + escText(lib.hfError) + '</div>';
  if (lib.hfMode === 'license' && lib.hfFiles && lib.hfFiles.license) {
    const lic = lib.hfFiles.license;
    html += '<div class="lib-job"><div class="lib-job-top"><span>' + escText(lic.id) + '</span></div>';
    html += '<div class="hint">' + escText(lic.summary || 'Review this license before downloading.') + ' This is not legal advice.</div>';
    html += '<div class="lib-actions">';
    if (lic.licenseUrl) html += '<button type="button" class="secondary small" data-act="hf-license-view" data-url="' + escAttr(lic.licenseUrl) + '">View license</button>';
    html += '<button type="button" class="secondary small" data-act="hf-back">Cancel</button>';
    html += '<button type="button" class="secondary small" data-act="hf-license-ok" data-repo="' + escAttr(lic.id) + '">Download anyway</button>';
    html += '</div></div>';
  } else if (lib.hfMode === 'files' && lib.hfFiles) {
    html += '<div class="lib-detail" style="padding:0 2px 6px">' + escText(lib.hfFiles.modelId) + '</div>';
    if (lib.hfFiles.companionHint) {
      html += '<div class="hint">Also fetches ' + escText(lib.hfFiles.companionHint) + '</div>';
    }
    (lib.hfFiles.files || []).forEach(function (file) {
      html += '<div class="lib-file"><span class="grow"><b>' + escText(file.label) + '</b>';
      if (file.detail) html += '<span class="lib-detail">' + escText(file.detail) + '</span>';
      html += '</span><button type="button" class="secondary small" data-act="hf-download" data-model="' + escAttr(lib.hfFiles.modelId) + '" data-file="' + escAttr(file.path) + '">Download</button></div>';
    });
    html += '<button type="button" class="secondary lib-wide" data-act="hf-back">Back to search</button>';
  } else if (lib.hfMode === 'search') {
    if (!lib.hfRepos.length && !lib.hfBusy && !lib.hfError) html += '<div class="hint">No GGUF repos for “' + escText(raw) + '”.</div>';
    lib.hfRepos.forEach(function (repo) {
      html += '<button type="button" class="secondary lib-wide" data-act="hf-repo" data-repo="' + escAttr(repo.id) + '">';
      html += '<b>' + escText(repo.id) + '</b><br><span class="lib-detail">' + escText((repo.badge || '') + (repo.downloads ? ' · ' + fmtCount(repo.downloads) : '')) + '</span></button>';
    });
    html += '<button type="button" class="secondary lib-wide" data-act="hf-back">Back to local models</button>';
  } else {
    const groups = sortLibraryGroups(allLibraryGroups()).map(function (g) {
      const blob = (g.title + ' ' + g.detail + ' ' + g.source).toLowerCase();
      const quants = (g.quants || []).filter(function (x) {
        if (!passesLibraryFilter(g, x)) return false;
        const filename = x.path.split(/[/\\]/).pop() || '';
        return !q || (blob + ' ' + x.label.toLowerCase() + ' ' + filename.toLowerCase()).indexOf(q) !== -1;
      });
      return { g: g, quants: quants };
    }).filter(function (x) { return x.quants.length; });
    if (!groups.length) {
      html += '<div class="hint" style="margin:0 0 8px">' + (q || lib.filterMode !== 'all' ? 'No local model matches.' : 'No models in the library yet.') + '</div>';
    } else {
      const total = groups.reduce((sum, x) => sum + x.quants.length, 0);
      lib.page = Math.max(0, Math.min(lib.page, Math.ceil(total / LIBRARY_PAGE_SIZE) - 1));
      const start = lib.page * LIBRARY_PAGE_SIZE;
      let offset = 0;
      html += '<div class="lib-list">';
      groups.forEach(function (x) {
        const visible = x.quants.slice(Math.max(0, start - offset), Math.max(0, start + LIBRARY_PAGE_SIZE - offset));
        offset += x.quants.length;
        if (!visible.length) return;
        const g = x.g;
        const vision = groupHasVision(g);
        const sub = [g.detail || '', g.source && g.source !== 'Llama AIO' ? g.source : '', g.hidden ? 'hidden' : ''].filter(Boolean).join(' · ');
        html += '<div class="lib-grp" title="' + escAttr(g.title + (g.detail ? ' — ' + g.detail : '') + ' · ' + g.source) + '"><span class="lib-title">' + escText(g.title) + '</span>' + (vision ? '<span class="lib-tag vision" title="A vision projector (mmproj) sits next to these files">vision</span>' : '') + '<span class="lib-detail">' + escText(sub) + '</span></div>';
        visible.forEach(function (quant) {
          const sel = quant.path === lib.selected;
          const draft = !!lib.drafts[quant.path];
          const fit = lib.fits[quant.path] || '';
          const tip = quant.path + (quant.shardCount > 1 ? '\n' + quant.shardCount + ' shards' : '') + (fit ? '\n' + fitText(fit) : '');
          html += '<div class="lib-quant' + (sel ? ' selected' : '') + (fit === 'wont-fit' ? ' nofit' : '') + '" role="button" tabindex="0" title="' + escAttr(tip) + '" data-path="' + escAttr(quant.path) + '" data-current="' + (quant.current ? '1' : '0') + '">';
          html += '<span class="lib-dot' + (quant.loaded ? ' live' : '') + '"' + (quant.loaded ? ' title="Running on the server"' : '') + '></span>';
          html += '<span class="lib-label">' + escText(quant.label);
          if (draft) html += '<span class="lib-tag draft" title="DFlash draft model: use it for speculative decoding, not as the main model">draft</span>';
              html += '</span>';
          html += '<span class="lib-meta">' + escText(quant.sizeLabel || '') + '</span>';
          html += fitBarHtml(quant.path);
          html += '<button type="button" class="lib-more" data-menu-path="' + escAttr(quant.path) + '" title="More actions" aria-haspopup="menu">⋯</button>';
          if (lib.menuPath === quant.path) {
            html += '<div class="lib-menu" role="menu">';
            if (draft) html += '<button type="button" role="menuitem" data-act="use-draft" data-path="' + escAttr(quant.path) + '">Use as draft for current model</button><div class="menu-sep"></div>';
            html += '<button type="button" role="menuitem" data-act="reveal" data-path="' + escAttr(quant.path) + '">Reveal in file manager</button>';
            html += '<button type="button" role="menuitem" data-act="copy-path" data-path="' + escAttr(quant.path) + '">Copy path</button>';
            html += '<button type="button" role="menuitem" data-act="' + (g.hidden ? 'unhide' : 'hide') + '" data-path="' + escAttr(quant.path) + '">' + (g.hidden ? 'Unhide' : 'Hide from library') + '</button>';
            if (quant.owned) {
              html += '<div class="menu-sep"></div><button type="button" role="menuitem" class="danger-item" data-act="remove" data-path="' + escAttr(quant.path) + '">Remove from disk…</button>';
            }
            html += '</div>';
          }
          html += '</div>';
        });
      });
      html += '</div>';
      if (total > LIBRARY_PAGE_SIZE) {
        html += '<div class="lib-pager" role="group" aria-label="Model library pages">' +
          '<button type="button" class="secondary small" data-act="lib-prev"' + (lib.page === 0 ? ' disabled' : '') + '>Previous</button>' +
          '<span role="status">' + (start + 1) + '–' + Math.min(total, start + LIBRARY_PAGE_SIZE) + ' of ' + total + '</span>' +
          '<button type="button" class="secondary small" data-act="lib-next"' + (start + LIBRARY_PAGE_SIZE >= total ? ' disabled' : '') + '>Next</button></div>';
      }
    }
    if (q.length >= 2) {
      html += '<button type="button" class="secondary lib-wide" data-act="hf-search">Search Hugging Face for “' + escText(raw) + '”</button>';
    }
  }
  root.innerHTML = html;
  renderLibraryFilters();
  renderLibraryQuick();
  refreshLibrarySum();
}

export function renderLibraryPartials() {
  const el = $('libraryPartials');
  if (!el) return;
  const n = (lib.payload.partials || []).length;
  if (!n) {
    el.classList.add('hidden');
    el.textContent = '';
    return;
  }
  el.classList.remove('hidden');
  const label = lib.payload.partialSizeLabel || '';
  el.innerHTML = '<span>' + n + ' partial download' + (n === 1 ? '' : 's') + (label ? ' · ' + escText(label) : '') + '</span><button type="button" class="secondary small" data-act="clean-partials">Clean</button>';
}

export function renderLibraryDialog() {
  const el = $('libraryDialog');
  if (!el) return;
  const found = lib.dialogPath ? findLibraryQuant(lib.dialogPath) : null;
  if (!found || !found.quant.owned) {
    lib.dialogPath = '';
    el.classList.add('hidden');
    el.innerHTML = '';
    return;
  }
  const quant = found.quant;
  const companions = found.group.companions || [];
  let html = '<div class="lib-dialog-title">' + (quant.loaded ? 'Stop and remove' : 'Remove') + ' ' + escText(quant.label) + '</div>';
  if (lib.payload.serverStarting) {
    html += '<div class="hint">The server is still starting. Wait, then remove this model.</div>';
  } else if (quant.loaded) {
    html += '<div class="hint">The server has this file mapped. It will stop before the files are removed.</div>';
  } else {
    html += '<div class="hint">Removes this quant from the Llama AIO models folder.</div>';
  }
  html += '<div class="hint">Files go to the trash when the system allows it.</div>';
  companions.forEach(function (c) {
    const checked = Object.prototype.hasOwnProperty.call(lib.dialogChecks, c.path) ? lib.dialogChecks[c.path] : !c.shared;
    html += '<label class="lib-check"><input type="checkbox" data-companion="' + escAttr(c.path) + '"' + (checked ? ' checked' : '') + ' /> ';
    html += escText(c.name) + ' · ' + escText(c.sizeLabel || '') + (c.shared ? ' · still used by other quants' : '') + '</label>';
  });
  html += '<div class="lib-actions"><button type="button" class="secondary small" data-act="cancel-remove">Cancel</button>';
  html += '<button type="button" class="danger small" data-act="confirm-remove"' + (lib.payload.serverStarting ? ' disabled' : '') + '>' + (quant.loaded ? 'Stop and remove' : 'Remove') + '</button></div>';
  el.classList.remove('hidden');
  el.innerHTML = html;
}

export function startLibrarySearch() {
  const raw = (($('libraryFilter') && $('libraryFilter').value) || '').trim();
  if (raw.length < 2) return;
  resetLibraryNavigation('search');
  lib.hfBusy = true;
  lib.hfError = '';
  lib.hfRepos = [];
  renderLibraryList();
  vscode.postMessage({ type: 'searchLibraryHf', query: raw, requestId: lib.hfRequest });
}

export function openLibraryMenu(filePath) {
  lib.menuPath = lib.menuPath === filePath ? '' : filePath;
  renderLibraryList();
  const first = lib.menuPath && document.querySelector<HTMLElement>('.lib-menu button');
  if (first) first.focus();
}

export function closeLibraryMenu() {
  if (!lib.menuPath) return;
  lib.menuPath = '';
  renderLibraryList();
}

export function onLibraryClick(e) {
  const more = e.target.closest ? e.target.closest('[data-menu-path]') : null;
  if (more) {
    e.stopPropagation();
    openLibraryMenu(more.getAttribute('data-menu-path') || '');
    return;
  }
  const btn = e.target.closest ? e.target.closest('[data-act]') : null;
  if (btn) {
    const act = btn.getAttribute('data-act');
    const filePath = btn.getAttribute('data-path') || '';
    if (btn.closest('.lib-menu')) lib.menuPath = '';
    if (act === 'reveal') vscode.postMessage({ type: 'revealInOs', path: filePath });
    else if (act === 'lib-prev' || act === 'lib-next') {
      lib.page += act === 'lib-next' ? 1 : -1;
      lib.menuPath = '';
      renderLibraryList();
      // Rendering replaces the activated button. Keep keyboard navigation on
      // the pager, including at the first/last page where that button disables.
      const pagerButton = document.querySelector<HTMLButtonElement>('[data-act="' + act + '"]:not(:disabled)') ||
        document.querySelector<HTMLButtonElement>('.lib-pager button:not(:disabled)');
      pagerButton?.focus();
    }
    else if (act === 'copy-path') vscode.postMessage({ type: 'copyPath', path: filePath });
    else if (act === 'use-draft') vscode.postMessage({ type: 'useLibraryDraft', path: filePath });
    else if (act === 'hide') vscode.postMessage({ type: 'hideLibraryModel', path: filePath });
    else if (act === 'unhide') vscode.postMessage({ type: 'unhideLibraryModel', path: filePath });
    else if (act === 'remove') {
      lib.selected = filePath;
      lib.dialogPath = filePath;
      lib.dialogChecks = {};
      renderLibraryList();
      renderLibraryDialog();
    } else if (act === 'hf-search') {
      startLibrarySearch();
    } else if (act === 'hf-back') {
      resetLibraryNavigation(lib.hfMode === 'files' || lib.hfMode === 'license' ? 'search' : 'local');
      renderLibraryList();
    } else if (act === 'hf-repo') {
      const id = btn.getAttribute('data-repo') || '';
      const repo = lib.hfRepos.find(function (item) { return item.id === id; });
      if (repo && repo.needsConfirm) {
        lib.hfFiles = { license: repo };
        lib.hfMode = 'license';
        renderLibraryList();
      } else if (id) {
        requestLibraryFiles(id);
      }
    } else if (act === 'hf-license-view') {
      const url = btn.getAttribute('data-url') || '';
      if (url) vscode.postMessage({ type: 'openExternal', url: url });
    } else if (act === 'hf-license-ok') {
      const id = btn.getAttribute('data-repo') || '';
      requestLibraryFiles(id);
    } else if (act === 'hf-download') {
      resetLibraryNavigation();
      if ($('libraryFilter')) $('libraryFilter').value = '';
      renderLibraryList();
      vscode.postMessage({
        type: 'downloadLibraryHf',
        modelId: btn.getAttribute('data-model') || '',
        filePath: btn.getAttribute('data-file') || ''
      });
    }         else if (act === 'dl-pause') vscode.postMessage({ type: 'pauseDownload', id: btn.getAttribute('data-job') || '' });
    else if (act === 'dl-resume') vscode.postMessage({ type: 'resumeDownload', id: btn.getAttribute('data-job') || '' });
    else if (act === 'dl-cancel') vscode.postMessage({ type: 'cancelDownload', id: btn.getAttribute('data-job') || '' });
    else if (act === 'dl-page') {
      const url = btn.getAttribute('data-url') || '';
      if (url) vscode.postMessage({ type: 'openExternal', url: url });
    } else if (act === 'dl-token') vscode.postMessage({ type: 'setHfToken' });
    if (btn.closest('.lib-menu')) renderLibraryList();
    return;
  }
  if (e.target.closest && e.target.closest('.lib-menu')) return;
  const row = e.target.closest ? e.target.closest('.lib-quant') : null;
  if (!row) return;
  selectLibraryPath(row.getAttribute('data-path') || '');
}

export function selectLibraryPath(filePath) {
  if (!filePath) return;
  const found = findLibraryQuant(filePath);
  lib.selected = filePath;
  lib.menuPath = '';
  renderLibraryList();
  if (!(found && found.quant.current)) vscode.postMessage({ type: 'useLibraryModel', path: filePath });
}
