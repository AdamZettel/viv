// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The behavior of the page chrome, shared by every page (Base.astro):
// the two side menus, the search box over /search-index.json, and a copy
// button on every code block. Bundled by Astro, so it adds no inline script
// to the Content-Security-Policy.

function initMenus() {
  const buttons = [...document.querySelectorAll('.side-btn')];
  const close = (except) => {
    for (const b of buttons) {
      if (b === except) continue;
      b.setAttribute('aria-expanded', 'false');
      document.getElementById(b.getAttribute('aria-controls')).hidden = true;
    }
  };
  for (const b of buttons) {
    const menu = document.getElementById(b.getAttribute('aria-controls'));
    b.addEventListener('click', () => {
      const open = b.getAttribute('aria-expanded') !== 'true';
      close(b);
      b.setAttribute('aria-expanded', String(open));
      menu.hidden = !open;
      if (open) menu.querySelector('a')?.focus({ preventScroll: true });
    });
    menu.addEventListener('keydown', (e) => {
      const links = [...menu.querySelectorAll('a')];
      const i = links.indexOf(document.activeElement);
      if (e.key === 'ArrowDown') links[(i + 1) % links.length].focus(), e.preventDefault();
      else if (e.key === 'ArrowUp') links[(i - 1 + links.length) % links.length].focus(), e.preventDefault();
      else if (e.key === 'Escape') close(), b.focus();
    });
  }
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.side')) close();
  });
  document.addEventListener('focusin', (e) => {
    if (!e.target.closest('.side')) close();
  });
}

function initTheme() {
  document.querySelector('.theme-btn')?.addEventListener('click', () => {
    const root = document.documentElement;
    const next = root.dataset.theme === 'light' ? 'dark' : 'light';
    root.dataset.theme = next;
    try {
      localStorage.setItem('viv-theme', next);
    } catch {}
  });
}

// ---- search ------------------------------------------------------------------
// A page index of a few hundred entries, scored in the browser: every word of
// the query must appear; a hit in the title outranks one in the text.
/** A result's text cut to about SNIPPET characters at a word, with an ellipsis when cut. */
const SNIPPET = 120;
const snippet = (t) => {
  if (t.length <= SNIPPET) return t;
  const cut = t.slice(0, SNIPPET);
  const sp = cut.lastIndexOf(' ');
  return (sp > SNIPPET / 2 ? cut.slice(0, sp) : cut).replace(/[\s,;:.]+$/, '') + '…';
};
let index = null;
async function loadIndex() {
  if (index) return index;
  const r = await fetch('/search-index.json');
  const data = await r.json();
  index = data.pages.map((p) => ({ ...p, t: p.title.toLowerCase(), k: (p.kind + ' ' + (p.text || '')).toLowerCase() }));
  return index;
}
function score(p, words) {
  let s = 0;
  for (const w of words) {
    const ti = p.t.indexOf(w);
    if (p.t === w || p.t.endsWith('.' + w)) s += 16;
    else if (ti >= 0) s += ti === 0 ? 12 : p.t[ti - 1] === '.' || p.t[ti - 1] === ' ' ? 8 : 5;
    else if (p.k.includes(w)) s += 1;
    else return 0;
  }
  return s + (p.boost || 0);
}
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

function initSearch() {
  const head = document.querySelector('.site-head');
  const input = document.querySelector('.search input');
  const list = document.getElementById('search-results');
  if (!input || !list) return;
  let sel = -1;
  let hits = [];
  const show = (on) => {
    list.hidden = !on;
    input.setAttribute('aria-expanded', String(on));
  };
  const render = () => {
    if (!input.value.trim()) return show(false);
    list.innerHTML = hits.length
      ? hits
          .map(
            (p, i) =>
              `<li><a id="sr-${i}" role="option" href="${esc(p.url)}" aria-selected="${i === sel}"><span class="sr-kind">${esc(p.kind)}</span><b>${esc(p.title)}</b>${p.text ? `<p>${esc(snippet(p.text))}</p>` : ''}</a></li>`,
          )
          .join('')
      : `<li class="sr-empty">Nothing matches "${esc(input.value.trim())}". Try a function name, such as lqr, or a topic, such as Hartree-Fock.</li>`;
    if (sel >= 0) input.setAttribute('aria-activedescendant', `sr-${sel}`);
    else input.removeAttribute('aria-activedescendant');
    show(true);
  };
  const run = async () => {
    const q = input.value.toLowerCase().trim();
    const words = q.split(/\s+/).filter(Boolean);
    const idx = await loadIndex();
    hits = words.length
      ? idx
          .map((p) => [score(p, words), p])
          .filter(([s]) => s > 0)
          .sort((a, b) => b[0] - a[0])
          .slice(0, 12)
          .map(([, p]) => p)
      : [];
    sel = hits.length ? 0 : -1;
    render();
  };
  input.addEventListener('focus', loadIndex, { once: true });
  input.addEventListener('input', run);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' && hits.length) (sel = (sel + 1) % hits.length), render(), e.preventDefault();
    else if (e.key === 'ArrowUp' && hits.length) (sel = (sel - 1 + hits.length) % hits.length), render(), e.preventDefault();
    else if (e.key === 'Enter' && hits[sel]) location.href = hits[sel].url;
    else if (e.key === 'Escape') {
      input.value = '';
      show(false);
      head.classList.remove('searching');
      input.blur();
    }
  });
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.search') && !e.target.closest('.search-open')) {
      show(false);
      head.classList.remove('searching');
    }
  });
  document.querySelector('.search-open')?.addEventListener('click', () => {
    head.classList.add('searching');
    input.focus();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key !== '/' || e.metaKey || e.ctrlKey) return;
    const t = e.target;
    if (t.closest('input, textarea, select, [contenteditable], .monaco-editor, .cm-editor')) return;
    e.preventDefault();
    head.classList.add('searching');
    input.focus();
  });
}

// ---- copy buttons --------------------------------------------------------------
function initCopy() {
  for (const pre of document.querySelectorAll('pre.code')) {
    if (pre.querySelector('.copy-btn')) continue;
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'copy-btn';
    b.textContent = 'Copy';
    b.setAttribute('aria-label', 'Copy this code');
    b.addEventListener('click', async () => {
      const code = pre.querySelector('code') || pre;
      // a console session copies its commands, not the prompts or the output
      let text = code.innerText;
      if (pre.dataset.lang === 'console')
        text = text.split('\n').filter((l) => /^\s*\$ /.test(l)).map((l) => l.replace(/^\s*\$ /, '')).join('\n') || text;
      try {
        await navigator.clipboard.writeText(text.replace(/\n$/, ''));
        b.textContent = 'Copied';
      } catch {
        b.textContent = 'Select and copy';
      }
      b.dataset.done = '';
      setTimeout(() => {
        b.textContent = 'Copy';
        delete b.dataset.done;
      }, 1600);
    });
    pre.appendChild(b);
  }
}

// ---- "On this page": mark the section in view -------------------------------------
// The section in view is the last heading above a line a quarter of the way
// down the window; above the first heading, none is.
function initTocSpy() {
  const links = [...document.querySelectorAll('.toc a[href^="#"]')];
  if (!links.length) return;
  const heads = links
    .map((a) => [document.getElementById(decodeURIComponent(a.getAttribute('href').slice(1))), a])
    .filter(([el]) => el);
  let queued = false;
  const mark = () => {
    queued = false;
    const line = window.innerHeight * 0.25;
    let on = null;
    for (const [el, a] of heads) if (el.getBoundingClientRect().top <= line) on = a;
    for (const a of links) if (a !== on) a.removeAttribute('aria-current');
    if (on) on.setAttribute('aria-current', 'location');
  };
  const queue = () => {
    if (!queued) {
      queued = true;
      requestAnimationFrame(mark);
    }
  };
  window.addEventListener('scroll', queue, { passive: true });
  window.addEventListener('resize', queue);
  mark();
}

// The section tree folds on a narrow screen: open on wide ones, closed on a phone.
function initTree() {
  const fold = document.querySelector('.sidetree-fold');
  if (fold && !window.matchMedia('(min-width: 900px)').matches) fold.open = false;
}

// On a phone the second row scrolls sideways: start it with the current page in
// view, scrolled no further than that takes, so the first links stay in sight
// when they can. The scroll stops where a link begins, so the link at the left
// edge shows whole, just past the edge's fade, and is not cut in half.
const SUBNAV_SLACK = 24;
function initSubnav() {
  const bar = document.querySelector('.sections');
  const cur = bar?.querySelector('[aria-current="page"]');
  if (!cur || bar.scrollWidth <= bar.clientWidth) return;
  const box = bar.getBoundingClientRect();
  const x = (el) => el.getBoundingClientRect().left - box.left + bar.scrollLeft;
  const right = x(cur) + cur.offsetWidth + SUBNAV_SLACK;
  if (right <= bar.clientWidth) return;
  const pad = parseFloat(getComputedStyle(bar).paddingLeft) || 0;
  const need = right - bar.clientWidth;
  const start = [...bar.children].map((el) => x(el) - pad).find((s) => s >= need);
  bar.scrollLeft = start ?? need;
}

export function initChrome() {
  initTree();
  initSubnav();
  initMenus();
  initTheme();
  initSearch();
  initCopy();
  initTocSpy();
}
