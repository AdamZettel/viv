// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The Viv prompt pack, as the viv_reference tool and the resources serve it.
//
// The pack is built with the site (www/scripts/llms-build.mjs, which checks
// every example and habit in it against the interpreter). The server reads the
// built files. When they are missing -- a server started from a checkout where
// the site was never built -- it runs that build once, since the build needs
// nothing the server does not already have (viv.wasm and the site's data).
import { readFileSync, existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const WWW = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'www');
export const FILES = {
  pack: join(WWW, 'public', 'viv-prompt-pack.md'),
  full: join(WWW, 'public', 'llms-full.txt'),
  index: join(WWW, 'public', 'llms.txt'),
};

// The pack's numbered sections, by the name the tool takes.
export const SECTIONS = {
  grammar: 'Grammar and types',
  habits: 'Python habits that do not work',
  traps: 'The interpreter runs these; the C compiler refuses them',
  builtins: 'Builtins',
  examples: 'Examples, by topic',
  posts: 'Writing a post',
};

let cache = null;

/** Read the pack from other paths (the Netlify Function bundles its own copy). */
export function useFiles(files) {
  Object.assign(FILES, files);
  cache = null;
}

function ensureBuilt() {
  if (Object.values(FILES).every(existsSync)) return;
  const r = spawnSync(process.execPath, [join(WWW, 'scripts', 'llms-build.mjs')], { encoding: 'utf8', cwd: WWW });
  if (r.status !== 0) throw new Error(`the prompt pack is not built and building it failed: ${(r.stderr || r.stdout || '').trim().split('\n').slice(0, 3).join(' ')}`);
}

export function load() {
  if (cache) return cache;
  ensureBuilt();
  const pack = readFileSync(FILES.pack, 'utf8');
  const parts = pack.split(/^(?=## \d+\. )/m);
  const sections = {};
  for (const [name, heading] of Object.entries(SECTIONS)) {
    const part = parts.find((p) => p.startsWith('## ') && p.slice(0, p.indexOf('\n')).endsWith(heading));
    if (!part) throw new Error(`the prompt pack has no section "${heading}"; rebuild it (npm run llms in www/)`);
    sections[name] = part.trim();
  }
  const topics = {};
  for (const t of sections.examples.split(/^(?=### )/m).slice(1)) topics[t.slice(4, t.indexOf('\n')).trim()] = t.trim();
  cache = { pack, preamble: parts[0].trim(), sections, topics, full: readFileSync(FILES.full, 'utf8'), index: readFileSync(FILES.index, 'utf8') };
  return cache;
}

/** What viv_reference returns for a section name, or for an example topic. */
export function reference({ section = 'core', topic = null } = {}) {
  const p = load();
  const toc = [
    'Sections of the Viv reference (ask for one by name with viv_reference):',
    ...Object.entries(SECTIONS).map(([k, v]) => `- ${k}: ${v}`),
    `- examples with a topic, one of: ${Object.keys(p.topics).join('; ')}`,
    `- all: the whole prompt pack (about ${Math.round(Buffer.byteLength(p.pack) / 3040)}k tokens)`,
  ].join('\n');
  if (topic) {
    const key = Object.keys(p.topics).find((t) => t.toLowerCase() === String(topic).toLowerCase()) || Object.keys(p.topics).find((t) => t.toLowerCase().includes(String(topic).toLowerCase()));
    if (!key) return { error: `no example topic "${topic}". ${toc}` };
    return { text: p.topics[key] };
  }
  if (section === 'all') return { text: p.pack };
  if (section === 'core') return { text: [p.preamble, p.sections.grammar, p.sections.habits, p.sections.traps, toc].join('\n\n') };
  if (!p.sections[section]) return { error: `no section "${section}". ${toc}` };
  return { text: p.sections[section] };
}
