// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The five tools for the Netlify Function, read from what the site build wrote
// instead of the feed's database.
//
// The function's directory (build-function.mjs writes it) holds:
//   viv.wasm            the interpreter the site serves
//   run-worker.mjs      the worker one run gets (fn-run-worker.mjs, bundled)
//   modules/            lib/, dft/, pde/, examples/ (lang/viv/examples) and posts/, for imports
//   reference/          the prompt pack: viv-prompt-pack.md, llms-full.txt, llms.txt
//   data/posts.json     the published posts (feed.json), each with its lineage summary
//   data/corpus.json    the search documents: search.mjs's repoDocs plus the posts
//
// Differences from the server (README "The Netlify Function"):
//   - a run may print at most 8 MB (fn-run-worker.mjs says why);
//   - a run stops at 8 s (a synchronous function is cut off at about 10 s), and a
//     check_post with a `from:` claim shares those 8 s between its runs;
//   - search is jsearch.mjs over corpus.json, not FTS5;
//   - posts are the committed ones the build published; there is no database,
//     so a post published on the site later is not seen until the next deploy;
//   - the run budget per address is kept per function instance, and Netlify
//     starts as many instances as it likes, so it bounds a burst, not an hour.
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { runProgram, configureRuns, RunTimeout, RunFailed, RunBusy, RunAborted } from '../run.mjs';
import { expandUses, setCompiler } from '../../www/src/lib/uses.js';
import { judge, headerOf } from '../../www/src/lib/postCheck.js';
import { isAnimRecord, buildTimeline } from '../../www/src/lib/animParse.js';
import { checkEdge } from '../../www/src/lib/lineage.js';
import { capMemory } from './wasmcap.mjs';
import { createJsonSearch } from './jsearch.mjs';
import { reference, useFiles } from './reference.mjs';
import { toolDefs } from './tooldefs.mjs';
import { fail, checkSource, runResult, checkPostResult, searchResult, postResult, SLUG_ARG } from './toolcore.mjs';
import { Window } from './protocol.mjs';

const DIRS = { dft: 'dft', pde: 'pde', lib: 'lib', examples: 'examples', posts: 'posts' };

/**
 * Options: dir (the function's directory), runTimeoutMs (default 8000),
 * memoryMb (one run's wasm memory, default 384), runsPerWindow and windowMs
 * (the per-address run budget of one instance, default 30 in 10 minutes).
 */
export function createFnTools({ dir, runTimeoutMs = 8000, memoryMb = 384, runsPerWindow = 30, windowMs = 600_000 }) {
  const data = (f) => JSON.parse(readFileSync(join(dir, 'data', f), 'utf8'));
  useFiles({ pack: join(dir, 'reference', 'viv-prompt-pack.md'), full: join(dir, 'reference', 'llms-full.txt'), index: join(dir, 'reference', 'llms.txt') });
  // one run at a time per instance (a function instance serves one request at a time), two for a claim's pair
  configureRuns({ slots: 2, queue: 2, memoryMb });

  let posts = null;
  let search = null;
  const postsData = () => (posts ||= data('posts.json'));
  const searchIndex = () => (search ||= createJsonSearch(data('corpus.json').docs));

  // the interpreter, compiled once per instance with the memory cap written in;
  // an instance of it in this thread expands imports (uses.js asks the compiler)
  let compiled = null;
  const moduleWithCap = () => (compiled ||= WebAssembly.compile(capMemory(readFileSync(join(dir, 'viv.wasm')), memoryMb * 1024 * 1024)).then(async (m) => {
    setCompiler((await WebAssembly.instantiate(m, {})).exports);
    return m;
  }));

  const resolve = async (name) => {
    if (name.startsWith('post:')) {
      const p = postsData().posts[name.slice(5)];
      if (!p || p.source == null) throw new Error(`module \`${name}\`: no published post by that slug`);
      return p.source;
    }
    const [head, ...rest] = name.split('/');
    if (!DIRS[head] || !rest.length || rest.some((x) => x === '..' || x === '' || x === '.')) throw new Error(`module \`${name}\`: modules live under ${Object.keys(DIRS).join(', ')}`);
    const file = join(dir, 'modules', DIRS[head], `${rest.join('/')}.viv`);
    if (!existsSync(file)) throw new Error(`module \`${name}\`: no module by that name`);
    return readFileSync(file, 'utf8');
  };

  const runs = new Window(runsPerWindow, windowMs);
  const timeoutText = (ms) => `the program ran for more than ${ms / 1000} s and was stopped. This endpoint runs on a Netlify Function, which stops a run at ${runTimeoutMs / 1000} s; make the problem smaller, or run it at https://viv-lang.org/workbench, where a run has no such limit.`;

  async function withRun(ip, fn) {
    if (!runs.take(ip)) return fail(`this address has used its ${runsPerWindow} program runs for now (run_viv and check_post each count); wait a few minutes and try again`);
    try {
      return await fn();
    } catch (e) {
      if (e instanceof RunTimeout) return { ...fail(timeoutText(runTimeoutMs)), data: { ok: false, error: 'timeout', timeoutSeconds: runTimeoutMs / 1000 } };
      if (e instanceof RunFailed || e instanceof RunBusy || e instanceof RunAborted) return fail(e.message);
      if (e instanceof WebAssembly.RuntimeError || (e && e.name === 'RuntimeError')) return fail(`the interpreter stopped (${e.message}): the program most likely needed more than the ${memoryMb} MB of memory a run may use`);
      throw e;
    }
  }

  // check_post's verdict: the server's verify (app.mjs) with the build's posts in place of the database
  async function verify(src, deadline) {
    const module = await moduleWithCap();
    let expanded;
    try {
      expanded = await expandUses(src, resolve);
    } catch (e) {
      return { errors: [e.message], tests: [], failed: [], figure: '', shown: [], seconds: null };
    }
    if (Buffer.byteLength(expanded) > 4 * 200 * 1024) return { errors: ['with its modules the program is over 800 KB'], tests: [], failed: [], shown: [], seconds: null };
    const h = headerOf(src);
    let run;
    try {
      run = await runProgram(expanded, { timeoutMs: Math.max(500, deadline - Date.now()), ladder: h.kind === 'ladder', module, waitMs: 1000 });
    } catch (e) {
      if (e instanceof RunTimeout) return { errors: [timeoutText(runTimeoutMs)], tests: [], failed: [], shown: [], seconds: null };
      if (e instanceof RunFailed) return { errors: [e.message], tests: [], failed: [], shown: [], seconds: null };
      throw e;
    }
    const extra = {};
    if (run.ladder) extra.ladder = run.ladder;
    if (h.solves) {
      const problem = postsData().posts[h.solves];
      extra.problem = problem && problem.source != null ? problem.source : null;
    }
    if (h.channel) extra.channels = postsData().channels;
    const v = judge(src, run.out, run.seconds, extra);
    const anim = run.out.split('\n').filter(isAnimRecord);
    if (anim.length) for (const e of buildTimeline(anim).errors) v.errors.push(`animation: ${e}`);
    const addresses = [...run.hash.matchAll(/^([0-9a-f]{8})\s+body ([0-9a-f]{8})\s+(.+)$/gm)].map((x) => `${x[2]} ${x[3]}`).sort();
    v.address = createHash('sha256').update(addresses.join('\n')).digest('hex').slice(0, 10);
    v.out = run.out;
    return v;
  }

  const handlers = {
    async run_viv({ source }, { ip }) {
      const bad = checkSource(source);
      if (bad) return fail(bad);
      return withRun(ip, async () => {
        const module = await moduleWithCap();
        let expanded;
        try {
          expanded = await expandUses(source, resolve);
        } catch (e) {
          return fail(e.message);
        }
        return runResult(await runProgram(expanded, { timeoutMs: runTimeoutMs, module, waitMs: 1000 }));
      });
    },

    async check_post({ source }, { ip }) {
      const bad = checkSource(source);
      if (bad) return fail(bad);
      return withRun(ip, async () => {
        const deadline = Date.now() + runTimeoutMs;
        const v = await verify(source, deadline);
        const h = headerOf(source);
        let claim = null;
        if (h.from && !v.errors.length) {
          const kind = h.edge || 'fork';
          const parent = postsData().posts[h.from];
          if (!parent || parent.source == null) claim = { kind, parent: h.from, ok: false, reasons: [`\`from: ${h.from}\`: there is no published post with that slug`] };
          else {
            const runner = async (s) => (await runProgram(await expandUses(s, resolve), { timeoutMs: Math.max(500, deadline - Date.now()), module: await moduleWithCap(), waitMs: 1000 })).out;
            const c = await checkEdge(kind, { src: parent.source }, { src: source, out: v.out || '' }, runner);
            claim = { kind, parent: h.from, ok: c.ok, reasons: c.reasons };
          }
        }
        return checkPostResult(v, h, claim);
      });
    },

    async search(args) {
      return searchResult(args, searchIndex().search);
    },

    async get_post({ slug }) {
      if (typeof slug !== 'string' || !SLUG_ARG.test(slug)) return fail('slug must be lowercase letters, digits and hyphens');
      const p = postsData().posts[slug];
      if (!p) return fail(`no published post "${slug}"; the search tool finds posts by their words`);
      const { source, ...fields } = p;
      return postResult({ ...fields }, { source });
    },

    async viv_reference({ section = 'core', topic = null } = {}) {
      const r = reference({ section, topic });
      if (r.error) return fail(r.error);
      return { text: r.text, data: null, isError: false };
    },
  };

  return { tools: toolDefs({ runSeconds: runTimeoutMs / 1000 }), handlers, warm: moduleWithCap };
}
