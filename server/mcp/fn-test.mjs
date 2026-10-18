// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Test the built Netlify Function without Netlify: copy the directory
// build-function.mjs wrote somewhere else (so a file it forgot to carry shows up
// as a failure), import it, and hand its default export Fetch API Requests the
// way Netlify does.
//
//     node server/mcp/fn-test.mjs                 the protocol and every tool, printed as a transcript
//     node server/mcp/fn-test.mjs --serve 8788    the function on http://127.0.0.1:8788/mcp, for
//                                                 curl, smoke.mjs or the MCP inspector
//
// Exits 1 when a check fails.
import { cpSync, rmSync, mkdtempSync } from 'node:fs';
import { createServer } from 'node:http';
import { join, dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const BUILT = join(HERE, '..', '..', 'www', 'netlify', 'functions', 'mcp');
const copy = join(mkdtempSync(join(process.env.TMPDIR || tmpdir(), 'viv-mcpfn-')), 'mcp');
cpSync(BUILT, copy, { recursive: true });
const { default: handler } = await import(pathToFileURL(join(copy, 'mcp.mjs')).href);
const URL_ = 'https://viv-lang.org/mcp';

const serveAt = process.argv.indexOf('--serve');
if (serveAt > 0) {
  const port = Number(process.argv[serveAt + 1] || 8788);
  createServer(async (req, res) => {
    const chunks = [];
    for await (const c of req) chunks.push(c);
    const body = Buffer.concat(chunks);
    const r = await handler(new Request(`http://127.0.0.1:${port}${req.url}`, { method: req.method, headers: req.headers, body: ['GET', 'HEAD'].includes(req.method) ? undefined : body }), { ip: req.socket.remoteAddress });
    res.writeHead(r.status, Object.fromEntries(r.headers));
    res.end(Buffer.from(await r.arrayBuffer()));
  }).listen(port, '127.0.0.1', () => console.log(`the function, from ${copy}, at http://127.0.0.1:${port}/mcp`));
} else {
  let failures = 0;
  const check = (ok, what) => {
    console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${what}`);
    if (!ok) failures++;
  };
  let id = 0;
  async function post(body, headers = {}, ip = '203.0.113.7') {
    const all = Object.fromEntries(Object.entries({ 'content-type': 'application/json', accept: 'application/json, text/event-stream', 'mcp-protocol-version': '2025-11-25', ...headers }).filter(([, v]) => v !== undefined));
    const r = await handler(new Request(URL_, { method: 'POST', headers: all, body: typeof body === 'string' ? body : JSON.stringify(body) }), { ip });
    const text = await r.text();
    return { status: r.status, headers: r.headers, json: text ? JSON.parse(text) : null };
  }
  const rpc = (method, params = {}, headers) => post({ jsonrpc: '2.0', id: ++id, method, params }, headers);
  const tool = async (name, args) => {
    const t0 = performance.now();
    const r = await rpc('tools/call', { name, arguments: args });
    return { ...r, ms: performance.now() - t0, result: r.json.result };
  };
  const abridge = (s, n = 6) => s.split('\n').slice(0, n).map((l) => `      ${l.slice(0, 140)}`).join('\n') + (s.split('\n').length > n ? '\n      ...' : '');

  console.log(`function copied to ${copy}`);
  console.log('> initialize');
  const init = await post({ jsonrpc: '2.0', id: ++id, method: 'initialize', params: { protocolVersion: '2025-11-25', capabilities: {}, clientInfo: { name: 'fn-test', version: '0' } } }, { 'mcp-protocol-version': undefined });
  console.log(`    ${init.status} ${JSON.stringify(init.json.result.serverInfo)} protocol ${init.json.result.protocolVersion}`);
  check(init.status === 200 && init.json.result.protocolVersion === '2025-11-25', 'initialize answers with the version asked for');
  check(!init.headers.get('mcp-session-id'), 'no session id (stateless)');
  check(init.headers.get('content-type').startsWith('application/json'), 'a JSON response, not an event stream');
  const note = await post({ jsonrpc: '2.0', method: 'notifications/initialized' });
  check(note.status === 202, 'notifications/initialized is 202');

  console.log('> tools/list');
  const list = await rpc('tools/list');
  const names = list.json.result.tools.map((t) => t.name);
  console.log(`    ${names.join(', ')}`);
  check(names.join() === 'search,get_post,viv_reference', 'the three tools that read are listed (READ_ONLY_TOOLS in netlify.mjs)');

  // the function lists no tool that runs a program: a call to one is an unknown tool
  for (const name of ['run_viv', 'check_post']) {
    console.log(`> tools/call ${name}`);
    const u = await rpc('tools/call', { name, arguments: { source: 'show 1.0\n' } });
    check(u.json.error && u.json.error.code === -32602, `${name}: not served by the function (-32602)`);
  }

  console.log('> tools/call search "runge"');
  let r = await tool('search', { query: 'runge', limit: 3 });
  console.log(`    ${r.ms.toFixed(0)} ms\n${abridge(r.result.content[0].text, 6)}`);
  check(r.result.structuredContent.hits.length > 0, 'search: finds something for runge');
  r = await tool('search', { query: 'eigenvalues', kind: 'module', limit: 5 });
  check(r.result.structuredContent.hits.every((h) => h.kind === 'module') && r.result.structuredContent.hits.length > 0, 'search: kind=module returns modules only');
  r = await tool('search', { query: 'solution', kind: 'post', limit: 25 });
  check(!r.result.structuredContent.hits.some((h) => h.postKind === 'solution'), 'search: never a solution post');

  console.log('> tools/call get_post "runge-phenomenon"');
  r = await tool('get_post', { slug: 'runge-phenomenon' });
  console.log(`${abridge(r.result.content[0].text, 4)}`);
  check(!r.result.isError && r.result.structuredContent.source, 'get_post: a post with its source');
  const solution = await tool('get_post', { slug: 'make-euler-stable-on-a-stiff-equation' });
  check(!solution.result.isError && solution.result.structuredContent.solutions !== undefined, 'get_post: a problem counts its solutions');
  r = await tool('get_post', { slug: 'no-such-post' });
  check(r.result.isError, 'get_post: an unknown slug is a tool error');

  console.log('> tools/call viv_reference');
  r = await tool('viv_reference', {});
  console.log(`    ${r.result.content[0].text.length} characters, starting\n${abridge(r.result.content[0].text, 2)}`);
  check(r.result.content[0].text.includes('Python habits'), 'viv_reference: the core sections');

  console.log('> resources/list, resources/read viv://prompt-pack and viv://llms-full');
  const res = await rpc('resources/list');
  check(res.json.result.resources.length === 2, 'two resources');
  for (const uri of ['viv://prompt-pack', 'viv://llms-full']) {
    const rr = await rpc('resources/read', { uri });
    console.log(`    ${uri}: ${rr.json.result.contents[0].text.length} characters`);
    check(rr.json.result.contents[0].text.length > 10000, `${uri} reads`);
  }

  console.log('> 2026-07-28 era');
  const META = { 'io.modelcontextprotocol/protocolVersion': '2026-07-28' };
  const disc = await post({ jsonrpc: '2.0', id: ++id, method: 'server/discover', params: { _meta: META } }, { 'mcp-protocol-version': '2026-07-28', 'mcp-method': 'server/discover' });
  check(disc.status === 200 && disc.json.result.resultType === 'complete', 'server/discover answers, with resultType');
  const bad = await post({ jsonrpc: '2.0', id: ++id, method: 'tools/list', params: { _meta: META } }, { 'mcp-protocol-version': '2026-07-28', 'mcp-method': 'tools/call' });
  check(bad.status === 400 && bad.json.error.code === -32020, 'a header that disagrees with the body is 400 / -32020');

  console.log('> HTTP edges');
  const get = await handler(new Request(URL_, { method: 'GET' }), {});
  check(get.status === 405 && get.headers.get('allow') === 'POST', 'GET is 405, Allow: POST');
  const foreign = await rpc('tools/list', {}, { origin: 'https://evil.example' });
  check(foreign.status === 403, 'a foreign Origin is 403');
  const ours = await rpc('tools/list', {}, { origin: 'https://viv-lang.org' });
  check(ours.status === 200 && ours.headers.get('access-control-allow-origin') === 'https://viv-lang.org', 'the site origin is served, with CORS');
  const local = await rpc('tools/list', {}, { origin: 'http://localhost:6274' });
  check(local.status === 200, 'a localhost page (the MCP inspector) is served');
  const pre = await handler(new Request(URL_, { method: 'OPTIONS', headers: { origin: 'http://localhost:6274', 'access-control-request-method': 'POST' } }), {});
  check(pre.status === 204 && /mcp-protocol-version/i.test(pre.headers.get('access-control-allow-headers')), 'a preflight from an allowed origin is 204');
  const text = await post('{not json');
  check(text.status === 400 && text.json.error.code === -32700, 'bad JSON is 400 / -32700');

  // the JSON index against the server's FTS5 index, on the documents both have (no posts: the server's come from its database)
  console.log('> search: the JSON index against FTS5, top 5 of the repository documents');
  const { createSearch, repoDocs } = await import('./search.mjs');
  const { createJsonSearch } = await import('./jsearch.mjs');
  const fts = createSearch(null);
  const js = createJsonSearch(repoDocs());
  let same = 0;
  let total = 0;
  for (const q of ['runge', 'symmetric eigenvalues', 'fft', 'landau damping', 'newton', 'hartree fock', 'random walk', 'sparse matrix', 'zeros', 'stencil']) {
    const a = fts.search(q, { limit: 5 }).map((h) => `${h.kind}:${h.key}`);
    const b = js.search(q, { limit: 5 }).map((h) => `${h.kind}:${h.key}`);
    const shared = a.filter((k) => b.includes(k)).length;
    same += shared;
    total += Math.max(a.length, b.length);
    console.log(`    "${q}": ${shared} of ${Math.max(a.length, b.length)} shared${a[0] === b[0] ? ', same first hit' : `; FTS5 first ${a[0]}, JSON first ${b[0]}`}`);
  }
  check(same / Math.max(1, total) >= 0.6, `the two indexes agree on ${same} of ${total} top-5 hits`);

  console.log(failures ? `${failures} check(s) failed` : 'every check passed');
  rmSync(dirname(copy), { recursive: true, force: true });
  process.exit(failures ? 1 : 0);
}
