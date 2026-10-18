// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The MCP protocol as one pure function: a request in ({ method, headers,
// body, ip, selfOrigin }), a response out ({ status, headers, body }). The
// feed's server wraps it in node:http; the Netlify Function wraps it
// in the Fetch API's Request and Response (netlify.mjs). Neither wrapper knows
// anything about MCP.
//
// Which protocol. It serves both eras the 2026-07-28 specification describes
// ("dual-era", basic/versioning):
//   - LEGACY, revisions 2025-03-26, 2025-06-18 and 2025-11-25: the client opens
//     with `initialize` and sends MCP-Protocol-Version on later requests. This
//     is what the official TypeScript SDK (1.30.1, 2026-09-23) speaks: its
//     latest version is 2025-11-25. This server is stateless in that era: it
//     issues no Mcp-Session-Id (sessions are a MAY), answers every request with
//     one JSON object (never an SSE stream), and answers GET with 405, which
//     the 2025-11-25 transport allows ("does not offer an SSE stream").
//   - MODERN, revision 2026-07-28: no handshake; every request carries its
//     version in params._meta and in the MCP-Protocol-Version header, plus
//     Mcp-Method (and Mcp-Name for tools/call and resources/read). A header that
//     disagrees with the body is refused with 400 and -32020 HeaderMismatch; an
//     unknown version with 400 and -32022, listing the versions served;
//     server/discover says what the server is; results carry resultType.
// Sources: modelcontextprotocol.io/specification/2025-11-25/basic/transports and
// /specification/2026-07-28/{basic/transports/streamable-http, basic/versioning,
// server/discover, server/tools}, read 2026-09-24.
//
// Origin and CORS. A request whose Origin header is not one of ours is refused
// with 403 (the spec's MUST, against DNS rebinding). MCP clients that are not
// browsers send no Origin, and the hosted ones (claude.ai, ChatGPT) call from
// their own servers, so they are not affected. A page on an allowed origin (the
// site itself, or a localhost tool such as the MCP inspector) gets CORS headers,
// and its preflight OPTIONS is answered.
// the pack's size, measured when the site builds it (www/scripts/llms-build.mjs)
import PACK from '../../www/src/data/promptPack.json' with { type: 'json' };
import { load as loadReference } from './reference.mjs';

export const LEGACY_VERSIONS = ['2025-11-25', '2025-06-18', '2025-03-26'];
export const MODERN_VERSIONS = ['2026-07-28'];
export const SUPPORTED_VERSIONS = [...MODERN_VERSIONS, ...LEGACY_VERSIONS];
const DEFAULT_LEGACY = '2025-03-26'; // what a request with no version header is taken to speak
export const BODY_LIMIT = 512 * 1024;
const META_VERSION = 'io.modelcontextprotocol/protocolVersion';

export const SERVER_INFO = { name: 'viv-lang', title: 'Viv', version: '0.0.1' };
const INSTRUCTIONS = 'Tools for the Viv language (viv-lang.org): run a Viv program, check a post against the posting rules, search posts and modules, read a post, and read the language reference. Viv is not Python: call viv_reference before writing a program, then run_viv to check it. Every tool only reads; nothing is published.';
const CAPABILITIES = { tools: { listChanged: false }, resources: { listChanged: false } };

const RESOURCES = [
  { uri: 'viv://prompt-pack', name: 'viv-prompt-pack', title: 'Viv prompt pack', description: `Grammar, Python habits that fail, builtins and every documented example with its output (about ${PACK.kTokens}k tokens).`, mimeType: 'text/markdown', file: 'pack' },
  { uri: 'viv://llms-full', name: 'llms-full', title: 'Viv full reference', description: 'The pack plus the CLI, the backend matrix and the engines that run each builtin.', mimeType: 'text/plain', file: 'full' },
];

// JSON-RPC and MCP error codes
const PARSE_ERROR = -32700;
const INVALID_REQUEST = -32600;
const METHOD_NOT_FOUND = -32601;
const INVALID_PARAMS = -32602;
const RESOURCE_NOT_FOUND = -32002;
const HEADER_MISMATCH = -32020;
const UNSUPPORTED_VERSION = -32022;

export class RpcError extends Error {
  constructor(code, message, { status = 200, data } = {}) { super(message); this.code = code; this.status = status; this.data = data; }
}

/** A sliding-window counter per key. */
export class Window {
  constructor(max, perMs) { this.max = max; this.perMs = perMs; this.hits = new Map(); }
  take(key) {
    const t = Date.now();
    const list = (this.hits.get(key) || []).filter((x) => t - x < this.perMs);
    if (list.length >= this.max) { this.hits.set(key, list); return false; }
    list.push(t);
    this.hits.set(key, list);
    if (this.hits.size > 20000) this.hits.clear(); // a flood of addresses resets the table rather than growing it
    return true;
  }
}

const LOCAL = /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/;

/** Origins allowed to reach the endpoint from a browser. With none configured
 *  (a developer's machine), local pages only; `allowLocal` keeps local pages
 *  allowed beside a configured list. */
export function originCheck(allowed, { allowLocal = false } = {}) {
  const set = new Set(allowed.filter(Boolean));
  return (origin) => (set.size ? set.has(origin) || (allowLocal && LOCAL.test(origin)) : LOCAL.test(origin));
}

const CORS_ALLOW_HEADERS = 'Content-Type, Accept, MCP-Protocol-Version, Mcp-Method, Mcp-Name, Mcp-Session-Id, Last-Event-ID';

/**
 * The protocol, given the tools. Options:
 *   tools              the tool list (tooldefs.mjs)
 *   handlers           { name: async (args, ctx) => { text, data, isError } }
 *   originOk           (origin) => bool: who may call from a browser (originCheck)
 *   requestsPerMinute  per address, every request (default 120)
 *   sameOriginOk       a page served by the same host may call (the function knows its own origin)
 * Returns handle({ method, headers, body, ip, selfOrigin }) => { status, headers, body },
 * with `headers` lowercased and `body` the request's text, already read.
 */
export function createProtocol({ tools, handlers, originOk = originCheck([]), requestsPerMinute = 120, sameOriginOk = false, instructions = INSTRUCTIONS }) {
  const requests = new Window(requestsPerMinute, 60_000);

  async function call(method, params, ctx) {
    switch (method) {
      case 'initialize': {
        if (ctx.modern) throw new RpcError(METHOD_NOT_FOUND, `initialize is not part of ${ctx.version}; send requests directly`, { status: 404 });
        const asked = params && params.protocolVersion;
        // the client's version when we speak it, else our latest of that era
        return { protocolVersion: LEGACY_VERSIONS.includes(asked) ? asked : LEGACY_VERSIONS[0], capabilities: CAPABILITIES, serverInfo: SERVER_INFO, instructions: instructions };
      }
      case 'server/discover':
        return { supportedVersions: SUPPORTED_VERSIONS, capabilities: CAPABILITIES, _meta: { 'io.modelcontextprotocol/serverInfo': SERVER_INFO }, instructions: instructions, ttlMs: 3_600_000, cacheScope: 'public' };
      case 'ping':
        return {};
      case 'tools/list':
        return { tools };
      case 'tools/call': {
        const name = params && params.name;
        const handler = handlers[name];
        if (!handler || !tools.some((t) => t.name === name)) throw new RpcError(INVALID_PARAMS, `Unknown tool: ${name}`);
        const args = (params && params.arguments) || {};
        if (typeof args !== 'object' || Array.isArray(args)) throw new RpcError(INVALID_PARAMS, 'arguments must be an object');
        const r = await handler(args, ctx);
        const result = { content: [{ type: 'text', text: r.text }], isError: !!r.isError };
        if (r.data) result.structuredContent = r.data;
        return result;
      }
      case 'resources/list':
        return { resources: RESOURCES.map(({ file, ...r }) => r) };
      case 'resources/read': {
        const r = RESOURCES.find((x) => x.uri === (params && params.uri));
        if (!r) throw new RpcError(RESOURCE_NOT_FOUND, `Resource not found: ${params && params.uri}`, { data: { uri: params && params.uri } });
        const ref = loadReference();
        return { contents: [{ uri: r.uri, mimeType: r.mimeType, text: r.file === 'pack' ? ref.pack : ref.full }] };
      }
      default:
        throw new RpcError(METHOD_NOT_FOUND, `Method not found: ${method}`, { status: ctx.modern ? 404 : 200 });
    }
  }

  /** Which era a message speaks, checked against the headers the way its revision requires. */
  function negotiate(header, msg) {
    const version0 = header('mcp-protocol-version');
    const meta = msg.params && msg.params._meta && msg.params._meta[META_VERSION];
    if (meta !== undefined || MODERN_VERSIONS.includes(version0)) {
      // modern: the version is in the body, and the headers mirror the body
      const version = meta;
      if (version === undefined) throw new RpcError(HEADER_MISMATCH, `Header mismatch: MCP-Protocol-Version is ${version0} but params._meta has no ${META_VERSION}`, { status: 400 });
      if (!SUPPORTED_VERSIONS.includes(version) || LEGACY_VERSIONS.includes(version)) {
        throw new RpcError(UNSUPPORTED_VERSION, 'Unsupported protocol version', { status: 400, data: { supported: SUPPORTED_VERSIONS, requested: version } });
      }
      if (version0 !== version) throw new RpcError(HEADER_MISMATCH, `Header mismatch: MCP-Protocol-Version header ${version0 === undefined ? 'is missing' : `'${version0}'`} but the body says '${version}'`, { status: 400 });
      const method = header('mcp-method');
      if (method !== msg.method) throw new RpcError(HEADER_MISMATCH, `Header mismatch: Mcp-Method header ${method === undefined ? 'is missing' : `'${method}'`} but the body's method is '${msg.method}'`, { status: 400 });
      if (['tools/call', 'resources/read', 'prompts/get'].includes(msg.method) && msg.id !== undefined) {
        const bodyName = msg.method === 'resources/read' ? msg.params.uri : msg.params.name;
        const name = header('mcp-name');
        if (name !== bodyName) throw new RpcError(HEADER_MISMATCH, `Header mismatch: Mcp-Name header ${name === undefined ? 'is missing' : `'${name}'`} does not match body value '${bodyName}'`, { status: 400 });
      }
      return { modern: true, version };
    }
    // legacy: initialize names the version; later requests carry it in the header
    if (msg.method === 'initialize') return { modern: false, version: LEGACY_VERSIONS.includes(msg.params && msg.params.protocolVersion) ? msg.params.protocolVersion : LEGACY_VERSIONS[0] };
    const version = version0 === undefined ? DEFAULT_LEGACY : version0;
    if (!LEGACY_VERSIONS.includes(version)) throw new RpcError(INVALID_REQUEST, `Unsupported MCP-Protocol-Version: ${version} (this server speaks ${SUPPORTED_VERSIONS.join(', ')})`, { status: 400, data: { supported: SUPPORTED_VERSIONS, requested: version } });
    return { modern: false, version };
  }

  const rpcError = (id, e) => ({ jsonrpc: '2.0', ...(id === undefined ? {} : { id }), error: { code: e.code, message: e.message, ...(e.data ? { data: e.data } : {}) } });

  return async function handle({ method: httpMethod, headers = {}, body = '', ip = '?', selfOrigin = null }) {
    // a header value, with the spec's `=?base64?...?=` form decoded
    const header = (name) => {
      const v = headers[name];
      if (v === undefined || v === null) return undefined;
      const s = Array.isArray(v) ? v.join(',') : String(v);
      const m = s.match(/^=\?base64\?(.*)\?=$/);
      return m ? Buffer.from(m[1], 'base64').toString('utf8') : s;
    };
    const origin = header('origin');
    const allowed = origin !== undefined && (originOk(origin) || (sameOriginOk && selfOrigin && origin === selfOrigin));
    const cors = allowed ? { 'Access-Control-Allow-Origin': origin, Vary: 'Origin' } : {};
    const send = (status, obj, extra = {}) => ({
      status,
      headers: { ...(obj === null ? {} : { 'Content-Type': 'application/json; charset=utf-8' }), 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', ...cors, ...extra },
      body: obj === null ? '' : JSON.stringify(obj),
    });

    if (origin !== undefined && !allowed) return send(403, rpcError(undefined, new RpcError(INVALID_REQUEST, `requests from the origin ${origin} are not accepted`)));
    // a browser's preflight, from an origin that may call
    if (httpMethod === 'OPTIONS' && allowed) return send(204, null, { 'Access-Control-Allow-Methods': 'POST', 'Access-Control-Allow-Headers': CORS_ALLOW_HEADERS, 'Access-Control-Max-Age': '86400' });
    // no standalone stream (legacy GET) and no sessions to end (DELETE)
    if (httpMethod !== 'POST') return send(405, rpcError(undefined, new RpcError(INVALID_REQUEST, `${httpMethod} is not supported at this endpoint; send JSON-RPC messages with POST`)), { Allow: 'POST' });
    if (!requests.take(ip)) return send(429, rpcError(undefined, new RpcError(INVALID_REQUEST, `too many requests from this address (${requestsPerMinute} a minute); wait and try again`)), { 'Retry-After': '60' });
    if (!/^application\/json\b/i.test(header('content-type') || '')) return send(415, rpcError(undefined, new RpcError(INVALID_REQUEST, 'the body must be JSON (Content-Type: application/json)')));
    if (Buffer.byteLength(body) > BODY_LIMIT) return send(413, rpcError(undefined, new RpcError(INVALID_REQUEST, `the request body is over ${BODY_LIMIT / 1024} KB`)));

    let msg;
    try {
      try { msg = JSON.parse(body); } catch { throw new RpcError(PARSE_ERROR, 'Parse error: the body is not JSON', { status: 400 }); }
      if (Array.isArray(msg)) throw new RpcError(INVALID_REQUEST, 'batches are not supported: send one JSON-RPC message per request', { status: 400 });
      if (!msg || typeof msg !== 'object' || msg.jsonrpc !== '2.0') throw new RpcError(INVALID_REQUEST, 'Invalid Request: not a JSON-RPC 2.0 message', { status: 400 });
    } catch (e) {
      if (e instanceof RpcError) return send(e.status === 200 ? 400 : e.status, rpcError(undefined, e));
      throw e;
    }

    // a notification, or a response to something we never asked: accepted, nothing to say
    if (msg.method === undefined || msg.id === undefined) {
      if (msg.method === undefined && msg.result === undefined && msg.error === undefined) return send(400, rpcError(undefined, new RpcError(INVALID_REQUEST, 'Invalid Request: no method')));
      return send(202, null);
    }
    if (typeof msg.method !== 'string' || !(typeof msg.id === 'string' || typeof msg.id === 'number')) return send(400, rpcError(msg.id ?? undefined, new RpcError(INVALID_REQUEST, 'Invalid Request: method must be a string and id a string or number')));

    try {
      const ctx = { ...negotiate(header, msg), ip };
      const result = await call(msg.method, msg.params || {}, ctx);
      return send(200, { jsonrpc: '2.0', id: msg.id, result: ctx.modern ? { resultType: 'complete', ...result } : result });
    } catch (e) {
      if (e instanceof RpcError) return send(e.status, rpcError(msg.id, e));
      console.error('mcp:', e);
      return send(500, rpcError(msg.id, new RpcError(-32603, 'Internal error: the server failed on this request; try again later')));
    }
  };
}
