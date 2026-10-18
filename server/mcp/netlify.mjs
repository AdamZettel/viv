// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The MCP endpoint as a Netlify Function: the Fetch API's Request in, a
// Response out, around the same protocol the feed's server speaks
// (protocol.mjs). build-function.mjs bundles this file with the tools for a
// static site (fntools.mjs) into www/netlify/functions/mcp/, beside the data
// the tools read, and netlify.toml routes /mcp to it.
//
// Who may call from a browser: the site's own origin (whatever host serves the
// function, deploy previews included), https://viv-lang.org, the origins in
// MCP_ALLOWED_ORIGINS, and pages on localhost (the MCP inspector). Clients
// that are not browsers send no Origin and are served.
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createProtocol, originCheck } from './protocol.mjs';
import { createFnTools } from './fntools.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const RUN_TIMEOUT_MS = 8000; // a synchronous Netlify Function is stopped at about 10 s
const MEMORY_MB = 384; // of the 1024 MB an invocation gets: the module, the worker and the data need the rest
// The hosted endpoint serves only the tools that read: no program of a
// caller's runs on Netlify (the owner's choice at launch, 2026-10-15).
// run_viv and check_post stay in the node server.
const READ_ONLY_TOOLS = ['search', 'get_post', 'viv_reference'];
const READ_ONLY_INSTRUCTIONS = 'Tools for the Viv language (viv-lang.org): search posts and modules, read a post, and read the language reference. Viv is not Python: call viv_reference before writing a program. Every tool only reads.';

/** A handler (request, context) => Response, reading its data from `dir`. */
export function createFunctionHandler({ dir = HERE, env = process.env, runTimeoutMs = RUN_TIMEOUT_MS, memoryMb = MEMORY_MB } = {}) {
  const all = createFnTools({ dir, runTimeoutMs, memoryMb });
  const tools = all.tools.filter((t) => READ_ONLY_TOOLS.includes(t.name));
  const handlers = Object.fromEntries(READ_ONLY_TOOLS.map((n) => [n, all.handlers[n]]));
  const allowed = ['https://viv-lang.org', 'https://www.viv-lang.org', env.URL, env.DEPLOY_PRIME_URL, ...String(env.MCP_ALLOWED_ORIGINS || '').split(',').map((s) => s.trim())].filter(Boolean);
  const handle = createProtocol({ tools, handlers, originOk: originCheck(allowed, { allowLocal: true }), sameOriginOk: true, requestsPerMinute: 120, instructions: READ_ONLY_INSTRUCTIONS });

  return async function mcp(request, context = {}) {
    const url = new URL(request.url);
    const headers = {};
    for (const [k, v] of request.headers) headers[k.toLowerCase()] = v;
    const body = request.method === 'POST' ? await request.text() : '';
    const ip = context.ip || headers['x-nf-client-connection-ip'] || '?';
    const r = await handle({ method: request.method, headers, body, ip, selfOrigin: url.origin });
    return new Response(r.body === '' ? null : r.body, { status: r.status, headers: r.headers });
  };
}

let handler = null;
export default async function mcp(request, context) {
  handler ||= createFunctionHandler();
  return handler(request, context);
}
