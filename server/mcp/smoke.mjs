// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Connect to a running MCP endpoint with the official TypeScript SDK client,
// list its tools and run a Viv program: the end-to-end check that a real MCP
// client can use the server.
//
//     node mcp/smoke.mjs                              http://localhost:8787/mcp
//     node mcp/smoke.mjs https://viv-lang.org/mcp
//
// Needs the SDK (a devDependency: `npm install` in server/).
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';

const url = new URL(process.argv[2] || 'http://localhost:8787/mcp');
const PROGRAM = `# The trapezoid rule on sin over [0, pi], whose integral is 2.
def trap(n) {
    h = 3.141592653589793 / n
    s = 0.0
    for k to n - 1 {
        s = s + sin((k + 1) * h)
    }
    return s * h
}
ns = [8.0, 16.0, 32.0, 64.0]
errs = zeros(4)
for k to 4 {
    errs[k] = abs(trap(ns[k]) - 2.0)
}
figure("trapezoid error", "panels", "|error|")
plot(ns, errs, "error")
ratio = errs[2] / errs[3]
show ratio
test "doubling the panels divides the error by four" {
    assert_near(ratio, 4.0, 0.01)
}
`;

const client = new Client({ name: 'viv-mcp-smoke', version: '0.1.0' });
const t0 = performance.now();
await client.connect(new StreamableHTTPClientTransport(url));
const server = client.getServerVersion();
console.log(`connected to ${server.name} ${server.version} at ${url.href}, protocol ${client.transport.protocolVersion || '(negotiated)'}`);
const { tools } = await client.listTools();
console.log(`tools: ${tools.map((t) => t.name).join(', ')}`);
const run = await client.callTool({ name: 'run_viv', arguments: { source: PROGRAM } });
console.log(`run_viv (isError ${run.isError}):\n${run.content[0].text}`);
const found = await client.callTool({ name: 'search', arguments: { query: 'runge', limit: 3 } });
console.log(`search "runge":\n${found.content[0].text}`);
await client.close();
console.log(`${((performance.now() - t0) / 1000).toFixed(2)} s`);
process.exit(run.isError || !run.structuredContent.ok ? 1 : 0);
