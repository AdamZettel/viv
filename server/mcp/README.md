# server/mcp: the read-only MCP endpoint

viv-lang.org answers [Model Context Protocol](https://modelcontextprotocol.io)
clients at `/mcp`: Claude, Claude Code, ChatGPT, Cursor, VS Code and anything
else that speaks MCP over HTTP. It needs no sign-in, because every tool only
reads. The site's Netlify Function serves three of the five tools below:
`search`, `get_post` and `viv_reference` (`READ_ONLY_TOOLS` in `netlify.mjs`).
`run_viv` and `check_post` run a program, and only the feed's server serves
them; at viv-lang.org a program runs in the workbench.

| Tool | What it does |
|---|---|
| `run_viv` | Runs a program in the interpreter (the site's `viv.wasm`) with a 15 s limit and a memory limit. Returns the `show`/`print` lines, each test as PASS or FAIL, and one line per figure in place of its data points. |
| `check_post` | Runs a post and applies the feed's rules (`www/src/lib/postCheck.js`), and the `from:`/`edge:` claim if there is one. Publishes nothing. |
| `search` | Full-text search (SQLite FTS5) over published posts, the example programs, the `lib/` modules, the builtins and the documented examples. |
| `get_post` | A published post: source, header, tests, lane and lineage. |
| `viv_reference` | The prompt pack (`www/scripts/llms-build.mjs`), whole or by section. |

The pack is also served as two resources, `viv://prompt-pack` and
`viv://llms-full`, and as static files at `/viv-prompt-pack.md`,
`/llms-full.txt` and `/llms.txt`.

Solution posts are not searchable and `get_post` does not return their source:
on the site a problem's answers stay hidden until the reader has solved it, and
a model's context is not where to hand them over. Drafts and attempts are never
read.

## Protocol

Written from the specification, on `node:http`, so the server still has no
runtime dependencies. It speaks both eras of revision 2026-07-28's versioning
page:

- **2025-03-26, 2025-06-18 and 2025-11-25**, which open with `initialize`. The
  official TypeScript SDK (1.30.1) speaks these; its latest is 2025-11-25. The
  server is stateless: it issues no `Mcp-Session-Id`, answers each POST with
  one JSON object, and answers GET and DELETE with 405.
- **2026-07-28**, with no handshake: the version in each request's `_meta`, the
  `MCP-Protocol-Version`, `Mcp-Method` and `Mcp-Name` headers checked against
  the body (400 and `-32020` when they disagree), `server/discover`, and
  `resultType` on every result.

A request whose `Origin` header is not the site's (`VIV_ORIGIN`, plus
`MCP_ALLOWED_ORIGINS`) is refused with 403; with neither set, only
`localhost` pages may call. Clients that are not browsers send no `Origin`.

## Limits

| Setting | Default | Environment |
|---|---|---|
| Requests per address per minute | 120 | `MCP_REQUESTS_PER_MINUTE` |
| Program runs (`run_viv`, `check_post`) per address per hour | 30 | `MCP_RUNS_PER_HOUR` |
| Memory one run may use | 512 MB | `MCP_MEMORY_MB` |
| Runs at once, whole server | 4 | `MCP_MAX_CONCURRENT_RUNS` |
| Time for one `run_viv` | 15 s | |

The memory limit is written into a copy of `viv.wasm`'s memory section
(`wasmcap.mjs`): a worker's `resourceLimits` cap the JavaScript heap, not wasm
memory. Measured: five 15-million-element arrays stop at 256 MB with an error,
while the uncapped module let ten of them take the process to 1.0 GB. Behind a
proxy, set `VIV_TRUST_PROXY=1` so the limits count the client's address (the
last `X-Forwarded-For` entry) and not the proxy's.

## Where it runs

The site's deploy serves the endpoint from a Netlify Function (next section).
The feed's long-running server serves the same protocol and tools and reads
posts from its database. That server is not part of this repository, so the
server column of the table below is for reference.

## The Netlify Function

The static site's deploy serves the same endpoint from a Netlify Function, so
`https://viv-lang.org/mcp` works without the server. The protocol is one pure
function, `protocol.mjs` (a request in, a response out); the server wraps it
in `node:http` and `netlify.mjs` in the Fetch API's `Request` and `Response`
for the function. The tool texts are shared too (`tooldefs.mjs`,
`toolcore.mjs`); only where the facts come from differs: the server reads its
database, `fntools.mjs` reads what the site build wrote.

After the site build, `build-function.mjs` writes `www/netlify/functions/mcp/`
(gitignored, about 17 MB): the bundled code, the run worker, `viv.wasm`, the
prompt pack, the `.viv` modules an import can name, the published posts from
`feed.json` with their lineage, and the search documents. `netlify.toml` runs
it after `npm run build`, lists that directory in `included_files`, and
rewrites `/mcp` to `/.netlify/functions/mcp`. The build refuses a bundle that
reaches `node:sqlite` or the database code.

The function serves `search`, `get_post` and `viv_reference`; a call to
`run_viv` or `check_post` gets `-32602` (unknown tool). The rows about runs below
describe `fntools.mjs`, which the function bundles but does not list.
How the function differs from the server:

| | Server | Function |
|---|---|---|
| `run_viv` time limit | 15 s | 8 s, under the 10 s Netlify gives a synchronous function; a run stopped there returns `isError` with `error: "timeout"` and says to use the workbench |
| memory per run | 512 MB | 384 MB, of the 1024 MB an invocation gets |
| output per run | no limit | 8 MB; a longer output is replaced by an `error` line that gives its size. The worker measures it in wasm memory, before it is decoded, so the main thread never holds it: a run's output is copied and parsed at several times its size, and 93 MB of plot points took the process to 954 MB without the limit (568 MB with it) |
| `check_post` | the database's posts, channels and parents | the posts, channels and parents the build published; a `from:` claim shares the 8 s with the post's own run |
| `search` | SQLite FTS5 | `jsearch.mjs` over `data/corpus.json`: the same words, prefixes, AND then OR, and column weights, with BM25 written out in JavaScript. On ten queries over the repository's documents the two agree on 44 of 44 top-5 hits. Snippets are close to FTS5's but not identical, and a word with an underscore is matched part by part, not as a phrase |
| `get_post` | the database | `data/posts.json`: committed posts and claims as of the deploy; the source is the file in `posts/`, header included |
| requests per address | 120 a minute | 20 a minute at Netlify's edge (`MCP_RATE_LIMIT` in `build-function.mjs`, written into the entry file's `config`), then 429; behind it the protocol's 120 a minute |
| run budget per address | 30 an hour | 30 per ten minutes in each function instance; Netlify runs as many instances as it likes, so this bounds a burst only |

A page may call from `https://viv-lang.org`, from the origin serving the
function (deploy previews included), from `MCP_ALLOWED_ORIGINS`, and from
localhost (the MCP inspector); other origins get 403, and an allowed one gets
CORS headers and an answer to its preflight. Clients that are not browsers send
no Origin.

```sh
cd www && npm run build && node ../server/mcp/build-function.mjs
node ../server/mcp/fn-test.mjs              # every tool and the protocol, against a copy of the directory
node ../server/mcp/fn-test.mjs --serve 8788 # the function at http://127.0.0.1:8788/mcp, for curl or smoke.mjs
```

## Adding it to a client

Everything below comes from the vendors' pages as read on 2026-09-24. None of
these clients has been tried against this endpoint yet; it was checked with the
TypeScript SDK's client (`smoke.mjs`). Clients that run on your own machine
(Claude Code, Cursor, VS Code) can also use the function served locally by
`fn-test.mjs --serve 8788`, at `http://127.0.0.1:8788/mcp`. The web apps (claude.ai, ChatGPT) connect
from their own servers, so they need the public URL.

**Claude (claude.ai, Desktop, Cowork).** Customize > Connectors > Add custom
connector, and give the URL `https://viv-lang.org/mcp`. On Team and Enterprise
plans an owner adds it in Organization settings. Free accounts get one custom
connector. Leave the OAuth client ID and secret (under Advanced settings)
empty; whether the form accepts a server with no sign-in was not checked. Source:
support.claude.com/en/articles/11175166.

**Claude Code.**

```sh
claude mcp add --transport http viv https://viv-lang.org/mcp
claude mcp add --transport http viv-local http://127.0.0.1:8788/mcp    # fn-test.mjs --serve 8788
```

Then `/mcp` in a session lists it. A project can commit the same entry in
`.mcp.json`. Source: code.claude.com/docs/en/mcp.

**Claude Messages API.** Send the beta header `mcp-client-2025-11-20`, list the
server in `mcp_servers` with its `url` (no `authorization_token` is needed), and
add an `mcp_toolset` tool for it. This connector is not eligible for zero data
retention and is not offered on Bedrock or Vertex. Source:
platform.claude.com/docs/en/agents-and-tools/mcp-connector.

**ChatGPT developer mode.** Settings > Security and login > Developer mode, on
the web, for Pro, Plus, Business, Enterprise and Education accounts. Add the URL
with no authentication. Source: developers.openai.com/api/docs/guides/developer-mode.

**OpenAI Responses API.** A tool
`{"type": "mcp", "server_label": "viv", "server_url": "https://viv-lang.org/mcp", "allowed_tools": ["search", "viv_reference"]}`.
Approval of each call is on by default (`require_approval`). Source:
developers.openai.com/api/docs/guides/tools-connectors-mcp.

**xAI API (Grok).** A tool
`{"type": "mcp", "server_url": "https://viv-lang.org/mcp", "server_label": "viv"}`,
optionally with `allowed_tools`. Source: docs.x.ai/docs/guides/tools/remote-mcp-tools.

**Cursor.** In `mcp.json`:

```json
{ "mcpServers": { "viv": { "url": "https://viv-lang.org/mcp" } } }
```

Source: cursor.com/docs/context/mcp.

**VS Code (Copilot agent mode).** In `mcp.json`:

```json
{ "servers": { "viv": { "type": "http", "url": "https://viv-lang.org/mcp" } } }
```

Source: code.visualstudio.com/docs/copilot/customization/mcp-servers.

Which protocol revision each of these clients speaks was not checked. A client
on 2025-03-26 through 2025-11-25 or on 2026-07-28 is served; an older one, on the
deprecated HTTP+SSE transport of 2024-11-05, is not.
