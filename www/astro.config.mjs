// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import sitemap from '@astrojs/sitemap';
import { readFileSync, existsSync, realpathSync, copyFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { searchForWorkspaceRoot } from 'vite';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

// Put back the generated, git-ignored files the pages load (scripts/dev-assets.mjs)
// whenever a dev server starts, however it was started: `npm run dev`,
// `npx astro dev` or an agent's server in a worktree.
function devAssets() {
  return {
    name: 'viv-dev-assets',
    hooks: {
      'astro:config:setup': ({ command }) => {
        if (command !== 'dev' || process.env.VIV_DEV_ASSETS === '0') return;
        const script = fileURLToPath(new URL('./scripts/dev-assets.mjs', import.meta.url));
        spawnSync(process.execPath, [script], { stdio: 'inherit' });
      },
    },
  };
}

// The compiler at a content-hashed address: the build copies public/viv/viv.wasm
// to /_astro/viv.<hash>.wasm, which netlify.toml caches forever like the rest
// of /_astro/, and every page and worker loads it from there
// (src/lib/vivWasm.js reads the address as __VIV_WASM_URL__). A new compiler is
// a new address, so no reader keeps a stale one. The dev server, and a build
// without a viv.wasm, use /viv/viv.wasm.
function vivWasmAddress() {
  const src = fileURLToPath(new URL('./public/viv/viv.wasm', import.meta.url));
  let url = '/viv/viv.wasm';
  return {
    name: 'viv-wasm-address',
    hooks: {
      'astro:config:setup': ({ command, updateConfig }) => {
        if (command === 'build' && existsSync(src)) {
          url = `/_astro/viv.${createHash('sha256').update(readFileSync(src)).digest('hex').slice(0, 12)}.wasm`;
        }
        // the compiler's version, for the terminal's `viv --version` (src/components/VivTerm.jsx)
        const cargo = fileURLToPath(new URL('../lang/viv/Cargo.toml', import.meta.url));
        const version = existsSync(cargo) ? (readFileSync(cargo, 'utf8').match(/^version\s*=\s*"([^"]+)"/m) || [])[1] || '' : '';
        updateConfig({ vite: { define: { __VIV_WASM_URL__: JSON.stringify(url), __VIV_VERSION__: JSON.stringify(version) } } });
      },
      'astro:build:done': ({ dir }) => {
        if (url !== '/viv/viv.wasm') copyFileSync(src, fileURLToPath(new URL('.' + url, dir)));
      },
    },
  };
}

// The public build (PUBLIC_SITE=1) has no server behind /api, so the client of
// that server stays out of its bundles: lib/api.js, components/Reviews.jsx and
// components/FeedLive.jsx resolve to their .public twins, which have the same
// exports and talk to nothing. A checkout without the originals (the public
// repository) builds from the twins too.
const PUBLIC_TWINS = ['src/lib/api.js', 'src/components/Reviews.jsx', 'src/components/FeedLive.jsx']
  .map((rel) => fileURLToPath(new URL('./' + rel, import.meta.url)));
function publicTwins() {
  const twin = (file) => file.replace(/\.(jsx?)$/, '.public.$1');
  const swap = new Set(PUBLIC_TWINS.filter((f) => process.env.PUBLIC_SITE === '1' || !existsSync(f)));
  return {
    name: 'viv-public-twins',
    enforce: 'pre',
    // an import (`../lib/api.js`), or a page's island, which Astro hands the
    // bundler as an absolute path
    resolveId(source, importer) {
      const bare = source.split('?')[0];
      let file = null;
      if (bare.startsWith('/') && !bare.startsWith('//')) file = bare;
      else if (importer && bare.startsWith('.')) file = fileURLToPath(new URL(bare, 'file://' + importer.split('?')[0]));
      return file && swap.has(file) ? twin(file) + source.slice(bare.length) : null;
    },
    // and whatever reaches the original by another road gets the twin's text
    load(id) {
      const file = id.split('?')[0];
      return swap.has(file) ? readFileSync(twin(file), 'utf8') : null;
    },
  };
}

export default defineConfig({
  site: 'https://viv-lang.org',
  integrations: [devAssets(), vivWasmAddress(), react(), // /atlas and /atlas/walk show a program only when the address names one
    sitemap({ filter: (page) => !/\/atlas\/(walk\/)?$/.test(page) })],
  devToolbar: { enabled: false },
  // The list of posts was the feed, at /feed, and the gallery was Applications.
  // netlify.toml has the same rules as 301s for production; these are for the dev server.
  redirects: { '/feed': '/posts', '/findings': '/c/claims', '/vipyr/findings': '/c/claims', '/applications': '/gallery', '/show': '/gallery', '/p/how-long-until-the-money-doubles': '/p/doubling-time-fixed-for-tiny-and-negative-rates', '/p/doubling-time-compounded-n-times-a-year': '/p/doubling-time-fixed-for-tiny-and-negative-rates' },
  vite: {
    plugins: [publicTwins()],
    // the compiler is a 2 MB wasm artifact fetched at runtime, never bundled
    assetsInclude: ['**/*.viv'],
    // Pre-bundle every browser dependency when the dev server starts. Left to
    // discover them on first use, Vite re-bundles mid-session, and pages still
    // holding the old bundle then fail with "504 Outdated Optimize Dep" -- which
    // took down the whole workbench (Monaco) and the math on the feed (KaTeX).
    optimizeDeps: { include: ['react', 'react/jsx-runtime', 'react/jsx-dev-runtime', 'react-dom', 'react-dom/client', '@monaco-editor/react', 'katex', 'marked'] },
    // One React however a module reaches it: a second copy (from a re-bundle
    // mid-session) leaves every island failing with "reading 'useState'".
    resolve: { dedupe: ['react', 'react-dom'] },
    // Vite's cache lives in the checkout, not in node_modules: git worktrees
    // symlink node_modules to this one, and a build or dev server there
    // rewrote the shared prebundle under a running dev server here, which then
    // served two copies of React and every island failed (the feed went empty).
    cacheDir: '.vite-cache',
    // A worktree's node_modules is a symlink to the main checkout's, which is
    // outside this root: without the real path on the allow list, Vite answers
    // 403 for every file a stylesheet there pulls in, KaTeX's fonts among them,
    // and the math falls back to the page font (a pmatrix gets small,
    // unstretched parentheses).
    // The proxy: the MCP endpoint (server/mcp) at /mcp, and /api for a local
    // server when one runs.
    server: { fs: { allow: [searchForWorkspaceRoot(process.cwd()), ...(existsSync('node_modules') ? [realpathSync('node_modules')] : [])] },
      proxy: { '/api': process.env.VIV_API || 'http://localhost:8787', '/mcp': process.env.VIV_API || 'http://localhost:8787' } },
    preview: { proxy: { '/api': process.env.VIV_API || 'http://localhost:8787', '/mcp': process.env.VIV_API || 'http://localhost:8787' } },
  },
});
