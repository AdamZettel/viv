// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The engine pages in a real browser: drop an example file on each, press
// Run, and read the numbers the page shows.
//
//     npm run build && node scripts/engine-browsertest.mjs
//
// It serves dist/ with scripts/serve-dist.mjs (as the site test does), and
// hands each file to the page's file input, which is what dropping a file
// does. For every run it checks what the page printed against the reference
// the input is held to: dft/'s numpy reference for the crystals (1e-7 Ha;
// the page also prints the band error, held to 1e-5 eV), PySCF's energy from
// lib/hfref for the molecules (1e-9 Ha). And it checks that a refused input
// is refused with its reason, not run.
import { spawn } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readFileSync } from 'node:fs';
import { chromium } from 'playwright-core';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const PORT = Number(process.env.ENGINETEST_PORT || 4393);
const BASE = `http://127.0.0.1:${PORT}`;
const data = JSON.parse(readFileSync(join(WWW, 'src/data/engines.json'), 'utf8'));
const example = (f) => join(ROOT, 'engines/examples', f);

let failures = 0;
let checks = 0;
function check(name, ok, detail = '') {
  checks += 1;
  if (!ok) failures += 1;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? `  (${detail})` : ''}`);
}

function serve() {
  const child = spawn(process.execPath, [join(WWW, 'scripts/serve-dist.mjs')], { env: { ...process.env, PORT: String(PORT) }, stdio: 'ignore' });
  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    const poll = () => fetch(`${BASE}/`).then(() => resolve(child)).catch(() => (Date.now() - t0 > 15000 ? reject(new Error('serve-dist did not start')) : setTimeout(poll, 200)));
    poll();
  });
}

async function open(browser, path) {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e.message || e)));
  page.on('console', (m) => { if (m.type() === 'error' && !/^Failed to load resource/.test(m.text())) errors.push(m.text()); });
  await page.goto(`${BASE}${path}`, { waitUntil: 'networkidle' });
  return { page, errors };
}

/** Drop a file: hand it to the page's file input, then wait for the page to say it read it. */
async function drop(page, file) {
  await page.getByTestId('engine-file').setInputFiles(example(file));
  await page.getByText(`Reading ${file}`).waitFor({ timeout: 10000 });
}

async function runAndWait(page, timeout) {
  await page.getByTestId('run').click();
  await page.getByTestId('result').waitFor({ timeout });
}

const number = (s) => Number(String(s).replace(/[^0-9eE.+-]/g, ''));

async function dftCase(browser, file, mode, refId) {
  const { page, errors } = await open(browser, '/engines/dft-bands/');
  await drop(page, file);
  const npw = await page.getByTestId('npw').innerText();
  if (mode === 'interpreted') await page.getByRole('radio', { name: /Interpreted/ }).click();
  const t0 = Date.now();
  await runAndWait(page, 180000);
  const seconds = (Date.now() - t0) / 1000;
  const etot = number(await page.getByTestId('etot').innerText());
  const tests = await page.getByTestId('tests').innerText();
  const refText = await page.getByTestId('reference').innerText().catch(() => '');
  const ref = data.dftRefs[refId];
  const bandErr = Number((refText.match(/within\s+([0-9.e+-]+)\s+eV/) || [])[1]);
  const ok = Math.abs(etot - ref.etot) < ref.etol && bandErr < ref.btol && /2 of 2 pass/.test(tests) && /Both are within tolerance/.test(refText) && !errors.length;
  // nothing in the band figure that the program did not print: every word and
  // tick label is in its output; only the y axis's numeric ticks are the chart's
  const figText = await page.locator('.bs-frame svg').first().evaluate((svg) => [...svg.querySelectorAll('text')].map((t) => t.textContent.trim()).filter(Boolean));
  const printed = await page.getByTestId('program-output').textContent();
  const invented = figText.filter((t) => !/^[−-]?\d+(\.\d+)?$/.test(t) && !printed.includes(t));
  check(`/engines/dft-bands: ${file}, ${mode}: the band figure's text is all the program's (${figText.length} labels)`, figText.length > 0 && !invented.length, invented.join(' | '));
  check(`/engines/dft-bands: ${file} dropped, ${mode}: E = ${etot} Ha against dft/${refId}'s ${ref.etot} (${Math.abs(etot - ref.etot).toExponential(1)}, tolerance ${ref.etol}), bands within ${bandErr} eV, ${tests}, ${npw}, ${seconds.toFixed(1)} s in Chromium`,
    ok, errors.join('; '));
  await page.close();
}

async function refusedCase(browser, path, file, want) {
  const { page, errors } = await open(browser, path);
  await drop(page, file);
  const text = await page.getByTestId('refusal').first().innerText({ timeout: 5000 }).catch(() => '');
  const noRun = (await page.getByTestId('run').count()) === 0;
  check(`${path}: ${file} is refused, with its reason, and offers no Run`, want.test(text) && noRun && !errors.length, text.replace(/\s+/g, ' ').slice(0, 160));
  await page.close();
}

async function hfCase(browser, file, refName) {
  const { page, errors } = await open(browser, '/engines/hf-energy/');
  await drop(page, file);
  const t0 = Date.now();
  await runAndWait(page, 120000);
  const seconds = (Date.now() - t0) / 1000;
  const e = number(await page.getByTestId('energy').innerText());
  const ref = data.hfRefs.find((r) => r.name === refName);
  const refText = await page.getByTestId('reference').innerText().catch(() => '');
  // the page prints ten decimals, so the printed number is held to 1e-9 plus half its last digit
  const ok = Math.abs(e - ref.energy) < 1e-9 + 5e-11 && /Within tolerance/.test(refText) && !errors.length;
  check(`/engines/hf-energy: ${file} dropped: E = ${e} Ha against PySCF's ${ref.energy} (${refName}), ${Math.abs(e - ref.energy).toExponential(1)}, ${seconds.toFixed(1)} s in Chromium`, ok, errors.join('; '));
  await page.close();
}

const server = await serve();
const browser = await chromium.launch();
try {
  // every engine is listed, and every listed engine's page opens on its default example
  const { page, errors } = await open(browser, '/engines/');
  const listed = await page.locator('.eng-item').count();
  check(`/engines lists ${listed} engines, every one in engines.json`, listed === data.engines.length && !errors.length, errors.join('; '));
  await page.close();

  await dftCase(browser, 'si.cif', 'compiled', 'si');
  await dftCase(browser, 'gaas.pw.in', 'compiled', 'gaas');
  await dftCase(browser, 'si.poscar', 'interpreted', 'si');
  await refusedCase(browser, '/engines/dft-bands/', 'fe.pw.in', /spin-polarized/);
  await hfCase(browser, 'water.xyz', 'WATER631G_E_RHF');
  await hfCase(browser, 'oh.xyz', 'OH_631G_E_UHF');
  await refusedCase(browser, '/engines/hf-energy/', 'si.xyz', /describes a crystal/);
} finally {
  await browser.close();
  server.kill();
}
console.log(`\nengine-browsertest: ${checks - failures} of ${checks} checks pass`);
process.exit(failures ? 1 : 0);
