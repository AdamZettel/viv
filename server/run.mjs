// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Run a Viv program server-side, the way a reader's tab would: the same
// viv.wasm, the same interpreter. Each run gets a fresh worker with a fresh
// wasm instance, so one program's memory never touches another's, and a
// program that runs too long is terminated with the worker.
//
// Two limits keep a burst of runs from taking the machine down:
//   - a memory maximum on every run. The worker's `resourceLimits` cap the JS
//     heap, not wasm linear memory, and viv.wasm declares its memory with no
//     maximum, so the module is compiled from a copy of the binary whose memory
//     section says `max = VIV_RUN_MEMORY_MB`. `memory.grow` past it fails, the
//     interpreter's allocator aborts, and the run is reported as out of memory.
//   - a global number of runs at once (VIV_RUN_SLOTS). A run that finds every
//     slot busy waits in a queue of at most VIV_RUN_QUEUE runs, for at most
//     `waitMs`; past either it is refused (RunBusy, HTTP 503) rather than
//     queued without end.
// So the memory the runs can hold at once is about slots x (maximum + a
// worker's own heap), whatever the programs do.
//
// The module is compiled once; the compiled Module is what the worker receives.
import { readFileSync } from 'node:fs';
import { Worker } from 'node:worker_threads';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
export const WASM_PATH = process.env.VIV_WASM || join(HERE, '..', 'www', 'public', 'viv', 'viv.wasm');
const PAGE = 64 * 1024;

const envNumber = (name, fallback) => {
  const v = Number(process.env[name]);
  return Number.isFinite(v) && v > 0 ? v : fallback;
};

/** The limits in force; `configureRuns` changes them (the tests do). */
export const limits = {
  slots: envNumber('VIV_RUN_SLOTS', 2),
  queue: envNumber('VIV_RUN_QUEUE', 16),
  memoryMb: envNumber('VIV_RUN_MEMORY_MB', 256),
};

export class RunTimeout extends Error {}
/** Every slot is busy and the queue is full, or the wait ran out. */
export class RunBusy extends Error {
  constructor(message) { super(message); this.status = 503; this.extra = { retryAfter: 5 }; this.transient = true; }
}
/** The program asked for more memory than a run may have, or trapped. */
export class RunFailed extends Error {}
/** The server is shutting down and stopped the run. */
export class RunAborted extends Error {
  constructor(message = 'the server is restarting and stopped this run; try again in a minute') { super(message); this.status = 503; this.transient = true; }
}

// ---- the memory maximum ------------------------------------------------------
const readLeb = (b, p) => {
  let r = 0, shift = 0, x;
  do { x = b[p++]; r += (x & 0x7f) * 2 ** shift; shift += 7; } while (x & 0x80);
  return [r, p];
};
const leb = (n) => {
  const out = [];
  do { let x = n % 128; n = Math.floor(n / 128); if (n) x |= 0x80; out.push(x); } while (n);
  return out;
};

/**
 * A copy of a wasm binary whose one defined memory has `max = maxPages` (64 KB
 * pages). The binary's other sections are copied byte for byte. Throws when the
 * module has no memory section, imports its memory, or needs more than the
 * maximum to start.
 */
export function capMemory(bytes, maxPages) {
  const b = Buffer.from(bytes);
  const parts = [b.subarray(0, 8)];
  let p = 8, found = false;
  while (p < b.length) {
    const id = b[p];
    const [size, body] = readLeb(b, p + 1);
    const end = body + size;
    if (id === 5) {
      let [count, q] = readLeb(b, body);
      if (count !== 1) throw new Error(`capMemory: the module defines ${count} memories, expected 1`);
      const flags = b[q++];
      if (flags & ~0x03) throw new Error(`capMemory: memory flags 0x${flags.toString(16)} (memory64?) are not handled`);
      let min, max = null;
      [min, q] = readLeb(b, q);
      if (flags & 1) [max, q] = readLeb(b, q);
      if (min > maxPages) throw new Error(`capMemory: the module starts with ${min} pages, over the maximum of ${maxPages}`);
      const newMax = max == null ? maxPages : Math.min(max, maxPages);
      const payload = Buffer.from([...leb(1), flags | 1, ...leb(min), ...leb(newMax), ...b.subarray(q, end)]);
      parts.push(Buffer.from([5, ...leb(payload.length)]), payload);
      found = true;
    } else {
      parts.push(b.subarray(p, end));
    }
    p = end;
  }
  if (!found) throw new Error('capMemory: the module has no memory section (an imported memory is not handled)');
  return Buffer.concat(parts);
}

let compiledFor = null;
let modulePromise = null;
function compiled() {
  if (!modulePromise || compiledFor !== limits.memoryMb) {
    compiledFor = limits.memoryMb;
    modulePromise = WebAssembly.compile(capMemory(readFileSync(WASM_PATH), Math.floor((limits.memoryMb * 1024 * 1024) / PAGE)));
  }
  return modulePromise;
}

// ---- slots -------------------------------------------------------------------
let running = 0;
const waiting = []; // { grant, refuse, timer }
const workers = new Set();
let peak = { running: 0, waiting: 0 };
let closed = false;

function acquire(waitMs) {
  if (closed) return Promise.reject(new RunAborted());
  if (running < limits.slots) { running++; peak.running = Math.max(peak.running, running); return Promise.resolve(); }
  if (waiting.length >= limits.queue) {
    return Promise.reject(new RunBusy(`the server is running ${running} programs and has ${waiting.length} waiting; try again in a few seconds`));
  }
  return new Promise((grant, refuse) => {
    const entry = { grant, refuse, timer: null };
    entry.timer = setTimeout(() => {
      waiting.splice(waiting.indexOf(entry), 1);
      refuse(new RunBusy(`the server was busy running other programs for ${waitMs / 1000} s; try again in a few seconds`));
    }, waitMs);
    waiting.push(entry);
    peak.waiting = Math.max(peak.waiting, waiting.length);
  });
}
function release() {
  const next = waiting.shift();
  if (next) { clearTimeout(next.timer); next.grant(); } // the slot passes straight to the next run
  else running--;
}

/** Change the limits (slots, queue, memoryMb); returns the previous ones. */
export function configureRuns(next) {
  const before = { ...limits };
  Object.assign(limits, next);
  return before;
}
/** Runs in progress and waiting now, and the most of each seen since the last reset. */
export function runStats({ reset = false } = {}) {
  const s = { running, waiting: waiting.length, peak: { ...peak }, limits: { ...limits } };
  if (reset) peak = { running, waiting: waiting.length };
  return s;
}

/** Wait until no run is in progress or waiting, for at most `ms`. True if it got there. */
export async function drain(ms) {
  const deadline = Date.now() + ms;
  while (running > 0 || waiting.length > 0) {
    if (Date.now() >= deadline) return false;
    await new Promise((r) => setTimeout(r, 50));
  }
  return true;
}
/** Refuse new runs, and stop the ones in progress and waiting (shutdown). */
export function abortAll() {
  closed = true;
  for (const w of waiting.splice(0)) { clearTimeout(w.timer); w.refuse(new RunAborted()); }
  for (const stop of workers) stop();
}
/** Accept runs again (for tests that abort). */
export function reopenRuns() { closed = false; }

// ---- one run -----------------------------------------------------------------
/**
 * @returns {Promise<{ out: string, seconds: number, hash: string, functions, ladder }>} -- `ladder` when asked for: every rung run.
 * `waitMs` bounds the wait for a free slot; `onStart` is called when the run gets one.
 * The time limit starts when the run starts, not while it waits.
 * `module` is a compiled interpreter to use instead of viv.wasm as it is (the MCP server passes one with a memory limit).
 */
export async function runProgram(src, { timeoutMs = 20000, ladder = false, waitMs = 5000, onStart = null, module: given = null } = {}) {
  const module = given || await compiled();
  await acquire(waitMs);
  if (onStart) onStart();
  let settled = false;
  return new Promise((resolve, reject) => {
    const w = new Worker(join(HERE, 'run-worker.mjs'), {
      workerData: { module, src, ladder },
      resourceLimits: { maxOldGenerationSizeMb: 64, maxYoungGenerationSizeMb: 16 },
    });
    const done = (fn, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      workers.delete(stop);
      w.terminate();
      release();
      fn(value);
    };
    const stop = () => done(reject, new RunAborted());
    workers.add(stop);
    const timer = setTimeout(() => done(reject, new RunTimeout(`the program ran for more than ${timeoutMs / 1000} s and was stopped`)), timeoutMs);
    w.once('message', (m) => {
      if (m.trap) {
        // an allocation past the maximum is the usual cause; the worker says how
        // much memory the run held when it stopped, so the message can tell
        const full = m.memoryBytes >= limits.memoryMb * 1024 * 1024 * 0.5;
        done(reject, new RunFailed(full
          ? `the program ran out of memory: a run may use at most ${limits.memoryMb} MB (it held ${Math.round(m.memoryBytes / 1048576)} MB when an allocation failed)`
          : `the program stopped the interpreter (${m.trap}); an allocation over the ${limits.memoryMb} MB limit does this`));
      } else done(resolve, m);
    });
    w.once('error', (e) => done(reject, e));
    w.once('exit', (code) => done(reject, new Error(`the program's run stopped unexpectedly (exit code ${code}); it may have run out of memory`)));
  });
}
