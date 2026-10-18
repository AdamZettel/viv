// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The public build's lib/api.js (astro.config.mjs swaps it in). The public
// site has no server behind /api: hasServer() answers null, so every island
// keeps what the build put on the page, and any call rejects without a request.

export class ApiError extends Error {
  constructor(status, body) { super(body.error || `HTTP ${status}`); this.status = status; this.errors = body.errors || null; this.body = body; }
}

/** Is there a server behind /api? Never, on the public site. */
export const hasServer = () => Promise.resolve(null);

const none = () => Promise.reject(new ApiError(404, { error: 'this site has no server' }));
export const api = new Proxy({}, { get: () => none });

/** "Sep 21, 2026" from an ISO date or timestamp. */
export const dayLabel = (iso, long = false) => new Date(iso.length === 10 ? `${iso}T12:00:00Z` : iso)
  .toLocaleDateString('en-US', { year: 'numeric', month: long ? 'long' : 'short', day: 'numeric', timeZone: 'UTC' });
