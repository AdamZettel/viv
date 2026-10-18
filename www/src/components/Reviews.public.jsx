// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The public build's Reviews.jsx (astro.config.mjs swaps it in). Reviews live
// on a server, and the public site has none, so there is nothing to show.

export const reviewsLabel = (reviews) => (reviews.n > 0 ? `${reviews.n} review${reviews.n === 1 ? '' : 's'}` : 'no reviews');

export default function Reviews() {
  return null;
}
