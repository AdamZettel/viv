// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// /agents: the layout's copy buttons (src/lib/chrome.js) change their own text
// when pressed; this says the result aloud too, through the page's live region.
document.addEventListener('click', (e) => {
  const b = e.target instanceof Element ? e.target.closest('.copy-btn') : null;
  const status = document.getElementById('copy-status');
  if (!b || !status) return;
  setTimeout(() => {
    status.textContent = b.textContent === 'Copied' ? 'Copied to the clipboard' : 'Could not copy; select the text and copy it';
  }, 50);
});
