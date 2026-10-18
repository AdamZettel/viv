// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
import { useCallback, useEffect, useRef, useState } from 'react';
import Editor, { loader } from '@monaco-editor/react';
import { loadWasm, callWasm } from './VivCell.jsx';
import { vivMonarch, proseRules } from '../lib/vivMonarch.js';

// The site's code editor: Monaco, self-hosted, speaking Viv, with the real
// compiler's diagnostics as red squiggles 350 ms after you stop typing. Colors
// are the site's own token palette (the tk-* variables every code block uses),
// so the editor looks like the code around it in either theme.
//
//   <VivEditor value onChange onRun minLines maxLines />
//
// It grows with its content between minLines and maxLines, then scrolls.
loader.config({ paths: { vs: '/monaco/vs' } });


function themeFromCss() {
  const cs = getComputedStyle(document.documentElement);
  const v = (name) => cs.getPropertyValue(name).trim();
  const hex = (name, fallback) => { const c = v(name); return /^#[0-9a-fA-F]{6}$/.test(c) ? c.slice(1) : fallback; };
  const dark = document.documentElement.dataset.theme !== 'light';
  return {
    base: dark ? 'vs-dark' : 'vs',
    inherit: true,
    rules: [
      { token: 'comment', foreground: hex('--tk-com', '8b95a7'), fontStyle: 'italic' },
      { token: 'comment.doc', foreground: hex('--tk-doc', '5fb6a6') },
      { token: 'string', foreground: hex('--tk-str', 'd0a35c') },
      { token: 'keyword', foreground: hex('--tk-kw', 'e06c8c'), fontStyle: 'bold' },
      { token: 'type', foreground: hex('--tk-type', '4cc2b4') },
      { token: 'identifier', foreground: hex('--text', 'e7e1f2') },
      { token: 'number', foreground: hex('--tk-num', 'a48fe8') },
      { token: 'operator', foreground: hex('--tk-op', '9d91b5') },
      { token: 'delimiter', foreground: hex('--tk-op', '9d91b5') },
      ...proseRules(hex('--text', dark ? 'e7e1f2' : '2c2338'), hex('--text-muted', dark ? 'a99cc0' : '5e5070')),
    ],
    colors: {
      'editor.background': `#${hex('--bg-card', dark ? '221834' : 'ffffff')}`,
      'editor.foreground': `#${hex('--text', dark ? 'e7e1f2' : '2c2338')}`,
      'editor.lineHighlightBackground': `#${hex('--bg-soft', dark ? '1b1329' : 'f4f1f8')}`,
      'editor.selectionBackground': dark ? '#35b3ad55' : '#17706c33',
      'editorLineNumber.foreground': `#${hex('--text-faint', '7b6d94')}`,
      'editorLineNumber.activeForeground': `#${hex('--text-muted', 'a99cc0')}`,
      'editorCursor.foreground': `#${hex('--heading', 'f6f2fc')}`,
      'editorIndentGuide.background': `#${hex('--border', '33254c')}`,
      'editorWidget.background': `#${hex('--bg-soft', '1b1329')}`,
      'editorHoverWidget.background': `#${hex('--bg-soft', '1b1329')}`,
      'editorHoverWidget.border': `#${hex('--border-strong', '493769')}`,
      'editorSuggestWidget.background': `#${hex('--bg-soft', '1b1329')}`,
      'editorSuggestWidget.border': `#${hex('--border-strong', '493769')}`,
      'scrollbarSlider.background': dark ? '#49376980' : '#c5bad680',
    },
  };
}

function registerViv(monaco) {
  if (monaco.languages.getLanguages().some((l) => l.id === 'viv')) return;
  monaco.languages.register({ id: 'viv' });
  monaco.languages.setMonarchTokensProvider('viv', vivMonarch());
  monaco.languages.setLanguageConfiguration('viv', {
    comments: { lineComment: '#' },
    brackets: [['{', '}'], ['[', ']'], ['(', ')']],
    autoClosingPairs: [{ open: '{', close: '}' }, { open: '[', close: ']' }, { open: '(', close: ')' }, { open: '"', close: '"' }],
    indentationRules: { increaseIndentPattern: /\{\s*$/, decreaseIndentPattern: /^\s*\}/ },
  });
}

// the compiler's text diagnostics -> Monaco markers
function markersOf(text, monaco) {
  const markers = [];
  const lines = text.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^error:\s*(.*)$/);
    if (!m) continue;
    let message = m[1];
    let line = 1, col = 1, len = 1;
    for (let j = i + 1; j < Math.min(i + 6, lines.length); j++) {
      const loc = lines[j].match(/-->\s*.*?:(\d+):(\d+)/);
      if (loc) { line = +loc[1]; col = +loc[2]; }
      const caret = lines[j].match(/(\^+)\s*(.*)$/);
      if (caret) { len = caret[1].length; if (caret[2]) message += ` — ${caret[2]}`; break; }
    }
    markers.push({ severity: monaco.MarkerSeverity.Error, message, startLineNumber: line, startColumn: col, endLineNumber: line, endColumn: col + len });
  }
  return markers;
}

export default function VivEditor({ value, onChange, onRun, minLines = 14, maxLines = 40, ariaLabel = 'The program' }) {
  const editorRef = useRef(null);
  const monacoRef = useRef(null);
  const timer = useRef(0);
  const onRunRef = useRef(onRun);
  onRunRef.current = onRun;
  const [height, setHeight] = useState(minLines * 21 + 20);
  const [themeName, setThemeName] = useState('viv-dark');

  const applyTheme = useCallback((monaco) => {
    const dark = document.documentElement.dataset.theme !== 'light';
    const name = dark ? 'viv-dark' : 'viv-light';
    monaco.editor.defineTheme(name, themeFromCss());
    monaco.editor.setTheme(name);
    setThemeName(name);
  }, []);

  const lint = useCallback(async () => {
    const ed = editorRef.current, mo = monacoRef.current;
    if (!ed || !mo) return;
    const ex = await loadWasm();
    const model = ed.getModel();
    if (!model) return;
    const res = callWasm(ex, ex.viv_compile, model.getValue());
    mo.editor.setModelMarkers(model, 'viv', res.includes('#include') ? [] : markersOf(res, mo));
  }, []);

  const fit = useCallback(() => {
    const ed = editorRef.current;
    if (!ed) return;
    const lines = ed.getModel()?.getLineCount() || 1;
    const h = Math.min(Math.max(lines, minLines), maxLines) * 21 + 20;
    setHeight(h);
    ed.layout();
  }, [minLines, maxLines]);

  const onMount = (editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;
    applyTheme(monaco);
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => onRunRef.current && onRunRef.current());
    fit();
    lint();
  };
  useEffect(() => {
    // follow the site's theme switch
    const mo = new MutationObserver(() => { if (monacoRef.current) applyTheme(monacoRef.current); });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => mo.disconnect();
  }, [applyTheme]);
  useEffect(() => () => window.clearTimeout(timer.current), []);

  return (
    <div className="ve" style={{ height }} aria-label={ariaLabel}>
      <Editor
        height="100%"
        defaultLanguage="viv"
        value={value}
        theme={themeName}
        beforeMount={registerViv}
        onMount={onMount}
        onChange={(v) => {
          onChange(v ?? '');
          fit();
          window.clearTimeout(timer.current);
          timer.current = window.setTimeout(lint, 350);
        }}
        options={{
          fontSize: 13,
          fontFamily: "'IBM Plex Mono', ui-monospace, Menlo, monospace",
          lineHeight: 21,
          minimap: { enabled: false },
          scrollBeyondLastLine: false,
          tabSize: 4,
          insertSpaces: true,
          automaticLayout: true,
          padding: { top: 10, bottom: 10 },
          renderLineHighlight: 'line',
          fixedOverflowWidgets: true,
          lineNumbersMinChars: 3,
          folding: false,
          overviewRulerLanes: 0,
          hideCursorInOverviewRuler: true,
          scrollbar: { alwaysConsumeMouseWheel: false, verticalScrollbarSize: 8 },
          wordWrap: 'on',
          wrappingIndent: 'indent',
        }}
      />
    </div>
  );
}
