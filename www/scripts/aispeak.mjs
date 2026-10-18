// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// aispeak -- a lint for the register an AI writes in when nobody stops it.
//
//     node scripts/aispeak.mjs            everything below
//     node scripts/aispeak.mjs --test     only the rules' own examples
//     node scripts/aispeak.mjs path...    named files
//     node scripts/aispeak.mjs --strict   gated rules fail everywhere (the deploy runs this)
//
// It reads text written for readers, not code:
//   - a post's fenced header (title, summary, notes), its rung and figure
//     titles, its `#|` notes, its test names, its `#` comments (they show in
//     the code panel) and its `show` lines (they print a name to the reader);
//   - channel syllabi, the READMEs of posts/, server/, engines/ and dft/, and
//     the design docs, with fenced code left out;
//   - the Jupyter kernel's README and example notebooks (tools/jupyter);
//   - the site's pages and components: the text between tags, attribute
//     values, and the string literals that read as prose;
//   - the string literals of the server and of www/src/lib (its .js and .ts),
//     which is where error messages, the lane notes and the topic pitches live;
//   - the vipyr gallery's scripts: their docstrings and `# ` markdown lines;
//   - the doc examples' titles and notes;
//   - the vipyr reference's generated prose (www/src/data/vipyr.json) and README;
//   - the feature catalog (lang/viv/features): each entry's title, what and
//     when, and the builtin table's notes and descriptions;
//   - the intent lens: the prose of every .intent file under lib/, scf/,
//     posts/ and dft/.
//
// PROVENANCE. A post that names an `agent:` was drafted by one and is linted.
// A post with no agent line is the author's own writing and is skipped
// entirely: the point is to catch the machine's tics, not to sand down a
// person's voice.
//
// Every rule carries examples it must flag (`yes`) and examples it must pass
// (`no`); they run before the lint, so a rule cannot be widened into a false
// positive without a test saying so. A match is a prompt to look, and
// `aispeak-ok` on the line, or on the line before it, says the writer looked
// and meant it.
import { readFileSync, readdirSync, existsSync, statSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse as babelParse } from '@babel/parser';
import { fencedHeader, parseFenced, meta, notesOf } from '../src/lib/postCheck.js';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);

// What a piece of text is, so a rule can be aimed:
//   title    a post, rung, figure or doc title
//   prose    paragraphs: notes, summaries, markdown, page copy
//   ui       a string literal in a component, the server or a library
//   test     a test's name
//   comment  a `#` comment in a post's program
//   code     a line of a post's program
const WORDS = ['title', 'prose', 'ui', 'test'];

// Acronyms and symbols that are allowed in capitals inside a test name.
const ACRONYMS = new Set(`AI API LAPACK LOBPCG BOYS ASCII BFGS BLAS CCSD CFL CG CIF CPU CSE DFT DIIS DOS ELO EOS ERI FFT FIR GCD GMRES GPU GTO HF HFB HOMO HTML IEEE IIR JSON KE LCG LDA LHS LQR LSB LU LUMO MD MLP MP2 MSB NAND NOR NaN ODE OK PBE PDE PDF PID QR RAM RHF RHS RK RK2 RK4 RMS RNG ROHF SCF SGD SI SIMD SOR SPD STO SVD TOV UHF URL USD UTF XOR XYZ AND OR NOT IO LIFO FIFO GELU RELU ReLU ADAM TDSE`.split(' '));

// The rules written for posts, which read badly on docs that enumerate
// contents ("def, return, docstrings, recursion, and optional type
// annotations") or state scope ("nothing else depends on it"), keep the reach
// they were tuned on: posts, channels, ladders, lib and the site's pages.
const POSTS_AND_SITE = /^(posts|claims|show|channels|ladders|lib|www\/src\/(?!data\/))/;
// Compiler vocabulary is the subject of the design docs and lib's notes
// ("lowering an index" is a recurrence there); it is jargon only on the site.
const NOT_DESIGN_OR_LIB = /^(?!design\/|lib\/)/;

const RULES = [
  // ---- the original rules ----------------------------------------------------
  { id: 'title-aphorism', re: /^[^.?!,]{6,60},\s+(and|but|not)\s+[^.?!]{4,}$/i, kinds: ['title'], only: POSTS_AND_SITE,
    say: 'a title shaped like an aphorism ("One gate, and everything else from it"). Say what it is.',
    yes: ['One gate, and everything else from it'], no: ['A four-bit adder built from NAND gates', 'The frontier: arrays, eigensolves, and the SCF'] },
  // the site, the design docs and the channels; the titles of posts and the
  // feature catalog's are the owner's to change, so the rule leaves them
  { id: 'slogan-heading', kinds: ['title', 'ui'], only: /^(www\/src\/|design\/|channels\/)/,
    re: /^[^.!?]{2,60}, (?:one \w+ at a time|by example|via (?:the|a|an) [^.!?]{2,30}|(?:\w+est|most \w+|least \w+) first|(?!only\b)\w{3,}ly)$/i,
    say: 'a heading shaped like a slogan ("Noun, adverb", "X, one Y at a time", "X, by example", "X, via the Y", "X, tightest first"). Label what the section contains.',
    yes: ['The language, one feature at a time', 'Plots, explicitly', 'Every construct, by example', 'Live, via the embed', 'Operators, tightest first'],
    no: ['A tour of Viv', 'Operator precedence', 'Examples of every construct', 'Water, Hartree-Fock and the basis set', 'Embedding a live cell'] },
  // the same slogan inside a sentence: "the energies of molecules, one approximation at a time"
  // (channels and the site; post titles are the owner's, and lib's notes use
  // the phrase for what a loop does: "one column at a time")
  { id: 'one-at-a-time', kinds: ['prose', 'title'], only: /^(channels\/|www\/src\/(?!data\/))/,
    re: /, one (?!at\b)\w+ at a time\b/i,
    say: 'a slogan ("X, one Y at a time"). Say what the thing covers instead.',
    yes: ['The energies of molecules from their electrons, one approximation at a time: Hartree-Fock', 'Physics, one problem at a time'],
    no: ['The cells run one at a time.', 'It adds one term at a time to the sum.'] },
  // the scene titles the owner flagged (2026-10-14): a title that narrates the
  // page's mechanics instead of naming its subject
  { id: 'scene-title', kinds: ['title'], only: /^(?!design\/|lang\/viv\/features\/)/,
    // warns until the content pass clears the posts that still say it; --strict (the deploy) fails
    gate: /^www\/src\/(components|layouts)\//,
    re: /^A run of\b|, (?:call by call|step by step|one \w+ at a time)$|\b(?:computed |run |running |solved )?in (?:this|your) (?:tab|browser)\b/i,
    say: 'a title that narrates the page ("A run of ...", "..., call by call", "..., step by step", "..., one X at a time", "computed in this tab"). Name the subject: "Newton\'s method on x^2 - 2".',
    yes: ['A run of Newton\'s method', 'The SCF loop, call by call', 'Gaussian elimination, step by step', 'Ising model, one sweep at a time', 'Silicon bands computed in this tab', 'A GPT trained in your browser'],
    no: ['Newton\'s method on x^2 - 2', 'The SCF loop', 'Silicon band structure', 'Running a program'] },
  { id: 'scene-phrase', kinds: ['title', 'prose', 'ui'], only: /^(?!design\/|lang\/viv\/features\/)/,
    // warns until the content pass clears the posts that still say it; --strict (the deploy) fails
    gate: /^www\/src\/(components|layouts)\//,
    re: /\bPrograms you can run here\b|\bEach page is one program\b/i,
    say: 'a stock line about the page ("Programs you can run here", "Each page is one program"). Say what the programs compute.',
    yes: ['Programs you can run here', 'Each page is one program, run in your browser.'],
    no: ['Programs that compute band structures', 'Each program prints its energy.'] },
  { id: 'story-heading', kinds: ['title', 'ui'], only: /^(www\/src\/|design\/)/,
    re: /^[A-Z][^.!?·]{2,60}, then [^.!?]{2,40}$|^(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|\d+) (?:things|bugs|mistakes|ways|lessons|surprises|fixes)\b[^.!?]*$|^What (?:the|it|a|an|we|I|this) [^.!?]{2,40}(?:had to|needed to|got wrong|learned|taught us)\b[^.!?]*$|\bgot wrong(?: first)?$|^How (?:we|I) [^.!?]{2,50}$/,
    say: 'a heading that tells a story ("The eigensolver converged, then diverged", "Three things it got wrong first", "What the language had to grow"). Name the topic: "Bugs fixed during development".',
    yes: ['The eigensolver converged, then diverged', 'Three things it got wrong first', 'What the language had to grow', 'How we fixed the cache'],
    no: ['Bugs fixed during development', 'Language features this needed', 'LOBPCG search direction', 'What a test prints'] },
  { id: 'title-colon-reveal', re: /^[^:]{4,50}:\s+(the|a|an)\s+one\b|^[^:]{4,50}:\s+what\b/i, kinds: ['title'], only: POSTS_AND_SITE,
    say: 'a title that withholds ("X: the one that is not obvious"). Name the thing.',
    yes: ['Stiff solvers: the one that is not obvious'], no: ['Stiff solvers compared on the Robertson problem'] },
  { id: 'from-nothing', re: /\bfrom nothing\b|\bnothing but\b|\bout of nothing\b/i,
    say: 'grandiosity ("from nothing"). Name the parts it is built from.',
    yes: ['a CPU built from nothing'], no: ['a CPU built from NAND gates'] },
  { id: 'absolutes', only: POSTS_AND_SITE, re: /\b(everything else|nothing else|the only thing|every other one|anything at all)\b/i,
    say: 'a sweeping absolute. Is it true, and does it earn the sentence?',
    yes: ['and everything else follows'], no: ['the other gates follow from NAND'] },
  { id: 'antithesis', re: /\b(is|are|was|were|does|do)\s+not\s+[^,.;]{2,40},\s+(it|they|that)\s+(is|are|does|do)\b/i,
    say: 'the "not X, it is Y" turn. State Y.',
    yes: ['This is not a trick, it is algebra.'], no: ['This is algebra.'] },
  // the denial as a tail: "would mean the method is wrong, not the code."
  { id: 'antithesis-tail', only: POSTS_AND_SITE, re: /\b(?:is|are|was|were|means?|would be|checks?|tests?)\b[^.;:]{1,60}, not (?:the|its|their|a|an) [a-z][\w-]*(?: [a-z][\w-]*)?\.(?:\s|$)/i,
    say: 'the "X, not the Y" tail. Say what is true and drop the denial.',
    yes: ['A failure here would mean the method is wrong, not the code.', "so this checks the implementation's rounding, not a market."],
    no: ['A failure here would mean the method is wrong.', 'A and b are copied, not changed.'] },
  // the same denial with "rather than" at the end of the sentence: "relaxes to
  // half its start rather than to zero."
  { id: 'rather-than-tail', only: POSTS_AND_SITE, re: /\brather than (?:to |at |by |in |on )?(?:the |its |their |a |an )?[\w$\\^{}()+\/-]+(?: [a-z][\w-]*)?(?:\.(?:\s|$)|, and\b)/i,
    say: 'the "X rather than Y" tail. Say what is true; if the contrast matters, give it its own sentence.',
    yes: ['It relaxes to $k^2/(1+k^2)$ of its start rather than to zero.', 'a zonal flow that relaxes to half its start rather than to zero, and the'],
    no: ['It relaxes to $k^2/(1+k^2)$ of its start.', 'It keeps a momentum, not the raw gradient: M = mu M + G.'] },
  // three clauses on one stem: "Same program, same answers, same tests passing."
  { id: 'triplet', only: POSTS_AND_SITE, re: /\b(same|one|no|every) [^,.;]{2,30}, \1 [^,.;]{2,30}, (?:and )?\1 [^,.;]{2,30}[.!]/i,
    say: 'a rhythmic triplet on one word ("Same program, same answers, same tests passing."). Write one sentence that says it.',
    yes: ['Same program, same answers, same tests passing.'],
    no: ['The program, its answers and its tests are the same.', 'one input, two outputs, three tests.'] },
  { id: 'portent', re: /\bthat is the whole\b|\bwhich is the point\b|\bthat sentence is\b|\band that is (that|all)\b|\bnothing more\b|\bthe whole (thesis|bet|trick|idea)\b|\bin the truest sense\b|\bgenuinely novel\b|\bthe capstone\b/i,
    say: 'a portentous closer or frame. Delete it or replace it with the fact.',
    yes: ['That is the whole method.', 'the whole thesis in one line', 'That division is the whole bet'], no: ['The method takes three lines.'] },
  { id: 'the-catch', re: /\b(and )?(here|there) is the catch\b|\bthe name is the catch\b|\bthe catch is\b/i,
    say: 'a reveal. Say the limitation plainly.',
    yes: ['Here is the catch: it diverges.'], no: ['It diverges for h > 2.'] },
  { id: 'is-what-x-is', re: /\bis what (a|an|the)\b[^.]{0,40}\bis\b/i,
    say: 'a definition that circles. Define it once, directly.',
    yes: ['This is what a gradient is.'], no: ['A gradient is the vector of partial derivatives.'] },
  { id: 'blacklist', re: /\bload-bearing\b|\bthe shape (does not|doesn't) change\b|\bdeep dive\b|\bunder the hood\b|\bat its core\b|\bit is worth noting\b|\bimportantly\b|\bseamless(ly)?\b|\bleverage[sd]?\b|\brobust(ly)?\b|\bthe real question\b/i,
    say: 'a phrase on the blacklist. Say what you mean in plain words.',
    yes: ['Three load-bearing decisions'], no: ['Three decisions'] },
  { id: 'honest-tic', re: /\bhonest(ly)?\b|\bto be honest\b/i,
    say: '"honest" used as a tic ("Scope is honest", "pedagogically honest"). Delete it or say what is true.',
    yes: ['Scope is honest and bounded', 'Honestly, it fails.'], no: ['The scope is two gates.'] },
  { id: 'intensifier', re: /\b(very|really|quite|extremely|incredibly|remarkably|surprisingly|genuinely|actually|truly|fundamentally|essentially)\s+\w+/i,
    say: 'an intensifier. Cut it or give the number.',
    yes: ['it actually reads the file', 'it genuinely needs arrays', 'fundamentally cannot'], no: ['it reads the file'] },
  // a dash standing alone is a table's empty cell, not punctuation
  { id: 'em-dash-pile', re: /\w[,;]? ?—[^—|]{1,80}— ?\w/,
    say: 'two em dashes in one breath. Use one, or a full stop.',
    yes: ['a — b — c'], no: ['a — b.', '| — | 180 | 258 | — |', 'chain    —    2.67 s    —'] },
  // words, not numbers: "2, 4 and 16" is data; "clean, fast and simple" is a flourish
  // and acronyms are names: "RHF, UHF and ROHF" lists three methods
  { id: 'rule-of-three', re: /\b([A-Za-z][a-z]{2,}), ([A-Za-z][a-z]{2,}),? and ([A-Za-z][a-z]{2,})\b[^.]{0,20}\.\s*$/, kinds: ['title', 'prose'], only: POSTS_AND_SITE,
    say: 'a rule-of-three flourish at the end of a sentence. Does the third item do work?',
    yes: ['It is clean, fast and simple.', 'Clean, fast and simple.'], no: ['It runs 2, 4 and 16 steps.', 'The loop, written once for RHF, UHF and ROHF.'] },

  // ---- vague claims where the checked fact belongs ---------------------------
  { id: 'vague-verdict',
    // followed by a colon, the fact comes next ("the order still holds: -x^2 + 1 = 0 has roots -1 and 1")
    re: /\b(it|they|that|this|which|still|both|and|claim|result|post)\s+(reproduc(?:es|ed|e)|holds?(?: up)?|held up)(?=\s*(?:[.,;)!?]|$|\s(?:and|but|or)\b))|\bholds? up\b(?!\s+(?:the|a|an|its|their|\w+ing)\b)|\ba review that reproduces\b|\bwhether (it|the \w+) (holds|reproduces)\b/i,
    say: 'a verdict word with nothing after it ("it reproduced", "the claim holds", "holds up"). Say what was run, against what, and what matched.',
    yes: ['Another member ran it and it reproduced.', 'Several have, and it still reproduces', 'Both hold.', 'the claim holds', 'reports whether it holds up', 'the compensated one holds up'],
    no: ['It reproduces the 1971 figure to three digits.', 'the bound holds for every n > 2', 'The kernel holds a copy of the matrix.', 'the order still holds: -x^2 + 1 = 0 has roots -1 and 1', 'the two ropes must hold up the weight'] },
  { id: 'vague-check', re: /\blocked (it )?to (PySCF|the reference|\w+'s)\b|\bpinned in a (check|test) file\b|\bholds? the \w+ to its (rules|claim)\b|\bheld to the same rules\b|\bearned (by|through) (review|reviews|other people)/i,
    say: 'a vague verification verb ("locked to PySCF", "pinned in a check file", "earned by review"). Say what is compared, to what tolerance, and where.',
    yes: ['the energy is locked to PySCF', 'pinned in a check file', 'holds the ladder to its rules', 'earned by review, not chosen'],
    no: ['the energy matches PySCF to 1e-10 Ha', 'Pin the version in package.json.'] },
  { id: 'label-by-name', re: /\bpast the (AI|Draft|Reviewed|Published) mark\b/,
    say: 'a label defined by its own name ("past the Reviewed mark"). Give the number or the rule.',
    yes: ['a rating past the Reviewed mark'], no: ['a rating past 1130'] },
  { id: 'loose-ratio', re: /\b(about|roughly|nearly|around|a little over|a little under) (half|twice|a third|double|a quarter) as (much|far|many)\b/i,
    say: 'a loose ratio. Give the exact one (a vote moves the rating 40/64 as far as a review).',
    yes: ['a vote counts about half as much as a review', 'Votes move it about half as far.'], no: ['a vote moves it 40/64 as far'] },
  { id: 'hidden-tolerance', re: /\bnear enough\b|\ba few percent\b|\bto machine (epsilon|precision)\b/i,
    say: 'a tolerance hidden behind a phrase. Give the number the test asserts.',
    yes: ['near enough to the exact value', 'to a few percent of the shift', 'agrees to machine epsilon relative'],
    no: ['agrees with the exact value to 4%', 'relative error below 1e-14'] },

  // ---- jargon shown to readers -----------------------------------------------
  { id: 'compiler-jargon', only: NOT_DESIGN_OR_LIB, re: /\blower(?:s|ed)? (it )?(to|into)\b|\blowering( time)?\b|\bdualiz\w+|\bnominal value type\b|\bshared handles?\b|\bemit trait\b|\bscalar core\b|\bgrowth front\b|\bparity pair\b|\bthe host adds nothing\b/i,
    say: 'compiler-internal vocabulary shown to a reader. Say what the reader sees happen.',
    yes: ['the compiler lowers it to C', 'Compiled runs the program as Viv lowered to C', 'at lowering time', 'deriv dualizes the body', 'arrays and lambdas are the growth front', 'covers the scalar core'],
    no: ['a down-vote lowers the rating', 'the compiler turns it into C'] },
  // the intent lens (.intent files) explains code to its reader, and a
  // registry's "member" is the term lib/hf defines for one of its strategies
  // as the design docs use it
  { id: 'site-jargon', only: /^(?!design\/)(?!.*\.intent$)/, re: /\b(another|a|every|production|each|the) members?\b(?! (function|functions|variable|variables|of|field|fields))|\bmembers (post|vote|review|run)\b|\bescrow(ed)?\b|\brun worker\b|\bthe adapters?\b|\bunknown verb\b|\bthe harness\b|\btweens?\b|\b\d+ defs\b|\ba def or two\b/i,
    say: 'site or code vocabulary shown to a reader ("member", "escrow", "run worker", "adapter", "harness", "def"). Say the plain thing.',
    yes: ['Another member ran it', 'Members post demos', 'Escrow 30 cr', 'the run worker exited with code 1', 'The adapter failed on this input', 'runs it under the harness', 'morph tweens a curve', '+2 defs'],
    no: ['Another reader ran it', 'the members of the struct', 'each member function', 'Hold 30 credits'] },
  { id: 'stale-header', raw: true, re: /`#! (post|from|channel|lane|agent|kind|tags|summary|date|solves|edge)\b|\bno `#! \w+` line\b/,
    say: 'the old `#! key value` header syntax. Posts now use the `key: value` header block.',
    yes: ['`#! from ${h.from}` names no post', 'no `#! post` line'], no: ['`from: ${h.from}` names no post', 'the `#!` lines of a ladder'] },
  { id: 'wasm-arrays', re: /\b(wasm|WebAssembly)\b[^.]{0,80}\b(no|not|only|cannot|can't|needs?)\b[^.]{0,40}\barrays?\b|\barrays?\b[^.]{0,40}\b(not|only|cannot|can't)\b[^.]{0,20}\b(supported|yet)\b[^.]{0,40}\b(wasm|WebAssembly)\b/i,
    say: 'a capability claim about the direct wasm backend and arrays. Check it against the compiler (lang/) before it ships.',
    yes: ['The direct wasm backend does not compile arrays yet', 'the WebAssembly backend handles scalar code only; it cannot compile arrays'],
    no: ['The direct wasm backend compiles arrays and lambdas', 'arrays of any length'] },

  // ---- metaphor, idiom, slogans -----------------------------------------------
  { id: 'idiom', re: /\bstays? put\b|\b(does not|doesn't) move\b(?! (a|an|the|any)\b)|\b(has not been|been|was) run down\b|\bdrifts? apart\b|\bsurvived and\b|\bcould not survive\b|\bfor free\b|\bcomes? free\b|\ball the way down\b|\btrusts you\b|\bvanish(es)? at the\b|\bcall-graph drama\b|\bwatch the error collapse\b|\bthat is a lot of\b|\bis a picture\b|\bsomething broke\b|\ba review says something\b/i,
    say: 'an idiom or metaphor standing in for a statement ("stays put", "for free", "all the way down"). Say what happens.',
    yes: ['the body hash stays put', 'the address doesn\'t move', 'it has not been run down', 'the two outputs drift apart', 'CSE for free', 'OpenMP and WASM all come free', 'Floats all the way down', 'eig_sym trusts you', 'Comments vanish at the lexer', 'that is a lot of runs', 'a demo with no check is a picture', 'something broke on the server'],
    no: ['the body hash is unchanged', 'the address is the same', 'the error falls to 1e-12', 'a free variable', 'a constant, so it does not move a band'] },
  { id: 'dismissive', only: POSTS_AND_SITE, re: /\bit is trivial\b|\bis trivial\b|\btrivially\b|\bjust (?:turns|works|runs|happens|moves|adds|rotates)\b|\beverything interesting\b|\bnothing else happens\b|\b(?:a|the) \w+'s habits\b|\bboth of (?:a|the) \w+'s\b/i,
    say: 'a dismissal or mannered aside ("it is trivial", "each mode just turns", "everything interesting happens", "a fluid\'s habits"). Say what happens and why it is easy.',
    yes: ['In Fourier space it is trivial -- each mode just turns', 'so everything interesting happens where they come closest', "The smallest equation with both of a fluid's habits in it", 'it moves, and nothing else happens'],
    no: ['In Fourier space each mode is multiplied by a phase', 'the bands come closest at K', 'a trivial representation'] },
  { id: 'slogan', re: /(?:^|[.!?]\s+)(?:[A-Z][\w-]+(?: [\w-]+){0,2}\. ){2}[A-Z][\w-]*(?: [\w-]+){0,2}\.(?:\s|$)|^[A-Z][^.:!?]{2,40} is not [^.:!?]{2,30}:|\bCompile to C, inherit everything\b/,
    say: 'a slogan: clipped fragments in a row ("One program. Three backends. No server.") or an aphorism opener ("Agreement is not proof:"). Write the sentence.',
    yes: ['One program. Three backends. No server.', 'Agreement is not proof: the two engines share a parser.', 'Compile to C, inherit everything C has.'],
    no: ['One program runs on three backends without a server.', 'The engines agree. They share a parser, so a parser bug would not show.'] },
  { id: 'filler-adjective', re: /\bthe real (backend|compiler|thing|work|code|program|arrays?|C)\b|\breal arrays\b|\bgenuinely\b|\bclean (compile|build|output)? ?error\b|\ba clean (compile|build|run)\b/i,
    say: '"real", "genuinely" or "clean" as filler ("the real backend", "a clean compile error"). Delete it or say what distinguishes it.',
    yes: ['the C is from the real backend', 'Viv had real arrays', 'genuinely held in a variable', 'a clean compile error'],
    no: ['the real part of z', 'Real math', 'a compile error that names the line'] },
  { id: 'throat-clearing', re: /(?:^|[.!?]\s+)(So the question is|Here is (what|how|why)|Before any algebra|This page (shows|explains|is about|walks)|Note that|It is worth noting|Let's (take a look|dive))\b/,
    say: 'throat-clearing. Start with the fact.',
    yes: ['So the question is how fast it converges.', 'Here is what that costs.', 'Before any algebra, a guess.', 'Note that the step is fixed.'],
    no: ['The error halves when the step halves.'] },
  { id: 'filler-hedge', re: /\bwhatever (that|it) is\b|\b(\w+) or (\w+) of the \3\b/i,
    say: 'a filler hedge ("whatever that is", "eight or nine of the nine"). Give the count, or cut it.',
    yes: ['the leading term, whatever that is', 'eight or nine of the nine'], no: ['eight of the nine'] },
  { id: 'rate-modifier', re: /\b\d+(\.\d+)?% (monthly|yearly|annual|daily|weekly) \w+ing (doubles|grows|takes|halves)\b/i,
    say: 'a misattached modifier ("At 7% monthly compounding doubles"). Give the sentence a subject.',
    yes: ['At 7% monthly compounding doubles in 10 months'], no: ['At 7% monthly compounding, money doubles in 10 months'] },
  { id: 'british', re: /\b(behaviour|colour|favour|honour|neighbour|labour|centre|metre|litre|analyse[sd]?|catalogue|modelling|modelled|travelled|licence|defence|(optimi|normali|initiali|minimi|maximi|recogni|organi|summari|visuali|paralleli|vectori|seriali|generali|characteri|speciali|lineari|discreti|regulari)s(e|ed|es|ing|ation|ations))\b/i,
    say: 'British spelling. Use American English.',
    yes: ['the colour of the line', 'normalised to one', 'the optimisation step'], no: ['the color of the line', 'normalized to one'] },

  // ---- test names --------------------------------------------------------------
  { id: 'test-caps', kinds: ['test', 'comment'],
    // a name the program itself uses (STEPS, TOL) is an identifier, not emphasis
    re: { exec: (s, idents = new Set()) => { for (const m of s.matchAll(/(?<![.\w])[A-Z]{3,}\b/g)) if (!ACRONYMS.has(m[0]) && !idents.has(m[0])) return m; return null; } },
    say: 'capitals for emphasis in a test name ("own EXACTLY as many points"). The assert says how exact; use lower case.',
    yes: ['own EXACTLY as many points', 'Simulate the NONLINEAR cart-pole'], no: ['the RHF energy matches PySCF', 'the FFT of a delta is flat', 'pyscf.lib.param.BOHR'] },
  { id: 'test-colon', kinds: ['test'], re: /^(?:[^\s:]+ ){0,2}[^\s:]+:\s|:[^:]*:/,
    say: 'a test name that opens with a label and a colon ("halving h: ...") or chains colons. Name the one thing the assert checks.',
    yes: ['halving h: the three errors shrink by 2, 4 and 16', 'hydrogen-like uranium: every level is -Z^2/2n^2', 'a: b: c'],
    no: ['the step commutes with the shift', 'the base of the logarithm does not matter: base 10 gives the same time'] },
  { id: 'test-is-the-one', kinds: ['test'], re: /\bis the (\w+ )?(?!\w+er )\w+ one\b/i,
    say: 'a test name that claims identity ("the linear model is the nonlinear one"). Name what is compared and the tolerance.',
    yes: ['the linear model is the nonlinear one', 'the n-times formula is the original one'], no: ['the linear model matches the nonlinear one to 1e-6 near the top', 'at 3% the rule of 70 is the closer one'] },

  // ---- a post's program --------------------------------------------------------
  { id: 'show-internal', kinds: ['code'], re: /^\s*show\s+\w*(member|family|fam_|swept|stage)\w*/,
    say: 'a `show` that prints an internal name to the reader. Rename it to what the number is.',
    yes: ['show eri_families_swept', 'show fam_len', 'show blamed_member_of_three'], no: ['show energy_hartree', 'show n_steps'] },
  { id: 'mode-comment', kinds: ['comment'], re: /\b(release mode|first pass)\b/i,
    say: 'library-internal vocabulary in a comment readers see in the code panel.',
    yes: ['# in release mode this is fast', '# In this first pass we skip it'], no: ['# the step size halves each time'] },

  // ---- data shown raw ------------------------------------------------------------
  { id: 'raw-key', kinds: ['jsxexpr'], raw: true, re: /^[\w.?]+\.(verdict|lane|edge|status|state)$/,
    say: 'a stored key rendered as text ("@alice: reproduces"). Map it to a label written for the reader.',
    yes: ['r.verdict', 'post.lane'], no: ['r.verdictText', 'LANE_LABELS[post.lane]', 'post.title'] },

  // ---- site chrome ----------------------------------------------------------------
  // a heading on the site, a doc example's title or a channel's heading: one
  // plain phrase. Post titles are their authors' and are left alone.
  { id: 'heading-punct', kinds: ['title'], only: /^(www\/src\/(?!data\/)|www\/src\/data\/vivDocExamples\.json$|channels\/)/,
    re: /^\d+[a-z]? — |[^\s:]: (?=\S)| — /,
    say: 'a colon or an em dash in a heading ("1 — Numbers", "Output: print and fmt"). Use one plain phrase.',
    yes: ['1 — Numbers', 'Output: print and fmt', 'Roadmap: from spans to content addresses', 'Narration — progress lines'],
    no: ['Numbers', 'Output with print and fmt', 'Hartree–Fock in STO-3G', 'Ratio 1:2'] },
  // "X doesn't; Y does": the denial and the claim in one breath
  { id: 'antithesis-turn', only: /^(www\/src\/|gallery\/|channels\/|lib\/|posts\/)/,
    re: /\b(?:do|does|did|is|are|was|were|can|will)(?: not|n't)\b[^.;:]{0,80}; [^.;:]{1,80}\b(?:do|does|did|is|are|was|were|can|will)\.(?:\s|$)|\b(?:isn't|aren't|wasn't|doesn't|don't)\s+[^,.;]{2,40},\s+(?:it's|they're|it is|they are)\b/i,
    say: 'the "X doesn\'t; Y does" or "isn\'t A, it\'s B" turn. State what is true.',
    yes: ['Comments and spacing do not change it; changing what the program computes does.', 'Python doesn\'t; Viv does.', 'It isn\'t a trick, it\'s algebra.'],
    no: ['It does not converge; the step is too large.', 'Changing what the program computes changes it.'] },
  // the denial that opens on what it is not: "The trouble is not the formula."
  { id: 'denial-opener', only: /^(www\/src\/|gallery\/|channels\/|posts\/|claims\/)/,
    re: /\bThe (?:trouble|problem|point|issue|catch|difficulty|question) (?:is|was) not\b|\bexists to answer one question\b|\bto answer one question:/,
    say: 'a sentence that frames by denial or announces a question. Say what the trouble is.',
    yes: ['The trouble is not the formula.', 'It exists to answer one question: how'],
    no: ['The recurrence carries both solutions.', 'The problem is ill-conditioned.'] },
  // the appositive denial: "Forward-mode autodiff, not finite differences"
  { id: 'antithesis-appositive', only: /^(www\/src\/|gallery\/|channels\/|posts\/)/,
    re: /^[A-Z][^.;:,]{2,50}, not (?:a |an |the )?[a-z][^.;:,]{2,40}:|, not a wrong (?:result|answer|number)\b/,
    say: 'the "X, not Y" turn. State X; drop the denial.',
    yes: ['Forward-mode autodiff, not finite differences: d/dx', 'an error at the line, not a wrong result when it runs'],
    no: ['deriv uses forward-mode autodiff.', 'A and b are copied, not changed.', 'if it does not, the loop stops'] },
  // speed claims against other packages ("100 times faster than groupby().apply()")
  { id: 'speed-brag', only: /^(www\/src\/|gallery\/|channels\/|posts\/|python\/)/,
    re: /\b\d[\d.,]*(?:\s*(?:to|-|–)\s*\d[\d.,]*)?\s*(?:x|×|times)\s+(?:faster|quicker)\s+than\b|\b(?:faster|quicker) than (?:NumPy|SciPy|pandas|PyTorch|Python|Julia|MATLAB|R|C|Fortran|Numba|JAX)\b|\boutperform(?:s|ed)?\b/i,
    say: 'a speed comparison with another package ("100 times faster than groupby().apply()"). Say what it computes; a measured timing belongs in a table with the machine named.',
    yes: ['run custom per-group logic 100 to 200 times faster than groupby().apply()', 'faster than NumPy', 'it outperforms SciPy'],
    no: ['Newton converges in 5 steps where bisection takes 50', 'the error falls faster than any power of N'] },
  // a vague verification verb: "held to mpmath at 50 digits", "held to agree"
  { id: 'held-to', only: /^(www\/src\/|gallery\/|channels\/|posts\/)/,
    re: /\bheld to (?:agree\b|mpmath\b|SciPy\b|NumPy\b|PySCF\b|NIST\b|R's\b|the reference\b|the closed\b|its bits\b|upstream\b|[A-Z][a-z]+(?:'s| and)\b)/,
    say: 'a vague verification verb ("held to mpmath at 50 digits"). Say what is compared and how close: "agrees with mpmath to 1e-15".',
    yes: ['the special functions under them held to mpmath at 50 digits', 'held to agree', "is held to Fang and Oosterlee's value", 'held to SciPy at 2e-14'],
    no: ['when generators are held to their reactive limits', 'agrees with mpmath to 1e-15'] },
  // a title that narrates where or how the page computes ("Water's energy, computed in this tab")
  { id: 'where-computed', kinds: ['title', 'ui'], only: /^(www\/src\/(pages|data\/scenes|data\/showcases)|posts\/|show\/|gallery\/|channels\/)/,
    re: /\b(?:in|inside) (?:this|your|the reader's) (?:tab|browser)\b|\bcomputed here\b|\brun live\b|\byou can run here\b|\bPrograms you can run\b/i,
    say: 'a title or heading that says where the page computes ("in this tab", "in your browser"). Name the physics.',
    yes: ["Water's Hartree-Fock energy, computed in this tab", 'A band structure, computed in your browser', 'Programs you can run here'],
    no: ["Water's electron density settles", 'Hartree-Fock for water'] },
  // the owner's standing rule (2026-10-15): the reader's page is never "this
  // tab". Every string a reader sees, not only titles; code comments are not
  // extracted, so they stay free to say it.
  { id: 'browser-tab', kinds: ['title', 'prose', 'ui'], only: /^(?!design\/)/,
    re: /\b(?:in|inside|from|on) (?:this|your|the reader's|a) tab\b|\b(?:this|your) tab's\b|\b(?:your|the|a) tab (?:runs|stays|keeps|remains)\b|\b(?:in|inside|for|from) the tab\b(?! (?:key|character|bar|order|stop|strip|list|panel|button)s?\b)|\bthe tab(?:'s| is offered| can (?:not )?take| would)\b/i,
    say: 'the reader\'s page called a tab ("in this tab", "in your tab", "this tab\'s", "too large for the tab"). Say "in your browser", "your run", or drop it.',
    yes: ['Computed in this tab in 8.1 s by viv.wasm.', 'Short programs that run in your tab', "this tab's copy", 'N = 200 runs in a tab in a tenth of a second', 'by the same interpreter your tab runs', 'It starts a Web Worker so the tab stays responsive', 'too heavy for the tab; its recorded run is from the nightly check', 'A 2000-particle PIC code in the tab measures', 'more than the 90 s the tab is offered'],
    no: ['Computed in 8.1 s by viv.wasm.', 'Short programs that run in your browser', 'Press the tab key to indent', 'Open the Notes tab to see them', 'insert a tab character'] },
  // title shapes that narrate the presentation instead of naming the subject
  { id: 'title-narrates', kinds: ['title', 'ui'], only: /^(www\/src\/(pages|data\/scenes|data\/showcases)|posts\/|show\/|gallery\/|channels\/)/,
    re: /^A run of\b|, (?:call|step|line|frame|cell) by (?:call|step|line|frame|cell)$|, one \w+ at a time$|, computed (?:in|by|here)\b/i,
    say: 'a title that narrates the presentation ("A run of X", "X, call by call", "X, one step at a time"). Name the subject.',
    yes: ['A run of the oxygen-16 solver, call by call', 'The density, step by step', 'Silicon, one band at a time', 'The energy, computed in Viv'],
    no: ['Solving for the ground state of oxygen-16', 'Two solitons pass through each other'] },
  { id: 'instead-of-opener', kinds: ['title', 'prose', 'ui'], only: /^(www\/src\/|gallery\/|channels\/)/,
    re: /(?:^|[.!?]\s+)Instead of\b/,
    say: 'a sentence that opens on what the thing is not ("Instead of hello-world, ..."). Start with what it is.',
    yes: ['Instead of hello-world, a program of the kind Viv was built for.'], no: ['A first program: Hartree-Fock on H2.', 'It uses QR instead of the normal equations.'] },
  // closers that point somewhere else instead of ending ("To learn how one of these is put together, ...")
  { id: 'signpost-closer', kinds: ['prose', 'ui'], only: /^(www\/src\/|gallery\/|channels\/)/,
    re: /(?:^|[.!?]\s+)To (?:learn|see|find out|read) (?:how|what|more|why)\b/,
    say: 'a signposting sentence ("To learn how one of these is put together, ..."). A link in the text or nothing.',
    yes: ['To learn how one of these is put together, a ladder builds a program.'], no: ['The ladder builds the program one step at a time.'] },
  // explainers the owner removed: how the site works, said on every page
  { id: 'site-explainer', kinds: ['prose', 'ui'], only: /^www\/src\//,
    re: /\bis a rough measure\b|\bOne version so far\b|\bThis history comes from\b|\bruns on its own, without the cells\b|\bComments and spacing do not change\b|\bRun a program, then read\b/,
    say: 'an explainer of the site\'s own machinery that the owner removed. Delete it.',
    yes: ['Owned by a person is a rough measure for now', 'One version so far.', 'Every cell below runs on its own, without the cells above it'],
    no: ['Run a cell to see its output beside the expected one.'] },

  // a page explaining its own machinery to the reader: how the site is built,
  // what the compiler hashes, how many times something was checked. The design
  // docs and the reference are where that belongs. `gate` names the files where
  // a hit fails the lint; elsewhere it is printed as a warning until the
  // wording sweep reaches it.
  { id: 'self-machinery', kinds: ['prose', 'ui', 'title'], only: /^(?!design\/|www\/src\/pages\/(docs|design)\/)/,
    gate: /^(ladders\/|www\/src\/pages\/ladders\/|www\/src\/components\/LadderRungs\.jsx$)/,
    re: /\bthe site (does not|doesn't|will not|won't) build if\b|\bchecks? (this|that|it|them) \d+ times\b|\b(\d+) of the \3 (are|were) (tested|checked|verified)\b|\bevery one of the \d+ (is|was) (tested|checked|verified)\b/i,
    say: 'the page explains its own machinery ("the site does not build if", "checks this 27 times", "12 of the 12 are tested"). Say what the reader gets; the machinery belongs in design/ or the reference.',
    yes: ['The compiler checks this 27 times on this ladder, and the site does not build if any check fails.', '12 of the 12 are tested.'],
    no: ['Each step is a complete program that runs.', 'The test checks the energy to four digits.', '3 of the 12 are tested.'] },
  { id: 'self-machinery-address', kinds: ['prose'], only: /^(?!design\/|www\/src\/pages\/(docs|design|reference|cli)\b|lang\/viv\/features\/|AGENTS\.md$|posts\/README\.md$)/,
    gate: /^(ladders\/|www\/src\/pages\/ladders\/|www\/src\/components\/LadderRungs\.jsx$)/,
    re: /\bcontent address(es)?\b/i,
    say: '"content address" in reader prose: the compiler\'s hashing is a design-doc and reference topic. Say what the reader can rely on instead.',
    yes: ['Beside each function is its content address.'], no: ['Beside each function is what it computes.'] },
  // ---- sentence shape ------------------------------------------------------------
  { id: 'colon-chain', kinds: ['prose', 'ui'], re: /(?:^|[.;]\s)[^.;:{}[\]]{2,60}:\s[^.;:{}[\]]{1,60}:\s+\S/,
    say: 'a colon chain ("X: Y: Z"). Write it as a sentence.',
    yes: ['Open the branch: fix: title', 'C backend output: C emitted: 12 lines'], no: ['no test ran: add a test block', 'Energy (Ha): -74.96', 'The two claims, [claim: a-b] and [claim: c-d], cover it', '> record Vec3 { x: Num, y: Num, z: Num }'] },
  { id: 'fragment', kinds: ['prose'], para: true, re: /^(?!(?:Yes|No|Done|Thanks|See|Try|It|This|That|They|We|You|He|She|Both)\b)[A-Z][a-z]+(?: [a-z'-]+)?\.(?=\s+[A-Z]|$)/,
    say: 'a paragraph that opens with a fragment ("Three things.", "Several have."). Start with a full sentence.',
    yes: ['Three things. First, the step.', 'Two changes.'], no: ['The step halves. Then the error halves.', 'See below.', 'It diverges. The step is too large.'] },
  // a note the author left for themself, published as copy (/agents, 2026-10-16)
  { id: 'internal-note', kinds: ['prose', 'ui'], only: /^(?!design\/|lang\/viv\/features\/|AGENTS\.md$|CHANGELOG\.md$|TODO\.md$)/,
    re: /\bnot (?:been )?recorded here\b|\b(?:has|have|was|were) not (?:been )?(?:checked|tried|verified|tested) yet\b/i,
    say: 'a note to the author ("was not recorded here", "has not been checked yet"). Find the fact and state it, or leave the sentence out.',
    yes: ['Where Cursor keeps that file was not recorded here.', 'Whether the form accepts it has not been checked yet.'],
    no: ['The file is .cursor/mcp.json in a project.', 'The run was checked against SciPy.'] },
  // Viv runs in three engines (interpreter, C, wasm): "both engines" is from before the third
  { id: 'engine-count', kinds: ['prose', 'ui'], only: /^www\/src\//,
    re: /\bboth engines\b/i,
    say: 'Viv has three engines (the interpreter, the C backend and the wasm backend). Name the ones meant, or say every engine.',
    yes: ['Both engines refuse this.', 'It gives the same output in both engines.'],
    no: ['Every engine refuses this.', 'The interpreter and the C backend agree.'] },
];

/** Rules whose `re` is a plain regex get the usual exec; strip code spans and
 *  math before matching unless the rule asks for the raw text. */
const plain = (s) => s.replace(/`[^`]*(`|$)/g, '‹code›').replace(/\$[^$]+\$/g, '‹math›');

// ---- extraction ------------------------------------------------------------------

/** Text items: [{ text, line, kind, para }]. */
function vivItems(file, src) {
  // the author's own writing is exempt; a post names its agent in `agent:`, a
  // claim (claims/*.viv) in `by:`
  if (!meta(src, 'agent') && !/^(claude|codex|agent)\b/i.test(meta(src, 'by'))) return [];
  const out = [];
  // the capitalized names the program uses, with strings and comments taken out
  const idents = new Set(src.split('\n').filter((l) => !/^\s*#/.test(l)).join('\n').replace(/"(?:[^"\\]|\\.)*"/g, ' ').replace(/#.*$/gm, ' ').match(/\b[A-Z][A-Z0-9_]{2,}\b/g) || []);
  const push = (text, line, kind, para = false) => { if (text && text.trim()) out.push({ text: text.trim(), line, kind, para, idents }); };
  const fence = fencedHeader(src);
  const lines = src.split('\n');
  if (fence) {
    const { fields, body } = parseFenced(fence.text);
    push(fields.post, 1 + fence.start, 'title');
    // the fields a page prints as prose: a post's summary, and a claim's
    // statement, scope and reference (claims/*.viv, shown at /p/<slug>)
    for (const k of ['summary', 'claim', 'scope', 'reference']) push(fields[k], 1 + fence.start, 'prose', true);
    let bl = body.split('\n');
    const off0 = fence.text.split('\n').length - bl.length;
    // fields parseFenced stops at: `scale <name>:` and `recorded <name>:`, and any after them
    let lead = 0;
    while (lead < bl.length && /^[a-z][\w -]*:\s/i.test(bl[lead])) {
      const f = bl[lead].match(/^summary:\s*(.*)$/);
      if (f) push(f[1], fence.start + 2 + off0 + lead, 'prose', true);
      lead++;
    }
    bl = bl.slice(lead);
    const off = off0 + lead;
    bl.forEach((l, i) => { push(l, fence.start + 2 + off + i, 'prose', i === 0 || !bl[i - 1].trim()); if (out.length && bl[i + 1]?.trim()) out[out.length - 1].next = bl[i + 1].trim(); });
  } else {
    push(meta(src, 'post'), 1, 'title');
    push(meta(src, 'summary'), 1, 'prose', true);
    notesOf(src).forEach((p) => push(p, 1, 'prose', true));
  }
  // a claim's page prints its header; its code, tests and comments are held
  // to the claims' own conventions (CLAIM_AUTHORING.md) and recorded runs
  if (/^claims\//.test(relative(ROOT, file))) return out;
  const inFence = (i) => fence && i >= fence.start && i <= fence.end;
  lines.forEach((l, i) => {
    if (inFence(i)) return;
    const rung = l.match(/^\s*rung\s+"((?:[^"\\]|\\.)*)"/);
    if (rung) push(rung[1], i + 1, 'title');
    const note = l.match(/^\s*#\|\s?(.*)$/);
    if (note) { push(note[1], i + 1, 'prose'); push(note[1], i + 1, 'comment'); return; }
    const fig = l.match(/^\s*(?:figure|stage)\("((?:[^"\\]|\\.)*)"/);
    if (fig) push(fig[1], i + 1, 'title');
    const test = l.match(/^\s*test\s+"((?:[^"\\]|\\.)*)"/);
    if (test) push(test[1], i + 1, 'test');
    const comment = l.match(/(?:^|\s)#(?![!|])\s?(.*)$/);
    // an SPDX license line (LICENSING.md) is not prose
    if (comment && !/"[^"]*#/.test(l) && !/^\s*SPDX-/.test(comment[1])) push(comment[1], i + 1, 'comment');
    if (/^\s*show\b/.test(l)) push(l, i + 1, 'code');
  });
  return out;
}

function mdItems(src) {
  const out = [];
  let code = false;
  const lines = src.split('\n');
  const unquote = (l) => (l || '').replace(/^\s*>\s?/, '');
  // front matter (`---` lines around `key: value`): only the values a reader
  // sees (a title, a reason) are prose; the rest are settings
  const front = lines[0] === '---' ? lines.indexOf('---', 1) : -1;
  lines.forEach((l, i) => {
    if (i <= front) {
      const m = l.match(/^(title|reason):\s*(.+)$/);
      if (m) out.push({ text: m[2].trim(), next: '', line: i + 1, kind: m[1] === 'title' ? 'title' : 'prose', para: true });
      return;
    }
    l = unquote(l);
    if (/^\s*(```|~~~)/.test(l)) { code = !code; return; }
    if (code || /^( {4}|\t)/.test(l) || /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(l)) return;
    const para = i === 0 || !lines[i - 1].trim() || /^#/.test(lines[i - 1]);
    const text = l.replace(/^#+\s*/, '').replace(/^\s*[-*]\s+/, '').replace(/^\s*\d+\.\s+/, '');
    const after = unquote(lines[i + 1]);
    const next = code || /^#/.test(l) || !after.trim() || /^\s*(```|~~~|#|[-*|]\s|\d+\.\s)/.test(after) ? '' : after.trim();
    if (text.trim()) out.push({ text: text.trim(), next, line: i + 1, kind: /^#/.test(l) ? 'title' : 'prose', para: para && !/^\s*[-*|]/.test(l) });
  });
  return out;
}

// Attributes and call arguments whose strings are code, not copy.
const CODE_ATTRS = new Set(['className', 'class', 'key', 'id', 'href', 'src', 'type', 'style', 'role', 'rel', 'target', 'lang', 'name', 'value', 'd', 'viewBox', 'fill', 'stroke', 'transform', 'data-testid', 'htmlFor', 'method', 'accept', 'autoComplete', 'inputMode', 'pattern']);
const CODE_CALLS = /^(querySelector(All)?|getElementById|closest|matches|addEventListener|removeEventListener|fetch|require|RegExp|URL|createElement|setAttribute|getAttribute|toggleAttribute|hasAttribute|removeAttribute|add|remove|toggle|contains|startsWith|endsWith|includes|split|join|replace|replaceAll|indexOf|padStart|padEnd|get|set|has|prepare|exec|run|all|getItem|setItem|removeItem|postMessage|localeCompare|toLocaleString|toFixed|test|match|search|Worker|import)$/;

/** Is this string something a reader will see as a sentence or a label? */
function readsAsProse(s) {
  if (!/[A-Za-z]{2,}[^\sA-Za-z]*\s+[A-Za-z(]{2,}[^\s]*\s+[A-Za-z(]{2,}/.test(s)) return false; // three words
  if (/\b(SELECT|INSERT INTO|UPDATE \w+ SET|DELETE FROM|CREATE (TABLE|INDEX)|WHERE|ALTER TABLE)\b/.test(s)) return false;
  if (/[{};]\s*$|^\s*[.#]?[\w-]+\s*\{|:\s*[\w#().,% -]+;/m.test(s)) return false; // css or code
  if (/^[\w./-]+(\s+[\w./-]+)*$/.test(s) && s.split(/\s+/).every((w) => /[-_./]/.test(w))) return false; // class lists, paths
  if (/^\s*(def|let|show|return|for|while|if|import|export|const|function|test)\b.*[(){}=]/m.test(s)) return false; // programs
  return true;
}

function jsItems(src, { jsx = false, ts = false, lineOffset = 0 } = {}) {
  let ast;
  try {
    ast = babelParse(src, { sourceType: 'module', plugins: jsx ? ['jsx'] : ts ? ['typescript'] : [], errorRecovery: true, allowReturnOutsideFunction: true, allowAwaitOutsideFunction: true });
  } catch (e) {
    return [{ error: `could not parse: ${e.message}` }];
  }
  const out = [];
  // a label (button text, a tooltip) can be one word and still be read
  const push = (text, node, label = false) => {
    const t = String(text).replace(/\s+/g, ' ').trim();
    if (t && (label ? /[A-Za-z]{3}/.test(t) : readsAsProse(String(text)) && readsAsProse(t))) out.push({ text: t, line: node.loc.start.line + lineOffset, kind: 'ui' });
  };
  const LABEL_ATTRS = new Set(['title', 'aria-label', 'placeholder', 'alt', 'label']);
  const skip = new Set();
  const walk = (node, parent) => {
    if (!node || typeof node.type !== 'string') return;
    switch (node.type) {
      case 'ImportDeclaration': case 'ExportAllDeclaration': return;
      case 'ExportNamedDeclaration': if (node.source) skip.add(node.source); break;
      case 'CallExpression': case 'NewExpression': {
        const c = node.callee;
        const name = c.type === 'Identifier' ? c.name : c.type === 'MemberExpression' && !c.computed ? c.property.name : '';
        if (CODE_CALLS.test(name)) for (const a of node.arguments) if (a.type === 'StringLiteral' || a.type === 'TemplateLiteral') skip.add(a);
        break;
      }
      case 'BinaryExpression': if (/^[!=]==?$/.test(node.operator)) { skip.add(node.left); skip.add(node.right); } break;
      case 'ObjectProperty': case 'ObjectMethod': case 'ClassProperty': if (!node.computed) skip.add(node.key); break;
      case 'MemberExpression': if (node.computed) skip.add(node.property); break;
      case 'SwitchCase': if (node.test) skip.add(node.test); break;
      case 'JSXAttribute': {
        const n = node.name.name?.name ?? node.name.name;
        if (CODE_ATTRS.has(n)) return;
        if (LABEL_ATTRS.has(n) && node.value?.type === 'StringLiteral') { push(node.value.value, node.value, true); return; }
        break;
      }
      case 'TaggedTemplateExpression': return; // css`...`, sql`...`
      case 'JSXElement': if (node.openingElement.name.name === 'style' || node.openingElement.name.name === 'code' || node.openingElement.name.name === 'pre') return; break;
      case 'StringLiteral': if (!skip.has(node) && parent?.type !== 'Directive') push(node.value, node); return;
      case 'TemplateLiteral': if (!skip.has(node)) push(node.quasis.map((q) => q.value.cooked ?? q.value.raw).join('…'), node); break;
      case 'JSXText': push(node.value, node, true); return;
      case 'JSXExpressionContainer':
        // a data key rendered as it is stored: `{r.verdict}` shows "reproduces"
        if ((parent?.type === 'JSXElement' || parent?.type === 'JSXFragment') && node.expression.type === 'MemberExpression')
          out.push({ text: src.slice(node.expression.start, node.expression.end), line: node.loc.start.line + lineOffset, kind: 'jsxexpr' });
        break;
    }
    for (const k of Object.keys(node)) {
      if (k === 'loc' || k === 'start' || k === 'end' || k === 'leadingComments' || k === 'trailingComments' || k === 'innerComments') continue;
      const v = node[k];
      if (Array.isArray(v)) v.forEach((c) => walk(c, node)); else if (v && typeof v.type === 'string') walk(v, node);
    }
  };
  walk(ast.program, null);
  return out;
}

function astroItems(src) {
  const out = [];
  const lines = src.split('\n');
  // the front matter is TypeScript-flavored JS
  let body = 0;
  if (lines[0].trim() === '---') {
    const end = lines.indexOf('---', 1);
    if (end > 0) {
      out.push(...jsItems(lines.slice(1, end).join('\n'), { lineOffset: 1 }).filter((x) => !x.error));
      body = end + 1;
    }
  }
  let skipBlock = false;
  lines.forEach((l, i) => {
    if (i < body) return;
    if (/<(style|script|pre|code)\b/.test(l) && !/<\/(style|script|pre|code)>/.test(l)) { skipBlock = true; return; }
    if (skipBlock) { if (/<\/(style|script|pre|code)>/.test(l)) skipBlock = false; return; }
    if (/^\s*(import|const|let|function|export|\/\/)/.test(l)) return;
    // strings inside {expressions} on this line
    for (const m of l.matchAll(/\{[^{}]*\}/g)) for (const s of m[0].matchAll(/'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)"/g)) {
      const t = s[1] ?? s[2];
      if (readsAsProse(t)) out.push({ text: t, line: i + 1, kind: 'ui' });
    }
    // a heading tag's text, and a page's heading= attribute, are titles
    for (const h of l.matchAll(/<h([1-6])\b[^>]*>(.*?)<\/h\1>|\bheading="([^"]+)"/g)) {
      const t = (h[2] ?? h[3]).replace(/<[^>]*>/g, '').replace(/\{[^}]*\}/g, '').trim();
      if (t) out.push({ text: t, line: i + 1, kind: 'title' });
    }
    const text = l.replace(/<code>.*?<\/code>/g, ' ‹code› ').replace(/<[^>]*>/g, ' ').replace(/\{[^}]*\}/g, ' ').replace(/\b(class|className|href|id|src|style|type)=("[^"]*"|'[^']*')/g, ' ');
    if (/[a-z]{3}\s+[a-z]{3}/.test(text)) out.push({ text: text.trim(), line: i + 1, kind: 'prose' });
  });
  return out;
}

function jsonItems(src, keys) {
  const out = [];
  const lines = src.split('\n');
  lines.forEach((l, i) => {
    const m = l.match(/^\s*"(\w+)":\s*"((?:[^"\\]|\\.)*)"/);
    if (m && keys.includes(m[1])) out.push({ text: JSON.parse(`"${m[2]}"`), line: i + 1, kind: m[1] === 'title' ? 'title' : 'prose', para: true });
  });
  return out;
}

/** The prose of an `.intent` file: the
 *  module's description, each field's and glossary line's text after its
 *  key, step texts and detail lines. Headers, `path` titles' ids and `$$`
 *  equations are not prose. */
export function intentItems(src) {
  const out = [];
  const lines = src.split('\n');
  const textOf = (l) => {
    const t = l.trim();
    if (!t || t.startsWith('$$')) return null;
    if (!/^\s/.test(l)) return null; // a header, `module ...` or `path ...`
    // a citation names a test or a hold; its words are not the writer's
    const cited = (x) => x.replace(/\[(test|hold|post|program|claim) "[^"]*"\]/g, '');
    const keyed = /^(\S+)\s{2,}(.*)$/.exec(t);
    if (keyed) return cited(keyed[2]);
    if (/^(=|>|why|how|run|predict)\s/.test(t)) return cited(t.replace(/^\S+\s+/, ''));
    return cited(t);
  };
  const texts = lines.map(textOf);
  const indent = (l) => l.length - l.trimStart().length;
  // a line continues the one above when it is indented deeper and has no key
  // of its own; only then is it read with it (a glossary line's neighbor is
  // another name's)
  const continues = (i) => !!texts[i] && !!lines[i - 1] && indent(lines[i]) > indent(lines[i - 1]) && !/^\S+\s{2,}/.test(lines[i].trim());
  texts.forEach((text, i) => {
    if (text) out.push({ text, line: i + 1, kind: 'prose', para: false, next: continues(i + 1) ? texts[i + 1] : '' });
  });
  return out;
}

/** The prose of the feature catalog (lang/viv/features): an entry's title,
 *  `what` and `when` (with the lines that continue them), and builtins.txt's
 *  group notes and the description column of its rows. */
export function featureItems(file, src) {
  const out = [];
  const lines = src.split('\n');
  if (file.endsWith('builtins.txt')) {
    lines.forEach((l, i) => {
      if (l.startsWith('note: ')) out.push({ text: l.slice(6), line: i + 1, kind: 'prose', para: true });
      else if (l.includes('\t')) out.push({ text: l.split('\t')[2] || '', line: i + 1, kind: 'prose', para: false });
    });
    return out;
  }
  lines.forEach((l, i) => {
    const m = /^(title|what|when): (.*)$/.exec(l);
    if (!m) return;
    let text = m[2];
    for (let k = i + 1; k < lines.length && /^  \S/.test(lines[k]); k++) text += ' ' + lines[k].trim();
    out.push({ text, line: i + 1, kind: m[1] === 'title' ? 'title' : 'prose', para: true });
  });
  return out;
}

/** A gallery script's reader-facing text (python/vipyr/_gallery.py): its
 *  docstring (the page's title and opening) and its `# ` markdown lines. */
function galleryPyItems(src) {
  const out = [];
  const lines = src.split('\n');
  const DQ = '"'.repeat(3);
  let inDoc = false;
  let docDone = false;
  lines.forEach((l, i) => {
    if (!docDone && l.trim().startsWith(DQ)) {
      if (inDoc) { inDoc = false; docDone = true; return; }
      inDoc = true;
      const t = l.trim().slice(3);
      if (t.trim()) out.push({ text: t.trim(), line: i + 1, kind: 'title', para: false });
      return;
    }
    if (inDoc) {
      const t = l.trim();
      if (t && !/^=+$/.test(t)) out.push({ text: t, line: i + 1, kind: /^=+$/.test((lines[i + 1] || '').trim()) ? 'title' : 'prose', para: false, next: (lines[i + 1] || '').trim() });
      return;
    }
    const m = l.match(/^# (.*)$/);
    if (m && !m[1].startsWith('%%')) out.push({ text: m[1].replace(/^#+\s*/, ''), line: i + 1, kind: /^#+ /.test(m[1]) ? 'title' : 'prose', para: false, next: (lines[i + 1] || '').replace(/^# ?/, '').trim() });
  });
  return out;
}

/** A shell script: its `# ` comments (not the shebang or SPDX lines) and the
 *  double-quoted strings it prints with echo, printf or a helper such as step. */
function shItems(src) {
  const out = [];
  src.split('\n').forEach((l, i) => {
    const c = l.match(/^\s*# (.*)$/);
    if (c && !/^SPDX-/.test(c[1])) { out.push({ text: c[1], line: i + 1, kind: 'prose', para: false }); return; }
    if (/^\s*(echo|printf|step|show)\b/.test(l)) {
      for (const m of l.matchAll(/"([^"$]*[A-Za-z][^"]*)"/g)) {
        if (m[1].trim().split(/\s+/).length > 2) out.push({ text: m[1], line: i + 1, kind: 'prose', para: false });
      }
    }
  });
  return out;
}

export function itemsOf(file) {
  const src = readFileSync(file, 'utf8');
  if (/lang\/viv\/features\/\w+\.txt$/.test(file)) return featureItems(file, src);
  if (file.endsWith('.intent')) return intentItems(src);
  if (file.endsWith('.viv')) return vivItems(file, src);
  // PROVENANCE, as for a post with no agent line: a markdown file that opens
  // with `<!-- by: user` is the owner's own writing (src/data/story.md, the
  // story on the home page) and is skipped whole. The headings and captions
  // the page puts around it live in index.astro and are linted there.
  if (file.endsWith('.md') && /^<!-- by: user\b/.test(src)) return [];
  if (file.endsWith('.md')) return mdItems(src);
  if (file.endsWith('llms.txt')) return mdItems(src);
  if (file.endsWith('.sh')) return shItems(src);
  if (file.endsWith('.astro')) return astroItems(src);
  if (file.endsWith('.jsx')) return jsItems(src, { jsx: true });
  if (/\.(m?js)$/.test(file)) return jsItems(src);
  // TypeScript in src/lib (site.ts holds the topic pitches)
  if (file.endsWith('.ts')) return jsItems(src, { ts: true });
  if (file.endsWith('.py') && /(^|\/)gallery\//.test(file)) return galleryPyItems(src);
  if (file.endsWith('vivDocExamples.json')) return jsonItems(src, ['title', 'intro', 'note']);
  if (file.endsWith('vipyr.json')) return jsonItems(src, ['summary', 'help', 'note', 'doc', 'about', 'text', 'where']);
  return [];
}

// ---- the rules' own tests -----------------------------------------------------------

/** A match that starts on this line; `next` (the following line of the same
 *  paragraph) is there so a clause broken across lines is read whole. */
function matchRule(rule, text, idents, next = '') {
  const s = rule.raw ? text : plain(text);
  if (!next) return rule.re.exec(s, idents);
  const joined = rule.raw ? `${text} ${next}` : plain(`${text} ${next}`);
  const m = rule.re.exec(joined, idents);
  return m && m.index < s.length ? m : null;
}

function selfTest() {
  const fails = [];
  for (const rule of RULES) {
    for (const y of rule.yes || []) if (!matchRule(rule, y)) fails.push(`${rule.id}: should flag  ${JSON.stringify(y)}`);
    for (const n of rule.no || []) { const m = matchRule(rule, n); if (m) fails.push(`${rule.id}: should pass  ${JSON.stringify(n)}  (matched ${JSON.stringify(m[0])})`); }
    if (!(rule.yes || []).length || !(rule.no || []).length) fails.push(`${rule.id}: needs at least one example each way`);
  }
  // the extractors: only human-facing strings
  const js = jsItems(`import x from 'a b c d';\nconst q = document.querySelector('.feed-item the main one');\nif (mode === 'run it and see') {}\nthrow new Error('the post has no title line here');\nconst o = { 'a key with words': 1 };\n`).map((x) => x.text);
  if (js.join('|') !== 'the post has no title line here') fails.push(`jsItems: expected only the error message, got ${JSON.stringify(js)}`);
  const jsx = jsItems(`const A = () => <p className="fp-vote fp-vote-down on">Run it in your browser<style>{\`.a b c { color: red; }\`}</style><b title="vote that the post is right">x</b></p>;`, { jsx: true }).map((x) => x.text);
  const button = jsItems('const B = () => <button type="submit">Escrow {n} cr</button>;', { jsx: true }).map((x) => x.text);
  const raw = jsItems('const L = () => <li>@{r.reviewer}: {r.verdict}</li>;', { jsx: true }).filter((x) => x.kind === 'jsxexpr').map((x) => x.text);
  if (raw.join('|') !== 'r.reviewer|r.verdict') fails.push(`jsItems(jsx): rendered member expressions, got ${JSON.stringify(raw)}`);
  if (button.join('|') !== 'Escrow') fails.push(`jsItems(jsx): a one-word button label is read, got ${JSON.stringify(button)}`);
  if (jsx.join('|') !== 'Run it in your browser|vote that the post is right') fails.push(`jsItems(jsx): got ${JSON.stringify(jsx)}`);
  const prog = jsItems('const EX = { src: `# A signal, its envelope: functions do the physics\ndef envelope(t) {\n    return exp(t)\n}` };').map((x) => x.text);
  if (prog.length) fails.push(`jsItems: a program in a template literal is code, got ${JSON.stringify(prog)}`);
  const wrapped = mdItems('Arrays still hold\nnumbers only.\n');
  if (matchRule(RULES.find((r) => r.id === 'vague-verdict'), wrapped[0].text, undefined, wrapped[0].next)) fails.push('mdItems: a clause broken across lines was read as two');
  const md = mdItems('Three things. First.\n\n```\nThe real backend is here.\n```\n    indented code line here\n').map((x) => x.text);
  if (md.join('|') !== 'Three things. First.') fails.push(`mdItems: got ${JSON.stringify(md)}`);
  return fails;
}

// ---- main ----------------------------------------------------------------------------

const args = process.argv.slice(2);
// --strict: the gated rules fail everywhere, not only in their gated files.
// `npm run build` (what the deploy runs) and the gate's www-static step use it.
const strict = args.includes('--strict');
const selfFails = selfTest();
if (selfFails.length) {
  console.log('aispeak: the rules fail their own examples:\n  ' + selfFails.join('\n  '));
  process.exit(2);
}
if (args.includes('--test')) {
  console.log(`aispeak: ${RULES.length} rules pass their examples`);
  process.exit(0);
}

const files = [];
const add = (p) => { if (existsSync(p)) files.push(p); };
const walk = (d, re, skipDir = /^(node_modules|test|dist)$/) => {
  if (!existsSync(d)) return;
  for (const f of readdirSync(d)) {
    const p = join(d, f);
    if (statSync(p).isDirectory()) { if (!skipDir.test(f)) walk(p, re, skipDir); } else if (re.test(f)) add(p);
  }
};
const named = args.filter((a) => !a.startsWith('--'));
if (named.length) named.forEach(add);
else {
  for (const dir of ['posts', 'channels', 'ladders', 'lib', 'show', 'pde']) {
    const d = join(ROOT, dir);
    if (existsSync(d)) for (const f of readdirSync(d)) if (/\.(viv|md)$/.test(f)) add(join(d, f));
  }
  for (const dir of ['server', 'engines', 'dft']) add(join(ROOT, dir, 'README.md'));
  // the intent lens: every .intent file of the libraries and programs
  for (const dir of ['lib', 'scf', 'posts', 'dft']) walk(join(ROOT, dir), /\.intent$/);
  walk(join(ROOT, 'design'), /\.md$/);
  // the claims: each statement, scope and test name, and the comments of their programs
  walk(join(ROOT, 'claims'), /\.(viv|py)$/);
  // the notes of the Python/Viv pairs (/compare) and the comments of their programs
  walk(join(ROOT, 'study/compare'), /\.(viv|md)$/);
  // the Jupyter kernel's README and its example notebooks
  walk(join(ROOT, 'tools/jupyter'), /\.(viv|md)$/);
  walk(join(WWW, 'src'), /\.(astro|jsx)$/);
  walk(join(WWW, 'src/lib'), /\.(js|ts)$/);
  walk(join(ROOT, 'server'), /\.mjs$/);
  // the feature catalog: its entries and the builtin table's rows and notes
  walk(join(ROOT, 'lang/viv/features'), /^(syntax|types|arrays|autodiff|correctness|lenses|tools|performance|libraries|builtins)\.txt$/);
  add(join(WWW, 'src/data/vivDocExamples.json'));
  // the scenes' and showcase pages' titles and sentences
  add(join(WWW, 'src/data/scenes.mjs'));
  add(join(WWW, 'src/data/showcases.mjs'));
  // the explorer's programs: their titles and blurbs
  add(join(WWW, 'src/data/explorables.mjs'));
  // the home page's story: read, and skipped as the owner's writing (itemsOf)
  add(join(WWW, 'src/data/story.md'));
  // the vipyr reference (/vipyr): its summaries, help and result docs, generated from the manifests
  add(join(WWW, 'src/data/vipyr.json'));
  add(join(ROOT, 'python/vipyr/README.md'));
  // the vipyr gallery: its scripts' text and the Viv files they load
  walk(join(ROOT, 'gallery'), /\.(py|viv)$/);
  // the notes for agents: how to write Viv for Python callers
  add(join(ROOT, 'AGENTS.md'));
  // the compiler's own guide for an agent (`viv help --agent`)
  add(join(ROOT, 'lang/viv/agent/guide.md'));
  add(join(WWW, 'public/llms.txt'));
  // the repository's front page and the quickstart script's comments and messages
  add(join(ROOT, 'README.md'));
  add(join(ROOT, 'quickstart.sh'));
}

let hits = 0;
let warns = 0;
const counts = {};
for (const file of files) {
  const lines = readFileSync(file, 'utf8').split('\n');
  const rel = relative(ROOT, file);
  const seen = new Set();
  for (const item of itemsOf(file)) {
    if (item.error) { console.log(`${relative(ROOT, file)}  ${item.error}`); hits++; continue; }
    const { text, line, kind, para, idents, next } = item;
    if ((lines[line - 1] || '').includes('aispeak-ok') || (lines[line - 2] || '').includes('aispeak-ok')) continue;
    for (const rule of RULES) {
      if (!(rule.kinds || WORDS).includes(kind)) continue;
      if (rule.para && !para) continue;
      if (rule.only && !rule.only.test(rel)) continue;
      const m = matchRule(rule, text, idents, next);
      if (!m) continue;
      const key = `${line}:${rule.id}:${m[0]}`;
      if (seen.has(key)) continue;
      seen.add(key);
      if (rule.gate && !strict && !rule.gate.test(rel)) {
        warns++;
        console.log(`${relative(ROOT, file)}:${line}  [${rule.id}, warning]  ${m[0].trim().slice(0, 70)}`);
        continue;
      }
      hits++;
      counts[rule.id] = (counts[rule.id] || 0) + 1;
      console.log(`${relative(ROOT, file)}:${line}  [${rule.id}]  ${m[0].trim().slice(0, 70)}`);
      console.log(`    ${rule.say}`);
    }
  }
}
const summary = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${v}`).join(', ');
const warned = warns ? ` (${warns} warning${warns > 1 ? 's' : ''} outside the gated files; --strict fails them)` : '';
console.log(hits ? `\naispeak: ${hits} in ${files.length} files — ${summary}${warned}` : `aispeak: ${RULES.length} rules, ${files.length} files, nothing flagged${warned}`);
process.exit(hits ? 1 : 0);
