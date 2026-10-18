// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// claimstest -- the claims extractor and linker held to examples.
//
//     node scripts/claimstest.mjs          (npm run claimstest)
//
// Unit tests of scripts/claims/*.mjs: sentence ids, sentence splitting, number
// parsing (units, percentages, scientific notation, powers of ten, fractions,
// what is not a claim), precision matching, the program's tests and printed
// values, and the linker's statuses. Then the two cases the linker must catch,
// replayed from git history when viv.wasm is built:
//   - the transfer-matrix post once said T + R = 1 "to 10^-15" while the run
//     gave 3.1e-14 (fixed in 2d62985): the old version must be a mismatch;
//   - the Ising post said |m| ~ 0.14 until a random-state fix made the run
//     give 0.1466 and the prose became 0.15 (65b74eb): the new program with
//     the old sentence must be a mismatch, and the old program with it not.
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { sentenceId, splitSentences, numbersIn, literal, classify, citationsIn, paragraphs, texEval } from './claims/text.mjs';
import { printed, evalConst, assertsOf, parseRun, testsOf, constantsOf, literalsIn } from './claims/program.mjs';
import { matches, candidates, linkPost, associatedTest } from './claims/link.mjs';
import { buildPost } from './claims-build.mjs';
import { runner } from './claims/run.mjs';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
let failed = 0;
let passed = 0;
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
function check(name, ok, detail = '') {
  if (ok) passed++;
  else { failed++; console.log(`FAIL ${name}${detail ? `\n     ${detail}` : ''}`); }
}
const close = (a, b, r = 1e-12) => Math.abs(a - b) <= r * Math.max(1, Math.abs(b));

// ---- ids ----
check('an id is 12 hex digits', /^[0-9a-f]{12}$/.test(sentenceId('The run gives 0.95669.')));
check('whitespace and case do not move an id', sentenceId('The  run\ngives $0.95669$.') === sentenceId('the run gives $0.95669$.'));
check('changing a number moves the id (the verdict goes stale)', sentenceId('The run gives $0.95669$.') !== sentenceId('The run gives $0.95668$.'));
check('the id of a fixed sentence never changes', sentenceId('They add to 1 at every energy, to within $10^{-13}$.') === 'c5a9b88f8a15', sentenceId('They add to 1 at every energy, to within $10^{-13}$.'));

// ---- sentences ----
const ss = (t) => splitSentences(t).map((s) => s.text);
check('initials do not end a sentence', eq(ss('Lars Onsager solved it in 1944, and C. N. Yang derived it in 1952. Next.'), ['Lars Onsager solved it in 1944, and C. N. Yang derived it in 1952.', 'Next.']));
check('abbreviations do not end a sentence', ss('See Phys. Rev. Lett. 81, 109 for it. And e.g. this.').length === 2, JSON.stringify(ss('See Phys. Rev. Lett. 81, 109 for it. And e.g. this.')));
check('a period inside math does not end a sentence', ss('It is $x = 1. Y$ here. Then more.').length === 2);
check('a decimal does not end a sentence', ss('It is 0.5 wide. Then 3.2 more.').length === 2);
check('a question ends a sentence', ss('Is it 1? Yes, it is.').length === 2);
const paras = paragraphs('## Heading\n\nOne line\nand the next.\n\n[figure]\n\n- an item\n- another\n\n$$x = 1$$\n', 10);
check('paragraphs: headings set the section, [figure] is skipped, list items stand alone', paras.length === 4 && paras[0].section === 'Heading' && paras[0].line === 12 && paras[1].text === 'an item', JSON.stringify(paras.map((p) => [p.text, p.line, p.section])));

// ---- numbers ----
const nums = (t) => numbersIn(t).numbers;
const one = (t) => nums(t)[0] || {};
check('decimal: value and the place of its last digit', close(one('It is 0.577 wide.').value, 0.577) && close(one('It is 0.577 wide.').ulp, 0.001));
check('scientific e-notation', close(one('the run gave 3.1e-14 there').value, 3.1e-14) && close(one('the run gave 3.1e-14 there').ulp, 1e-15));
check('mantissa times ten to a power, in math', close(one('it sits $1.1 \\times 10^{-5}$ below').value, 1.1e-5) && close(one('it sits $1.1 \\times 10^{-5}$ below').ulp, 1e-6));
check('a bare power of ten, in math', close(one('to $10^{-15}$.').value, 1e-15) && one('to $10^{-15}$.').ulp === 0);
check('a power of ten in Unicode superscripts', close(nums('T + R = 1 to 10⁻¹⁵ at every energy')[1].value, 1e-15), JSON.stringify(nums('T + R = 1 to 10⁻¹⁵ at every energy')));
check('mantissa times 10 in Unicode', close(one('an error of 3.1 × 10⁻¹⁴ here').value, 3.1e-14), JSON.stringify(nums('an error of 3.1 × 10⁻¹⁴ here')));
check('a percentage', one('It overshoots by 8.95% of the jump.').percent === true && close(one('It overshoots by 8.95% of the jump.').value, 8.95));
check('a percentage in math', one('which is $8.95\\%$ of the jump').percent === true);
check('a unit after the number', one('it took 38 seconds natively').unit === 'seconds' && one('an energy of 13.6 eV').unit === 'eV');
check('thousands separators', one('a factor of 57,000').value === 57000);
check('trailing zeros of a large integer are not significant, of a small one they are', one('a factor of 57,000').ulp === 1000 && one('chains of 20000 sweeps').ulp === 10000 && one('over 1200 sweeps').ulp === 1 && one('20 batches').ulp === 1);
check('a fraction in math', close(one('a slope of $3/2$ exactly').value, 1.5) && close(one('the ratio $\\tfrac12$ here').value, 0.5));
check('a symbol from the math it is assigned in', one('at $T = 1.8$ it settles').symbol === 'T' && close(one('at $T = 1.8$ it settles').value, 1.8));
check('an uncertainty after \\pm', eq(nums('an average of $0.9567 \\pm 0.0013$.').map((n) => n.role), ['value', 'uncertainty']));
check('a number inside a formula is not a claim', eq(nums('The formula $2/\\ln(1 + \\sqrt2) = 2.26919$ is it.').map((n) => n.raw), ['2.26919']));
check('subscripts and exponents are not claims', nums('the term $x_1^2 + k_3$ here').length === 0);
check('a year is a year', one('Onsager solved it in 1944.').role === 'year');
check('a date is not a number', nums('recorded 2026-09-22 on a laptop').length === 0);
check('a version is not a number', nums('it matches PySCF 2.5.0 within bounds').length === 0);
check('a numbered thing is a label', one('the check in rung 3 passes').role === 'label');
check('names in code spans are symbol hints, and their digits are not claims', eq(numbersIn('the grid has $|m| \\approx 0.14$ (`m_T`) and `N_1 = 3`').names, ['m_T', 'N_1']) && nums('the grid has $|m| \\approx 0.14$ (`m_T`) and `N_1 = 3`').length === 1);
check('"to" before a small number is a bound', one('They add to 1 at every energy, to $10^{-15}$.') && nums('They add to 1 at every energy, to $10^{-15}$.')[1].qualifier === 'upper');
check('"to" in a range is not a bound', nums('from 0.05 to 0.001 in steps').every((n) => n.qualifier !== 'upper'));
check('"about" makes it approximate', one('about 0.36 of it').qualifier === 'approx' && one('$|m| \\approx 0.15$').qualifier === 'approx');
check('"under" and "more than" are bounds', one('under 1% of it').qualifier === 'upper' && one('more than 0.999 of it').qualifier === 'lower');

// what is not a claim, found in the orphan triage (see the orphans branch)
check('a decade is not a number: "the 1960s" is not 196', nums('RANDU was IBM\'s generator in the 1960s: slow.').length === 0, JSON.stringify(nums('RANDU was IBM\'s generator in the 1960s: slow.')));
check('a basis-set or method name is not a number: 6-31G, STO-3G, Teter-93', nums('water in 6-31G and STO-3G, with the Teter-93 fit').length === 0, JSON.stringify(nums('water in 6-31G and STO-3G, with the Teter-93 fit')));
check('a hyphenated setup number still is one: a 4000-walker estimate, a degree-11 fit, a 5-10 range', eq(nums('a 4000-walker estimate and a degree-11 fit over 5-10 steps').map((n) => n.value), [4000, 11, 5, 10]), JSON.stringify(nums('a 4000-walker estimate and a degree-11 fit over 5-10 steps').map((n) => n.value)));
check('a digit run is never cut: "31G" does not give 3', nums('the 31G tables').length === 0);
check('an equation range is a label: Eqs. (4.9)-(4.12)', nums('Machleidt and Entem, Eqs. (4.9)-(4.12), with it').length === 0, JSON.stringify(nums('Machleidt and Entem, Eqs. (4.9)-(4.12), with it')));
check('"Phys. Rep." does not end a sentence, and its volume and page are not claims', ss('(Machleidt and Entem, Phys. Rep. 503, 1 (2011), with m = 1.) Next.').length === 2 && nums('Machleidt and Entem, Phys. Rep. 503, 1 (2011), with it').length === 0);
check('an ApJ volume, page and year are not claims, and they are a citation', nums('the two at the top (Lindblom, ApJ 398, 569, 1992).').every((n) => n.role === 'year') && citationsIn('the two at the top (Lindblom, ApJ 398, 569, 1992).').some((c) => c.text === 'ApJ 398, 569, 1992'), JSON.stringify([nums('the two at the top (Lindblom, ApJ 398, 569, 1992).'), citationsIn('the two at the top (Lindblom, ApJ 398, 569, 1992).')]));
check('a negative number and a hyphenated range still parse', eq(nums('it lies between -5 and 9 here').map((n) => n.value), [-5, 9]) && eq(nums('from 5 to 160 steps').map((n) => n.value), [5, 160]));
check('an expression set equal to a number is the arithmetic, not a claim', eq(nums('not $100/7 = 14.3$ years').map((n) => [n.raw, n.role]), [['100/7', 'expression'], ['14.3', 'value']]));
check('an expression that is itself the value of a symbol stays a claim: $\\varepsilon = 2^{-52} \\approx 2.2 \\times 10^{-16}$', nums('so $\\varepsilon = 2^{-52} \\approx 2.2 \\times 10^{-16}$ here').map((n) => n.role).join() === 'value,value', JSON.stringify(nums('so $\\varepsilon = 2^{-52} \\approx 2.2 \\times 10^{-16}$ here')));
check('a fraction on its own is still a claim', one('a slope of $3/2$ exactly').role === 'value');

// ---- arithmetic the prose writes out ----
check('texEval: logs, fractions, powers, pi, implicit products', close(texEval('\\ln 2 / \\ln 1.07'), Math.log(2) / Math.log(1.07)) && close(texEval('\\pi^2/6'), Math.PI ** 2 / 6) && close(texEval('0.4^2/12'), 0.16 / 12) && close(texEval('2\\pi'), 2 * Math.PI) && close(texEval('(3.2 \\times 10^{-5})^2'), 1.024e-9) && close(texEval('\\tfrac12'), 0.5) && close(texEval('2/\\ln(1 + \\sqrt2)'), 2.269185314213022));
check('texEval: a letter standing for a quantity has no value', texEval('x^2') === null && texEval('T') === null && texEval('\\tfrac12(11|11)') === null && texEval('\\gamma_{14}') === null);

// ---- the program's literals ----
check('a quotient of two literals is given too: 1.0 / 24.0 gives 1/24', literalsIn('h = 1.0 / 24.0').some((v) => close(v, 1 / 24)) && literalsIn('h = 1.0 / 24.0').includes(24));
check('literals inside a call are not quotients: f(1.0) / 2.0 gives 1 and 2 only', eq(literalsIn('y = f(1.0) / 2.0'), [1, 2]));
const tl = testsOf('test "the root at k = 0.3" {\n    expected = 0.0126\n    r = root(0.3)\n    assert_near(r, expected, 1.0e-4)\n}\n', {});
check('a test\'s own constant is its expected value, and the literals it passes are the test\'s', tl[0].asserts[0].expectedValue === 0.0126 && tl[0].literals.includes(0.3), JSON.stringify(tl));

// ---- kinds ----
check('a question is rhetorical', classify('Before running: is $T$ equal to 1?', []) === 'rhetorical');
check('a sentence with a number is numeric', classify('The run gives 0.95669 here.', nums('The run gives 0.95669 here.')) === 'numeric');
check('a defined term is a definition', classify('One pass over all the spins is a *sweep*.', []) === 'definition');
check('the program doing something is procedural', classify('The cell compares the matrix product with this formula.', []) === 'procedural');
check('the rest is factual', classify('Leapfrog is symplectic, so its energy error stays bounded.', []) === 'factual');

// ---- citations ----
const cites = citationsIn('Lars Onsager solved it, see Takahasi and Mori (1974), (Landau, 1946), Rev. Mod. Phys. 81, 109 (2009), CODATA 2018 and NIST data.').map((c) => c.text);
check('citations: author-year, parenthetical, journal, dataset', ['Takahasi and Mori (1974)', 'Landau (1946)', 'Rev. Mod. Phys. 81, 109 (2009)', 'NIST'].every((c) => cites.includes(c)), JSON.stringify(cites));

// ---- precision matching ----
const lit = (s) => ({ ...literal(s), percent: false });
check('0.577 matches 0.5767095 (rounds to it at 3 decimals)', matches(lit('0.577'), { value: 0.5767095, ulp: 0 }));
check('0.577 does not match 0.5776', !matches(lit('0.577'), { value: 0.5776, ulp: 0 }));
check('0.14 does not match 0.146563 (it rounds to 0.15)', !matches(lit('0.14'), { value: 0.146563, ulp: 1e-6 }));
check('0.15 matches 0.146563', matches(lit('0.15'), { value: 0.146563, ulp: 1e-6 }));
check('more digits than printed: 0.21079395 matches a shown 0.210794', matches(lit('0.21079395'), { value: 0.210794, ulp: 1e-6 }));
check('a percentage matches its fraction', matches({ ...literal('6.33'), percent: true }, { value: 0.063297625, ulp: 0 }));
check('an exact power of ten matches only itself', matches({ ...literal('1'), value: 1e-13, ulp: 0 }, { value: 1e-13, ulp: 0 }) && !matches({ value: 1e-13, ulp: 0 }, { value: 1.2e-13, ulp: 0 }));

// ---- the program ----
check('printed %g values carry six significant digits', close(printed('1').ulp, 1e-5) && close(printed('0.956686').ulp, 1e-6) && close(printed('3.10862e-14').ulp, 1e-19));
check('printed with fixed decimals (a reference script) keeps its last digit', close(printed('1.29', false).ulp, 0.01));
check('bit patterns are labels, not claims: "count from 000 to 111"', (() => {
  const b = numbersIn('its three inputs count from 000 to 111, and the sum 11 + 7 is 18.').numbers;
  const r = (raw) => (b.find((n) => n.raw === raw) || {}).role;
  // 11 stays a number: no zero-padded pattern of width 2 beside it
  return r('000') === 'label' && r('111') === 'label' && r('18') !== 'label' && r('11') !== 'label';
})());
check('a small integer that is an operand of a formula in text is not a claim; "= 1", "(3, 2)" and "5-point" are', (() => {
  const role = (s, raw) => (numbersIn(s).numbers.find((n) => n.raw === raw) || {}).role;
  return role('the slab is -v0 / (1 + exp(x))', '1') === 'expression' && role('it exceeds 2 pi by dt^2/12', '2') === 'expression'
    && role('the first count + 1 entries', '1') === 'expression' && role('with 1 < omega < 2', '2') === 'expression'
    && role('above the top T = 1 exactly', '1') === 'count' && role('they cross at (3, 2)', '3') === 'count'
    && role('with the 5-point rule', '5') === 'count';
})());
check('a reference\'s "1e-11" (repr or %g: no decimal point) is good to six digits, so it establishes the prose\'s exact 10^{-11}',
  close(printed('1e-11', false).ulp, 1e-16) && close(printed('1.0e-11', false).ulp, 1e-12)
  && matches(numbersIn('scipy at a tolerance of $10^{-11}$.').numbers[0], { value: 1e-11, ...printed('1e-11', false) }));
check('constant expressions', close(evalConst('2.0 / log(1.0 + sqrt(2.0))'), 2.269185314213022) && close(evalConst('0.15 * X', { X: 56.21 }), 8.4315) && evalConst('m_T[3]') === null);
const as = assertsOf('assert_near(horizon, 18.4 / L, 0.15 * 56.21)\n assert_lt(0.999, back)');
check('assertions parse', as.length === 2 && as[0].fn === 'assert_near' && as[0].args[2] === '0.15 * 56.21' && as[1].args[0] === '0.999');
const run = parseRun('m_T: [0.979375, 0.146563]\nTC: 2.26919\nPASS the test\nFAIL other -- why\nplot: a ; [1, 2]\nT, mean: 1.6 0.9797');
check('a run: shown scalars and arrays, printed numbers, pass and fail', run.shown.length === 3 && run.shown[1].index === 1 && run.printed.length === 2 && eq(run.pass, ['the test']) && eq(run.fail, ['other -- why']));

// ---- the linker ----
function linkOne(src, out) {
  const rec = buildPost({ file: '2026-01-01-synthetic.viv', src, run: { out, cells: { 1: out } }, refsDir: '/nonexistent' });
  return rec.numbers;
}
const SYN = `###
post: Synthetic
date: 2026-01-01
tags: test

The error is under $10^{-15}$ (\`err\`). The run gives $x = 0.577$ (\`x\`), and $y = 0.40$ (\`y\`).
A value 7.25 nothing computes. Takahasi and Mori (1974) found 4.4 for it. The limit is 12.5.
###

LIMIT = 12.5
err = 3.1e-14
x = 0.5767095
y = 0.4501
show err
show x
show y
test "x is 0.577" {
    assert_near(x, 0.5767095, 1.0e-12)
}
test "the error is under 1e-13" {
    assert_lt(err, 1.0e-13)
}
`;
const OUT = 'err: 3.1e-14\nx: 0.57671\ny: 0.4501\nPASS x is 0.577\nPASS the error is under 1e-13\n';
const ln = linkOne(SYN, OUT).filter((n) => n.origin !== 'test');
const st = (raw) => ln.find((n) => n.raw === raw)?.status;
check('a bound the named value breaks is a mismatch', st('10^{-15}') === 'mismatch', JSON.stringify(ln.map((n) => [n.raw, n.status, n.note]).slice(0, 2)));
check('a number a passing test expects is checked', st('0.577') === 'checked');
check('a named value that does not round to the prose is a mismatch', st('0.40') === 'mismatch');
check('a number nothing establishes is an orphan', st('7.25') === 'orphan');
check('an unmatched number in a sentence with a citation is cited', st('4.4') === 'cited');
check('a number the program is given is an input', st('12.5') === 'input');
{
  // a count is an exact integer: the test's 0.99919 and its tolerance 1e-5 do
  // not establish the 1 and the 0 in the prose, an exact 3 does
  const SRC = `###
post: Synthetic counts
date: 2026-01-01
tags: test

The rate is $(\\kappa - 1)/(\\kappa + 1)$ and the step is 1 or 0 here, and there are 3 roots. The speed is $c > 0$.
###

rate = 0.99919
show rate
test "the rate is 0.99919" {
    assert_near(rate, 0.99919, 1.0e-5)
}
test "3 roots" {
    assert_near(3.0, 3.0, 0.0)
}
`;
  {
    const SRC2 = `###
post: Synthetic literal
date: 2026-01-01
tags: test

The units have q = m = 1 throughout.
###

def f(x) {
    return x * x
}
test "f squares its argument" {
    assert_near(f(1.0), 1.0, 0.0)
}
`;
    const qm = linkOne(SRC2, 'PASS f squares its argument\n').filter((n) => n.origin !== 'test' && n.raw === '1');
    check('a literal 1 in an unrelated test makes the prose\'s "q = m = 1" an input, not checked', qm.length === 1 && qm[0].status === 'input', JSON.stringify(qm.map((n) => [n.status, n.link])));
  }
  const cn = linkOne(SRC, 'rate: 0.99919\nPASS the rate is 0.99919\nPASS 3 roots\n').filter((n) => n.origin !== 'test');
  const at = (raw) => cn.filter((n) => n.raw === raw).map((n) => n.link?.value);
  check('a count matches only its exact integer, not a nearby value or a small tolerance',
    !at('1').includes(0.99919) && !at('0').some((v) => v !== undefined && v !== 0) && cn.filter((n) => n.raw === '0').length === 2 && cn.find((n) => n.raw === '3')?.status === 'checked',
    JSON.stringify(cn.map((n) => [n.raw, n.status, n.link?.value])));
}
// the arithmetic check, and a test's body literals
const AR = `###
post: Arithmetic
date: 2026-01-01
tags: test

Not $100/7 = 14.3$ years, and $\\ln 2 \\approx 0.693$, but never $1.5^2 = 2.5$. At $k = 0.3$ the root is right.
###

test "the root" {
    r = 0.3 * 2.0
    assert_near(r, 0.6, 1.0e-12)
}
`;
const ar = linkOne(AR, 'PASS the root\n').filter((n) => n.origin !== 'test');
const ars = (raw) => ar.find((n) => n.raw === raw)?.status;
check('arithmetic the sentence writes out is checked by evaluating it', ars('14.3') === 'checked' && ars('0.693') === 'checked' && ar.find((n) => n.raw === '100/7')?.status === 'skip', JSON.stringify(ar.map((n) => [n.raw, n.status])));
check('arithmetic that does not round is a mismatch', ars('2.5') === 'mismatch', JSON.stringify(ar.map((n) => [n.raw, n.status, n.note])));
check('a literal a passing test passes to the code is checked', ars('0.3') === 'checked', JSON.stringify(ar.map((n) => [n.raw, n.status])));
check('a test name is linked to its own test', associatedTest('the error is under 1e-13', [{ name: 'the error is under 1e-13', asserts: [] }, { name: 'x is 0.577', asserts: [] }])?.name === 'the error is under 1e-13');

// ---- the two cases from history ----
const wasm = existsSync(join(WWW, 'public/viv/viv.wasm'));
if (!wasm) console.log('skip: the history cases need www/public/viv/viv.wasm (sh www/build-wasm.sh)');
else {
  const r = runner(WWW);
  const git = (rev, path) => execFileSync('git', ['show', `${rev}:${path}`], { cwd: ROOT, encoding: 'utf8', maxBuffer: 1 << 26 });
  const TM = '2026-09-25-transfer-matrices-and-resonant-tunneling.viv';
  const tmOld = git('2d62985^', `posts/${TM}`);
  const tmRec = buildPost({ file: TM, src: tmOld, run: await r.interp(TM, tmOld) });
  const tmN = tmRec.numbers.find((n) => n.origin === 'summary' && n.raw === '10^{-15}');
  check('history: the transfer-matrix summary\'s "to 10^-15" before 2d62985 is a mismatch against the run\'s 3.1e-14', tmN?.status === 'mismatch' && /3\.1\d*e-14/.test(tmN.note), JSON.stringify(tmN));
  const tmNew = git('2d62985', `posts/${TM}`);
  const tmNewN = buildPost({ file: TM, src: tmNew, run: await r.interp(TM, tmNew) }).numbers.find((n) => n.origin === 'summary' && n.raw === '10^{-13}');
  check('history: after 2d62985 its "to within 10^-13" is checked', tmNewN?.status === 'checked', JSON.stringify(tmNewN));
  const IS = '2026-09-25-metropolis-ising-against-onsager.viv';
  const isOld = git('65b74eb^', `posts/${IS}`);
  const isNew = git('65b74eb', `posts/${IS}`);
  const hybrid = isNew.replace('|m| \\approx 0.15$', '|m| \\approx 0.14$');
  check('history: the hybrid really has the old sentence', hybrid !== isNew);
  const at = async (src, raw) => buildPost({ file: IS, src, run: await r.interp(IS, src) }).numbers.find((n) => n.raw === raw && /approx/.test(n.context));
  const a = await at(isOld, '0.14');
  const b = await at(hybrid, '0.14');
  const c = await at(isNew, '0.15');
  check('history: before 65b74eb, "|m| ~ 0.14" matches the run', a && a.status !== 'mismatch', JSON.stringify(a));
  check('history: 65b74eb\'s program with the old "|m| ~ 0.14" is a mismatch (m_T[9] = 0.146563)', b?.status === 'mismatch' && /0\.146563/.test(b.note), JSON.stringify(b));
  check('history: 65b74eb\'s "|m| ~ 0.15" is checked', c?.status === 'checked' || c?.status === 'computed', JSON.stringify(c));
}

console.log(`claimstest: ${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
