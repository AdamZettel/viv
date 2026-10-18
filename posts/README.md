# posts — the feed

<https://viv-lang.org/feed> is generated from this directory. A post is one Viv
program, and posting is committing the file.

```sh
cargo build --release -p viv
posts/check.sh                                  # every post, through viv lint and BOTH engines
posts/check.sh 2026-09-21-gibbs-overshoot       # one
node www/scripts/feed-build.mjs                 # what the site build runs
```

## A post

`posts/<YYYY-MM-DD>-<slug>.viv`: lowercase, hyphens. The file is one Viv
program, and its header is a **fenced block** -- a line that is exactly `###`,
the fields, a blank line, the prose, then `###` again. The compiler skips the
block, so a line inside it needs no `#` in front and a paragraph is a
paragraph:

```viv
###
post: The Gibbs overshoot never shrinks
date: 2026-09-21
kind: article
channel: numerics
tags: fourier, analysis
equation: \max_x |S_m(x) - f(x)| \to 0.1790
summary: One to three sentences under the title. $math$, `code` and *emphasis* work.
agent: claude
lane: draft
from: the-slug-this-was-forked-from
solves: the-slug-of-a-problem-post
scale large: N=2000
recorded large: compiled Viv -> C -> cc -O2, one core of an M2 Pro, 283 s

Notes: why it is true, and where the number comes from. A blank line starts a
new paragraph. `$math$` inline, `$$math$$` displayed, `\$` a dollar sign.

## A heading, in an article
###

#! use pde/heat_1d

...the program...

figure("title", "x label", "y label")
plot(x, y, "legend")

test "the claim, as a sentence" {
    assert_near(measured, expected, 1.0e-6)
}
```

Every field is optional but `post`, `date` and `tags`. `#! use` is a compiler
directive, not header: it stays in the program, below the fence.

A fence works anywhere in a file, not only at the top, so a program can carry
prose between two of its parts. The site reads the first block as the post's
header. Every later block is a **prose cell** (Markdown: paragraphs, `##`
headings, lists, `**bold**`, *emphasis*, `code`, `$math$` and `$$display$$`),
and the code between two of them is a **code cell**: a post with prose cells
opens on its page as a notebook, each cell with its own Run and its output under
it. Running cell N runs cells 1 to N as one program, so the whole file still
runs from the command line, in both engines, exactly as written. Example:
`2026-09-24-newton-doubles-the-digits.viv`. Any program in the repository can
be read this way at `/nb/<path>` (`/nb/lib/lqr`, `/nb/posts/<slug>`).

Five kinds of post. A **demo** (the default) leads with its figure: the claim
and the run that backs it, with the notes behind a "Notes" link on the card.
An **article** (`kind: article`) shows on the feed as its title, lede,
figure and reading time, and on its own page it is laid out the way the PDE
and ladder pages are: the notes are the body, in sections, with the figure in the flow,
then what the run showed, then the program with its functions. An article must
have notes. Two marks give the body structure: a paragraph `## Title` is a
section heading, and a paragraph that is only `[figure]` is where the figure
goes (with no mark, it follows the lede).

A **ladder** (`kind: ladder`) is a program built one `rung` at a time:

```viv
rung "The slope of a secant" {
    #| Why this is the first step, then paragraphs of notes; a bare `#|` line
    #| ends a paragraph. $math$ and `code` work.

    def forward(f, x, h) { ... }
    test "the forward difference of x^3 is 3x^2 + 3xh + h^2, exactly" { ... }
}
rung "A secant centered on the point" {
    def centered(f, x, h) { ... }
    test "halving h quarters the error" { ... }
}
```

The program at rung *k* is everything before the first rung plus rungs 1
through *k*; a `def` in a later rung replaces an earlier one of the same name,
and that is an edit. `viv run` runs the top rung; `viv ladder` runs every rung
and checks the ladder's rules (every test passes on every rung; a test,
once written, stays; every rung after the first adds a test; every rung adds
something; a function a rung does not redefine keeps its content address). The
card lists the rungs; the page walks them, with each rung's code, its ledger
of new, edited and carried functions, and its tests. No figure is required.
Example: `2026-09-22-the-derivative-one-rung-at-a-time.viv`.

A **problem** (`kind: problem`) is a program whose tests do not pass yet:
stub functions, and tests that say what the right answers are. It is the one
kind published with failing tests -- and it must have at least one, since a
problem every test of which passes is solved. Its card lists the tests to
solve and leads with Solve it, which opens the composer with the program
marked as a solution. A reader can also fill in the stubs inside the card and
press Run; when every test passes the button reads Post your solution and the
composer opens with that edit. Example: `2026-09-22-solve-the-quadratic.viv`.

A **solution** (`kind: solution`, `solves: <slug>`) is a problem's program
with the functions filled in. It keeps every test of the problem word for word
(the gate compares the test blocks), may add tests of its own, and must pass
them all. Its card says what it solves. Example:
`2026-09-22-the-quadratic-formula-applied.viv`.

**Solving is not publishing.** Press Solve it on a problem's card and the
program opens there, editable; run it and, if you are signed in, your work is
kept on the server as an ATTEMPT -- yours, private, unfinished or not, and
resumable from any tab (`/api/problems/<slug>/attempt`). Posting it as a
solution is a separate choice, and the card offers it once the tests pass.
A problem shows how many people have solved it, never what they wrote.

**A problem does not hand out its answers.** The solutions posted to a problem
stay hidden on its page until you have solved it yourself, asked to see them
(one click), or wrote the problem. A solution is still an ordinary post, so the
feed can show it; only the list of solutions on the problem's page is hidden.

Every post is in one of four lanes, from least checked to most:

| lane | what it says |
|---|---|
| **AI** (`lane: ai`) | Posted by an AI model without review, or moved down here after reviews found problems. Its own tests pass; nobody else has confirmed it. You can post AI-written programs here without reviewing them first. |
| **Draft** (`lane: draft`) | Its author wrote the tests and read the output. Other people have not confirmed it yet, or reviews found problems. |
| **Reviewed** | People other than the author approved it in reviews or up-votes. From Draft, one approving review or two up-votes gets a post here. Neither requires running the program; a review shows the reviewer's test results only if they ran it. |
| **Published** | People other than the author approved it further. From Draft that takes four approving reviews or six up-votes. Later reviews and votes can move it back down. |

The author chooses one of the first two. A post with an `agent:` line starts
in AI; saying `draft` instead is a claim about the *work* -- tests written for
it, its output read and edited -- not about who typed it. The last two come
from other people's reviews and votes; the author cannot choose them.

On the site, signed-in users vote on whether a post's claim is right (one vote each, changeable)
and review it; its rating is Elo against a fixed bar. From Draft, one review that confirms its output makes a
post Reviewed and four make it Published; from AI it is two reviews to Draft, three to Reviewed and six to Published.
Downward, a published post is back to Reviewed after two reviews that find a
problem, to Draft after six and to AI after nine. An up-vote moves it about 0.6 as
far as a review (K = 40 against 64), and between the edges a post keeps its lane.

## Math

`$x^2$` in a summary, a note, a channel's syllabus or a rung's note is inline
math; `$$ ... $$` is a display equation; `\$` is a dollar sign (money, not
mathematics). Both go through KaTeX at build time,
and a formula KaTeX cannot parse fails the build, so no page ships an equation
that renders as red source.

`equation: <LaTeX>` is the one formula the post is about. It sits under the
title on the card, before the prose says the same thing in words -- the claim
in its own notation. Leave the `$` off: the line is LaTeX already.

A `#|` note above a `def` is what that function IS, and the site shows it
beside the function in a ladder's ledger and in an article's list of
functions. Math belongs there: `#| $t_2 = \frac{\ln 2}{\ln(1 + r)}$: solve
$(1+r)^t = 2$ by taking logs.` A docstring (the string on the first line of a
def) is the fallback when there is no note. It is a Viv string, whose escapes
are `\" \\ \n \t`, so a formula belongs in the `#|` note.

## The register

`npm run aispeak` reads every agent-drafted post and flags phrasing that models
produce often: titles shaped like aphorisms, grandiosity, sweeping absolutes,
"X is not Y, it is Z", portentous closers, intensifiers, two em dashes in one
sentence, and a list of three words ending a sentence. The patterns are listed,
with what to do about each, at the top of the script. It is part of
`npm run check`. A match asks the writer to look again; `aispeak-ok` on the
line, or the line above it, records that the writer looked and kept the
wording.

A post with no `agent:` line is the author's own writing and is not read at all.

## Channels

A channel is a feed with a syllabus above it: a course, or a topic. A post
joins one by naming it (`channel: algebra-2`); a post with no channel is in
the main feed only. The syllabus is `channels/<slug>.md`:

```markdown
# Algebra 2
One paragraph: the summary, shown in the feed's rail and at the top of the page.
More paragraphs: the introduction.

## Quadratics
- the-quadratic-formula-derived — a note after the slug, optional
- solve-the-quadratic
```

A syllabus lists posts of its own channel, each once (the build refuses
anything else); the channel's other posts follow it as "Latest". On the site a
signed-in reader can start a channel, edit its syllabus in place, and follow
channels; the channels they follow make up their **For you** feed, which opens
by default once they follow one. The channel page marks the problems the reader
has solved. `www/src/lib/channels.js` is the one parser.
The first course is `channels/algebra-2.md`: a ladder that derives the
quadratic formula, two problems on it, then exponentials and series.

## Modules: `#! use`

A post can be made of other programs. A line `#! use lib/cartpole` is replaced,
in place, by that module's text (its own `#! use` lines expanded in turn, its
other `#!` lines dropped), so a library of functions is used at the top and a
module that reads globals -- the DFT engine reads its deck's `ECUT` -- is used
after the lines that set them:

```viv
ECUT = 10.0
NKMESH = 2
#! use dft/kmesh
#! use dft/pw_common
#! use dft/pwdft_it
test "..." { ... }
```

Module names are repository paths without the `.viv`: `dft/pwdft_it`,
`pde/heat_1d`, `lib/cartpole`, `examples/tov` (in `lang/viv/examples`), and
`post:<slug>` for another post. The compiler resolves them from the file
system, the site from its staged programs, the server from the repository and
its database; cycles and nesting past eight are errors. `lib/` is for programs
written to be used this way -- `lib/cartpole` is the dynamics, an integrator
and the drawing of a cart-pole; a post is then the controller and the checks.

## Scales and recorded runs

A calculation that takes hours will not run in a browser, but a small version of
it will, and the two can be the same program. The program as written is the
browser scale. A header line `scale <name>: KEY=value ...` declares a bigger one by naming
the top-level constants to change; `viv run --set N=2000 post.viv` (or `viv
--set ... post.viv -o post.c` to compile it) makes that substitution, on a
top-level `NAME = <number>` line and nowhere else, so the two runs are the same
program by the content address of every function. Run it where it fits --
`viv --set N=20000 post.viv -o big.c && cc -O2 big.c -lm && ./a.out > post.large.out`
-- and put the output beside the post as `<file>.<scale>.out`, with
a header line `recorded <name>: <note>` saying how it was run. On the site, attach the
output in the composer. The card gets a switch between the browser run and the
recorded one; the recorded one is shown as the author's, not run by the server,
which checks only what it can: the output passed its tests and drew something.

`agent:` is optional: it says the post was drafted with an AI agent, and the
card says so. `from: <slug>` says this post was forked from another -- taken
to the composer with **Fork** and rewritten. The card links to the original,
and the original counts its forks. A fork drops the `agent` line: what you
publish is yours. The slug must name a post that exists.

## Lineage: `from:` and `edge:`

A post derived from another says so in its header, and says what kind of work
it was:

```
from: doubling-time
edge: rewrite
```

`edge:` is one of `fork` (the default), `fix`, `extend`, `rewrite`,
`adversarial-review` and `validate` (which also needs `reference: <where the
numbers come from>`). Each is a claim, and `feed-build.mjs` checks it by
running both programs, with the rules in `www/src/lib/lineage.js`; a false
claim fails the build the way a failing test does:

- **fix**: a test of this post, run on the parent's program, fails there and
  passes here;
- **extend**: every test of the parent is here word for word and passes, and a
  test is added;
- **rewrite**: both programs print the same thing, line for line;
- **adversarial-review**: a problem (`kind: problem`) whose added tests fail
  when run on the parent's program; its program is the reviewed one, and
  fixing it is solving it;
- **validate**: the parent's tests kept and passing, a test against the named
  reference added.

Every post page shows the history of its family: the versions as a tree, each
edge with what its check found, the lanes, the current best version, and per
version the tests, the reviews of that exact version (on the server), and the
adversarial findings still open there -- each finding is run against every
version of the family. The demo is the doubling-time family: an extension to
compounding $n$ times a year, a rewrite of it in plainer words, an adversarial
review that finds it wrong at tiny and negative rates, the fix, and a branch
comparing the rule of 72 with the rule of 70.

## What gets a post refused

`feed-build.mjs` runs every post in the interpreter a reader's tab uses, and the
build stops if a post

- fails a test, or has none (a problem is the exception: its tests fail by
  design, and at least one must)
- draws no figure (demos and articles; a ladder, problem or solution may)
- is a ladder that breaks a ladder's rule (`viv ladder` names the rung and the
  rule), or a solution that drops or edits one of its problem's tests, or
  names a problem that does not exist
- runs for more than 10 seconds
- emits more than 160 KB of figure data (the feed page carries it: plot
  fewer points, or use `plot2d` for a dense picture)
- has a `date` that disagrees with its filename, a `kind` that is not one of
  the five, a `from` that names no post, an `edge` whose claim does not hold,
  or a formula KaTeX cannot parse

`posts/check.sh` is the other half: the same programs through Viv → C → native,
with the two engines required to print the same thing line for line (a
problem's failing tests fail identically in both).

Before the engines it runs `viv lint`, which refuses an integer literal used
as a size: an argument of `zeros`, the count of `linspace`, a `for` loop's
bound, or a comparison with an index. Take the size from the data
(`len(a)`, `shape(A)`), or name it once at the top level (`N_CURVE = 200`,
with a comment saying why that number). 0, 1 and 2, literal indices,
`NAME = value` lines, array literals and a test's expected values are all
allowed.

## Figures

The card draws only what the program declares, so say it all in
`figure(title, xlabel, ylabel[, xlo, xhi, ylo, yhi[, style]])`:

- A **window** clips a line plot to the part worth looking at. On a `plot2d` it
  is the picture's physical extent, and gives it real axes instead of cell
  numbers.
- Style `"pixels"` on a `plot2d` turns smoothing off. Use it when a cell is a
  category or a count (basins, histograms); leave it off for a field.
- Give every `plot` a label: it is the legend. Several plots on the same `x`
  array share one chart.

## Writing a good one

- A summary is optional. When there is one, state a claim a reader could
  doubt, and make the tests check it.
- Test against something the program did not compute the same way: a closed form,
  a conserved quantity, a second method. Numbers from the literature are read
  from the source, never recalled.
- Tolerances are what was measured, not what makes the test pass. If the claim
  turns out false at the settings you chose, change the claim or the settings,
  and say which.
- A function is called with numbers throughout or with arrays throughout: compiled Viv has no
  generics.
- A simulation is animated with keyframes: draw its objects (`dots("atoms", X, Y, r)`,
  `circle`, `rect`, `segment`, `poly`) and call `frame(dt)` as it runs; the
  player moves each object between frames. Record a frame every few steps, not
  every step: the feed carries the figure data, and a post is refused over
  160 KB of it (a recorded run gets four times that).
