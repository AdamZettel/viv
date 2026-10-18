<!-- by: user. The owner's own writing, printed on the home page as written.
     The owner chose this light edit of their 2026-10-15 text (repeats
     removed, their voice kept). Do not edit it; the aispeak lint skips it
     (scripts/aispeak.mjs). One paragraph per blank-line block; a block
     that starts with "## " is a heading the owner approved with the text. -->

I started off wondering whether agents could develop a programming language that would be useful for deploying anywhere. I wanted instant calculations, so I chose WebAssembly. I wanted portability and performance, so C. And I wanted quicker debugging, so an interpreter.

Then I noticed I wasn't writing any of the language; an agent was. So the aim became making code understandable. That led to the code atlas, a mix of callgraph, debugger and documentation that turns tangled programs into something like slideshows you can explore. There's a lot of code out there I don't understand, including plenty on this site, and I hope Viv helps you untangle some of it.

N-version programming turned out to be a big part of the language, so I designed contracts: multiple paths to the same answer. The idea is that you can look at an atlas, pick a part, swap it for another method and see what happens; that demo is next.

As a test of the language, I had it port a set of scientific codes and check them against references, which gave me an atlas of a complicated program. I'd guess many of those programs have errors that closer inspection will find, the classic problem with AI-generated output, so they need review.

Most programming language projects fail, so I hedged right away: the whole language is usable from a Python package. My language has a user base of one, and maybe a Python layer will get a few more people to try it.

The project is also a home for scientific calculations I've always wanted to understand and to teach. I hope Viv helps both of us understand them, and that this site helps people adopt some of the ideas I've had along the way.

The whole project will come with Model Context Protocol and agent tooling, so you can take it for a spin.

## Next, in 0.0.2

Some kind of GPU support, and bootstrapping the compiler. It's written in Rust and needs untangling itself, which will probably take a lot of tokens.
