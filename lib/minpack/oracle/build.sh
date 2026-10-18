#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Build the Fortran oracle from the netlib sources in lib/minpack/ref and run
# it on the netlib LMDER test data. Needs gfortran.
#
#     lib/minpack/oracle/build.sh <outdir>
#
# Writes into <outdir>:
#   netlib.txt   the netlib test driver (ref/file17.f) on its data
#                (ref/file22.dat), unmodified: the summary table MINPACK
#                ships as its own test
#   plain.txt    oracle/driver.f with the unmodified lmder.f and lmpar.f:
#                final results at 17 digits
#   trace.txt    the same driver with lmder.f and lmpar.f instrumented: the
#                same final results, plus one line per trial step
#   thurber.txt  oracle/thurber.f, the traced build on NIST's Thurber data
#                (data/Thurber.dat) from its two starting points: the post
#
# The instrumented copies are made here, by inserting lines into the netlib
# files (the awk below is the whole change): a write after the starting
# point, a record after every trial step (after label 290), and in lmpar a
# common block that hands out its iteration count. They add statements and
# change no arithmetic, and the script stops unless plain.txt and the F lines
# of trace.txt are identical, byte for byte.
#
# Flags: -O0 -ffp-contract=off, so that no multiply-add is fused. Viv's
# engines do not fuse either, so the two can agree to the last bit.
set -e
here=$(cd "$(dirname "$0")" && pwd)
ref="$here/../ref"
out=${1:?usage: build.sh <outdir>}
mkdir -p "$out"
FC=${FC:-gfortran}
FFLAGS="-O0 -ffp-contract=off -std=legacy -w"

# the problem routines of the netlib driver: everything from SUBROUTINE FCN on
sed -n '/^      SUBROUTINE FCN/,$p' "$ref/file17.f" > "$out/mgh.f"

awk '
/^      double precision dpmpar,enorm/ {
    print; print "      double precision trdin,trplm"; print "      integer lmit"
    print "      common /lmtrc/ lmit"; next }
/^      fnorm = enorm\(m,fvec\)/ {
    print; print "      write (91,901) nfev,fnorm,(x(j), j = 1, n)"; next }
/^            call lmpar\(/ { print "            trdin = delta" }
/^     \*                 wa3,wa4\)/ { print; print "            trplm = par"; next }
/^  290       continue/ {
    print
    print "            write (91,902) iter,nfev,trdin,trplm,pnorm,fnorm1,"
    print "     *         actred,prered,ratio,delta,par,"
    print "     *         merge(1,0,ratio .ge. p0001),fnorm,gnorm,lmit,"
    print "     *         (x(j), j = 1, n)"
    print "  901       format (\"S \",i6,99(1x,es25.17e3))"
    print "  902       format (\"T \",2i6,9(1x,es25.17e3),i2,"
    print "     *               2(1x,es25.17e3),i4,99(1x,es25.17e3))"
    next }
{ print }' "$ref/lmder.f" > "$out/lmder_trace.f"

awk '
/^      double precision dpmpar,enorm/ {
    print; print "      integer lmit"; print "      common /lmtrc/ lmit"; next }
/^  220 continue/ { print; print "      lmit = iter"; next }
{ print }' "$ref/lmpar.f" > "$out/lmpar_trace.f"

for f in lmder_trace lmpar_trace; do
    n=$(grep -c . "$out/$f.f")
    [ "$n" -gt 100 ] || { echo "oracle: $f.f is empty"; exit 1; }
done
grep -q 'write (91,902)' "$out/lmder_trace.f" || { echo "oracle: the trace record was not inserted"; exit 1; }
grep -q 'trplm = par' "$out/lmder_trace.f" || { echo "oracle: the par record was not inserted"; exit 1; }
grep -q 'lmit = iter' "$out/lmpar_trace.f" || { echo "oracle: the lmpar count was not inserted"; exit 1; }

common="$ref/lmder1.f $ref/qrfac.f $ref/qrsolv.f $ref/enorm.f $ref/dpmpar.f"
$FC $FFLAGS -o "$out/netlib" "$ref/file17.f" $common "$ref/lmder.f" "$ref/lmpar.f"
$FC $FFLAGS -o "$out/plain" "$here/driver.f" "$out/mgh.f" $common "$ref/lmder.f" "$ref/lmpar.f"
# the plain lmpar has no /lmtrc/; the traced build links the traced copy of each
$FC $FFLAGS -o "$out/trace" "$here/driver.f" "$out/mgh.f" $common "$out/lmder_trace.f" "$out/lmpar_trace.f"

"$out/netlib" < "$ref/file22.dat" > "$out/netlib.txt"
"$out/plain" < "$ref/file22.dat" > "$out/plain.txt"
"$out/trace" < "$ref/file22.dat" > "$out/trace.txt"
# the post's fit: NIST's Thurber data (nfev and njev are counted from the trace)
$FC $FFLAGS -o "$out/thurber" "$here/thurber.f" $common "$out/lmder_trace.f" "$out/lmpar_trace.f"
"$out/thurber" < "$here/../data/Thurber.dat" > "$out/thurber.txt"

if ! grep -v '^[ST] ' "$out/trace.txt" | cmp -s - "$out/plain.txt"; then
    echo "oracle: the instrumented build's final results differ from the plain build's"
    exit 1
fi
echo "oracle: $(grep -c '^C ' "$out/plain.txt") cases, $(grep -c '^T ' "$out/trace.txt") trial steps; instrumented = plain, byte for byte"
