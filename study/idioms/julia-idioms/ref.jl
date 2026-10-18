# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Reference for the julia-idioms study.
#
# Part A runs the UNMODIFIED TheAlgorithms/Julia sources, loaded with `include`
# from the shallow clone (viv-wt/src/julia-idioms). Part B is idiomatic Julia
# written for this study in the style of the Julia manual and the SciML
# tutorials (in-place f!, keyword arguments, do-blocks, structs, dispatch).
#
# Every number the Viv ports check against is written to ref_values.viv as a
# Viv global, and printed here.
#
#     julia ref.jl        (about 2 s)

const SRC = joinpath(@__DIR__, "..", "..", "..", "..", "src", "julia-idioms", "src")

# ---------------------------------------------------------------- a shared RNG
# Viv's rand() is Park-Miller MINSTD; srand(s) sets state = floor(|s| mod
# (2^31 - 2)) + 1. The same stream here, so random draws agree bit for bit.
module PM
const state = Ref(0.0)
srand(s) = (state[] = floor(mod(abs(s), 2147483646.0)) + 1.0; nothing)
function rand()
    p = 16807.0 * state[]
    state[] = p - 2147483647.0 * floor(p / 2147483647.0)
    return state[] / 2147483647.0
end
end

# Each original file goes into its own module; the two that call rand() get
# PM's rand, which shadows Base.rand inside that module.
macro load(modname, files...)
    body = [:(include(joinpath($SRC, $f))) for f in files]
    esc(:(module $modname
        import ..PM
        rand() = PM.rand()
        $(body...)
    end))
end

@load ODE "math/runge_kutta_integration.jl" "math/euler_method.jl" "math/verlet.jl"
@load QUAD "math/riemann_integration.jl" "math/monte_carlo_integration.jl" "math/simpsons_integration.jl" "math/trapezoid_integration.jl"
@load LA "matrix/lu_decompose.jl" "matrix/determinant.jl" "matrix/gauss_jordan_elim.jl" "matrix/rotation_matrix.jl"
@load STATS "statistics/variance.jl" "statistics/pearson_correlation.jl" "math/average_median.jl" "math/average_mode.jl" "basic/prefix_sum.jl" "basic/difference_arr.jl"
@load NT "math/sieve_of_eratosthenes.jl" "math/collatz_sequence.jl" "math/prime_factors.jl" "math/catalan_number.jl"
@load KM "machine_learning/k_means.jl"

# ---------------------------------------------------------------- output
const OUT = IOBuffer()
fmt(x::Real) = repr(Float64(x))
fmt(v::AbstractVector) = "[" * join(fmt.(v), ", ") * "]"
fmt(A::AbstractMatrix) = "[" * join([fmt(A[i, :]) for i in axes(A, 1)], ",\n    ") * "]"
function emit(name, v)
    println(OUT, "REF_", name, " = ", fmt(v))
    println(rpad(name, 22), v)
end

# ================================================================ Part A

# 1. ODE integrators (TheAlgorithms math/)
xs, ys = ODE.runge_kutta_integration((x, y) -> y, 0, 1.0, 0.1, 1.0)
emit("RK_X", xs); emit("RK_Y", ys)
xs2, ys2 = ODE.runge_kutta_integration((x, y) -> exp(x), 0, 1.0, 0.1, 0.1)   # the docstring example
emit("RK_DOC_Y", ys2)
xe, te = ODE.euler_method((x, t) -> -x, 1.0, (0, 1))
emit("EULER_N", length(xe)); emit("EULER_LAST", xe[end]); emit("EULER_T_LAST", te[end])
xv, tv = ODE.verlet_integration(x -> -x, [1.0, 0.0], (0.0, 1.0), 1.0e-2)
emit("VERLET_N", length(xv)); emit("VERLET_X", xv[[1, 2, 50, 100]]); emit("VERLET_T_LAST", tv[end])

# 2. Quadrature
PM.srand(42.0)
emit("RIEMANN_MID", QUAD.riemann_integration(x -> x, 1, 3, 1_000, :midpoint))
emit("RIEMANN_LEFT", QUAD.riemann_integration(x -> x, 1, 3, 1_000, :left))
emit("RIEMANN_RIGHT", QUAD.riemann_integration(x -> x, 1, 3, 1_000, :right))
emit("RIEMANN_SIN", QUAD.riemann_integration(x -> sin(x), 0, pi, 1000))
emit("MONTE_CARLO", QUAD.monte_carlo_integration(x -> 3 * x^2, 0, 1, 1000))
emit("SIMPSON", QUAD.simpsons_integration(x -> sin(x), 0, pi, 100))
emit("TRAPEZOID", QUAD.trapezoid_integration(x -> sin(x), 0, pi, 100))

# 3. Matrices
A = [4.0 3 2; 2 1 3; 3 2 1]
L, U = LA.lu_decompose(A)
emit("LU_L", L); emit("LU_U", U)
emit("DET", LA.determinant(A))
emit("GJ_DOC", LA.gauss_jordan([1 2 3; 4 5 6]))            # integer method -> Float64 method
b = [25.0, 13, 16]
emit("GJ_SYS", LA.gauss_jordan([A b]))
emit("BACKSLASH", A \ b)                                   # Julia's own solve
emit("ATB", A' * b)
R = LA.rotation_matrix(pi / 6)
emit("ROT_V", R * [1.0, 0.0])
emit("ROT_ORTH", maximum(abs.(R'R - [1 0; 0 1])))

# 4. Linear regression: the whole script body of machine_learning/linear_regression.jl
# (the file runs itself on include; re-run its data here to get the numbers)
module LR
include(joinpath(Main.SRC, "machine_learning", "linear_regression.jl"))
end
emit("LR_W", LR.W[1]); emit("LR_B", LR.b); emit("LR_J_FIRST", LR.J_history[1])
emit("LR_J_LAST", LR.J_history[end]); emit("LR_PRED", vec(LR.output))

# 5. k-means on 24 points from three blobs; coordinates from the shared RNG
PM.srand(7.0)
centers = [[0.0, 0.0], [5.0, 5.0], [0.0, 6.0]]
pts = [centers[mod1(i, 3)] .+ [PM.rand() - 0.5, PM.rand() - 0.5] .* 3 for i in 1:24]
emit("KM_POINTS", permutedims(reduce(hcat, pts)))
cl = KM.KMeans.cluster(3, 0, pts)
emit("KM_CLUSTERS", permutedims(reduce(hcat, cl)))
cl2 = KM.KMeans.cluster(3, 2, pts)                          # capped at 2 iterations
emit("KM_CLUSTERS_2", permutedims(reduce(hcat, cl2)))

# 6. Statistics and basic arrays
data = [2.0, 4, 4, 4, 5, 5, 7, 9, 3, 6]
data2 = [1.0, 3, 2, 5, 4, 6, 8, 7, 9, 10]
emit("VAR", STATS.variance(data))
emit("PEARSON", STATS.pearson_correlation(data, data2))
emit("MEDIAN", STATS.median(data))
emit("MEDIAN_ODD", STATS.median(data[1:9]))
emit("MODE", STATS.mode(data))
emit("MODE_MULTI", STATS.mode([1.0, 2, 2, 3, 3, 1, 5]))
emit("PREFIX", STATS.prefix_sum(data))
da = STATS.DifferenceArray.create_diff_arr(data)
STATS.DifferenceArray.add_to_arr(da, 3, 6, 10.0)
STATS.DifferenceArray.add_to_arr(da, 1, 10, -1.0)
emit("DIFF_ARR", STATS.DifferenceArray.calculate_arr(da))

# 7. Number theory
emit("PRIMES", NT.eratosthenes(100))
cz = NT.collatz_sequence(27)
emit("COLLATZ_LEN", length(cz)); emit("COLLATZ_MAX", maximum(cz))
emit("PFACT_360", NT.prime_factors(360)); emit("PFACT_97", NT.prime_factors(97))
emit("PFACT_35", NT.prime_factors(35))                     # the original's `% i == 1` test mis-factors 35
emit("CATALAN", [NT.catalan(n) for n in 0:25])
emit("CATALAN_30", NT.catalan(30))                         # BigInt; 3814986502092304 < 2^53

# ================================================================ Part B

# 8. SciML-style ODE: an in-place right-hand side, parameters in a struct with
# defaults, a fixed-step RK4 that takes keyword arguments and a do-block
# callback, preallocated stages updated with @. .
Base.@kwdef struct LorenzParams
    σ::Float64 = 10.0
    ρ::Float64 = 28.0
    β::Float64 = 8 / 3
end

function lorenz!(du, u, p, t)
    (; σ, ρ, β) = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
    return nothing
end

function rk4!(callback, f!, u, tspan, p; dt = 0.01)
    k1, k2, k3, k4, tmp = (similar(u) for _ in 1:5)
    t, (t0, t1) = tspan[1], tspan
    nsteps = round(Int, (t1 - t0) / dt)
    for n in 1:nsteps
        f!(k1, u, p, t)
        @. tmp = u + dt / 2 * k1;  f!(k2, tmp, p, t + dt / 2)
        @. tmp = u + dt / 2 * k2;  f!(k3, tmp, p, t + dt / 2)
        @. tmp = u + dt * k3;      f!(k4, tmp, p, t + dt)
        @. u += dt / 6 * (k1 + 2k2 + 2k3 + k4)
        t = t0 + n * dt
        callback(t, u)
    end
    return u
end

u = [1.0, 0.0, 0.0]
zmax = Ref(-Inf)
samples = Float64[]
rk4!(lorenz!, u, (0.0, 2.0), LorenzParams(); dt = 1.0e-3) do t, u
    zmax[] = max(zmax[], u[3])
    isapprox(t % 0.5, 0; atol = 1e-9) && push!(samples, u[1])
end
emit("LORENZ_U", u); emit("LORENZ_ZMAX", zmax[]); emit("LORENZ_X_EVERY_HALF", samples)
u2 = [1.0, 0.0, 0.0]
rk4!((t, u) -> nothing, lorenz!, u2, (0.0, 1.0), LorenzParams(ρ = 14.0); dt = 1.0e-3)
emit("LORENZ_RHO14_U", u2)

# 9. The Julia manual's dual numbers: a struct, methods added to Base
# operators by multiple dispatch, and derivative by overloading.
struct Dual <: Number
    v::Float64
    d::Float64
end
Base.:+(a::Dual, b::Dual) = Dual(a.v + b.v, a.d + b.d)
Base.:-(a::Dual, b::Dual) = Dual(a.v - b.v, a.d - b.d)
Base.:-(a::Dual) = Dual(-a.v, -a.d)
Base.:*(a::Dual, b::Dual) = Dual(a.v * b.v, a.d * b.v + a.v * b.d)
Base.:/(a::Dual, b::Dual) = Dual(a.v / b.v, (a.d * b.v - a.v * b.d) / b.v^2)
Base.sin(a::Dual) = Dual(sin(a.v), cos(a.v) * a.d)
Base.cos(a::Dual) = Dual(cos(a.v), -sin(a.v) * a.d)
Base.exp(a::Dual) = Dual(exp(a.v), exp(a.v) * a.d)
Base.convert(::Type{Dual}, x::Real) = Dual(x, 0.0)
Base.promote_rule(::Type{Dual}, ::Type{<:Real}) = Dual
derivative(f, x) = f(Dual(x, 1.0)).d

g(x) = x^3 - 2x - 5                  # Wallis's cubic
h(x) = exp(-x) * sin(3x) / (1 + x^2)
function newton(f, x; tol = 1e-14, maxiter = 50)
    for it in 1:maxiter
        dx = f(x) / derivative(f, x)
        x -= dx
        abs(dx) < tol && return x, it
    end
    return x, maxiter
end
emit("DUAL_DG", derivative(g, 2.0)); emit("DUAL_DH", derivative(h, 0.7))
r, its = newton(g, 2.0);                 emit("NEWTON_WALLIS", r); emit("NEWTON_WALLIS_IT", its)
r2, its2 = newton(x -> cos(x) - x, 1.0); emit("NEWTON_DOTTIE", r2); emit("NEWTON_DOTTIE_IT", its2)

# Multiple dispatch over a heterogeneous vector of structs.
abstract type Shape end
struct Circle <: Shape; r::Float64; end
struct Rect <: Shape; w::Float64; h::Float64; end
struct Tri <: Shape; a::Float64; b::Float64; c::Float64; end
area(s::Circle) = π * s.r^2
area(s::Rect) = s.w * s.h
area(s::Tri) = (p = (s.a + s.b + s.c) / 2; sqrt(p * (p - s.a) * (p - s.b) * (p - s.c)))
perimeter(s::Circle) = 2π * s.r
perimeter(s::Rect) = 2(s.w + s.h)
perimeter(s::Tri) = s.a + s.b + s.c
shapes = Shape[Circle(1.0), Rect(2.0, 3.0), Tri(3.0, 4.0, 5.0), Circle(0.5)]
emit("SHAPES_AREA", sum(area, shapes)); emit("SHAPES_PERIM", sum(perimeter, shapes))
emit("SHAPES_BIGGEST", argmax(area, shapes) |> area)

# Comprehensions, generators, ranges.
emit("GEN_ODDSQ", sum(x^2 for x in 1:100 if isodd(x)))
emit("COMP_TABLE", [i * j for i in 1:3, j in 1:4])
emit("RANGE_STEP", collect(0:0.25:2))
emit("RANGE_LEN", length(0.1:0.1:0.9))

open(joinpath(@__DIR__, "ref_values.viv"), "w") do io
    println(io, "# GENERATED by ref.jl -- do not edit. Reference values, one global each.")
    print(io, String(take!(OUT)))
end
