# Physics and linear-algebra kernels (battle-corpus ph_*): CPU checks against an independent answer.
# For each kernel: (1) primal against an independent reference, (2) tangent against an independent Jacobian-vector product,
# (3) tangent against central finite differences, (4) tangent-adjoint dot-product identity, (5) adjoint gradient against
# finite differences, (6) HVP against finite differences of the gradient. Adjoint and HVP use keep_push_pop = true
# (a `while` is refused with false). Float arguments that steer a trip count (tol) are frozen: never perturbed.
# usage: julia validate_physics_kernels.jl [kernel ...]
using LinearAlgebra, Random
include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

const PH_DIR = joinpath(@__DIR__, "battle-corpus")

# ---------------------------------------------------------------- discrete operators for the references
function ph_lap2d(nint)                    # 5-point operator 4I - neighbours on the interior grid, row-major over (row, col)
    N = nint * nint; A = zeros(N, N)
    id(r, c) = (r - 1) * nint + c
    for r in 1:nint, c in 1:nint
        A[id(r, c), id(r, c)] = 4.0
        r > 1 && (A[id(r, c), id(r - 1, c)] = -1.0)
        r < nint && (A[id(r, c), id(r + 1, c)] = -1.0)
        c > 1 && (A[id(r, c), id(r, c - 1)] = -1.0)
        c < nint && (A[id(r, c), id(r, c + 1)] = -1.0)
    end
    return A
end
ph_inner(M, nint) = vec(permutedims(M[2:nint+1, 2:nint+1]))          # padded matrix -> row-major interior vector
function ph_pad(v, nint)
    M = zeros(nint + 2, nint + 2)
    M[2:nint+1, 2:nint+1] = permutedims(reshape(v, nint, nint))
    return M
end
ph_lap1d(n) = Matrix(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))

# ---------------------------------------------------------------- specs
# each spec: file, int_args, values (all float and int-array args), frozen, ref(values)->Dict out, jref(values,d)->Dict out, outs (compared args)
function ph_spec_jacobi2d()
    nint = 5; n = nint + 2
    f = zeros(n, n); f[2:nint+1, 2:nint+1] = STADE.val_randn(nint, nint)
    h2 = 0.07
    vals = Dict{Symbol,Any}(:u => zeros(n, n), :v => zeros(n, n), :f => f, :h2 => h2, :tol => 1e-26, :its => zeros(Int, 1))
    A = ph_lap2d(nint)
    ref = V -> Dict(:u => ph_pad(A \ (V[:h2] * ph_inner(V[:f], nint)), nint))
    jref = (V, D) -> begin
        fi = ph_inner(V[:f], nint); fd = ph_inner(D[:f], nint)
        Dict(:u => ph_pad(A \ (D[:h2] * fi + V[:h2] * fd), nint))
    end
    return (file = "ph_jacobi2d.jl", int_args = Dict(:nint => nint), values = vals, frozen = [:tol], ref = ref, jref = jref, outs = [:u], iters = :its, border = [:u, :v, :f])
end

function ph_spec_cg()
    nint = 5; n = nint + 2
    b = zeros(n, n); b[2:nint+1, 2:nint+1] = STADE.val_randn(nint, nint)
    vals = Dict{Symbol,Any}(:x => zeros(n, n), :b => b, :r => zeros(n, n), :p => zeros(n, n), :ap => zeros(n, n), :tol => 1e-22, :its => zeros(Int, 1))
    A = ph_lap2d(nint)
    ref = V -> Dict(:x => ph_pad(A \ ph_inner(V[:b], nint), nint))
    jref = (V, D) -> Dict(:x => ph_pad(A \ ph_inner(D[:b], nint), nint))
    return (file = "ph_cg_laplace2d.jl", int_args = Dict(:nint => nint), values = vals, frozen = [:tol], ref = ref, jref = jref, outs = [:x], iters = :its, border = [:x, :b, :r, :p, :ap])
end

function ph_spec_power()
    n = 9
    c = 2.0 .+ 2.0 .* rand(n)                              # spread diagonal: a clear gap to the second eigenvalue
    x0 = ones(n) ./ sqrt(n)
    vals = Dict{Symbol,Any}(:c => c, :x => x0, :y => zeros(n), :lam => zeros(1), :tol => 1e-26, :its => zeros(Int, 1))
    mk(c) = Matrix(SymTridiagonal(c, fill(-1.0, n - 1)))
    ref = V -> (E = eigen(Symmetric(mk(V[:c]))); Dict(:lam => [E.values[end]]))
    # Hellmann-Feynman: d lambda = sum v_i^2 dc_i
    jref = (V, D) -> (E = eigen(Symmetric(mk(V[:c]))); v = E.vectors[:, end]; Dict(:lam => [sum(v .^ 2 .* D[:c])]))
    return (file = "ph_power_iter.jl", int_args = Dict(:n => n), values = vals, frozen = [:tol], ref = ref, jref = jref, outs = [:lam], iters = :its)
end

function ph_spec_gs1d()
    n = 12
    f = STADE.val_randn(n); f[1] = 0.0; f[n] = 0.0
    h2 = 0.05
    vals = Dict{Symbol,Any}(:u => zeros(n), :f => f, :h2 => h2, :tol => 1e-24, :its => zeros(Int, 1))
    A = ph_lap1d(n - 2)
    ref = V -> (u = zeros(n); u[2:n-1] = A \ (V[:h2] .* V[:f][2:n-1]); Dict(:u => u))
    jref = (V, D) -> (u = zeros(n); u[2:n-1] = A \ (D[:h2] .* V[:f][2:n-1] .+ V[:h2] .* D[:f][2:n-1]); Dict(:u => u))
    return (file = "ph_gs1d.jl", int_args = Dict(:n => n), values = vals, frozen = [:tol], ref = ref, jref = jref, outs = [:u], iters = :its, border = [:u, :f])
end

function ph_spec_kepler()
    n = 6
    M = 2.0 .* rand(n) .+ 0.2; e = 0.1 .+ 0.7 .* rand(n)
    vals = Dict{Symbol,Any}(:mean_anom => M, :ecc => e, :ea => zeros(n), :tol => 1e-14)
    ref = V -> Dict(:ea => [begin                                   # bisection in BigFloat on g(E) = E - e sin E - M
        lo = BigFloat(0); hi = BigFloat(V[:mean_anom][i]) + 1
        for _ in 1:200; mid = (lo + hi) / 2; (mid - V[:ecc][i] * sin(mid) - V[:mean_anom][i]) > 0 ? (hi = mid) : (lo = mid); end
        Float64((lo + hi) / 2) end for i in 1:n])
    jref = (V, D) -> (E = ref(V)[:ea]; Dict(:ea => [(D[:mean_anom][i] + sin(E[i]) * D[:ecc][i]) / (1 - V[:ecc][i] * cos(E[i])) for i in 1:n]))
    return (file = "ph_kepler_newton.jl", int_args = Dict(:n => n), values = vals, frozen = [:tol], ref = ref, jref = jref, outs = [:ea], iters = nothing)
end

function ph_spec_colebrook()
    n = 6
    re = 10.0 .^ (4 .+ 3 .* rand(n)); rough = 10.0 .^ (-5 .+ 2 .* rand(n))
    vals = Dict{Symbol,Any}(:re => re, :rough => rough, :fric => zeros(n), :tol => 1e-13)
    solve(Re, r) = (x = 8.0; for _ in 1:200; x = -2 * log10(r / 3.7 + 2.51 * x / Re); end; x)
    ref = V -> Dict(:fric => [1 / solve(V[:re][i], V[:rough][i])^2 for i in 1:n])
    # implicit function theorem on g(x, Re, r) = x + 2 log10(r/3.7 + 2.51 x/Re)
    jref = (V, D) -> Dict(:fric => [begin
        Re = V[:re][i]; r = V[:rough][i]; x = solve(Re, r); w = r / 3.7 + 2.51 * x / Re; k = 2 / log(10)
        gx = 1 + k * (2.51 / Re) / w; gRe = k * (-2.51 * x / Re^2) / w; gr = k * (1 / 3.7) / w
        dx = -(gRe * D[:re][i] + gr * D[:rough][i]) / gx
        -2 * dx / x^3 end for i in 1:n])
    return (file = "ph_colebrook.jl", int_args = Dict(:n => n), values = vals, frozen = [:tol], ref = ref, jref = jref, outs = [:fric], iters = nothing)
end

function ph_spec_saha()
    n = 6
    s = 10.0 .^ (-1 .+ 2 .* rand(n)); w = 1.0 .+ 0.5 .* rand(n) .* [1, 3, 5, 7, 9, 11]   # different brackets: different step counts
    vals = Dict{Symbol,Any}(:s => s, :w => w, :xion => zeros(n), :tol => 1e-9)
    x(s) = (-s + sqrt(s * s + 4s)) / 2
    ref = V -> Dict(:xion => [x(V[:s][i]) for i in 1:n])
    jref = (V, D) -> Dict(:xion => [(1 - x(V[:s][i])) / (2 * x(V[:s][i]) + V[:s][i]) * D[:s][i] for i in 1:n])
    return (file = "ph_saha_bisect.jl", int_args = Dict(:n => n), values = vals, frozen = [:tol], ref = ref, jref = jref, outs = [:xion], iters = nothing, rtol_value = 1e-8, rtol_deriv = 1e-6)
end

function ph_spec_react()
    n = 9
    u0 = 0.5 .+ rand(n)
    vals = Dict{Symbol,Any}(:u0 => u0, :v => zeros(n), :u => zeros(n), :kap => 0.8, :dt => 0.3, :nu => 0.2, :tol => 1e-14)
    function vsol(u0i, kap, dt)
        lo = BigFloat(0); hi = BigFloat(u0i)
        for _ in 1:200; mid = (lo + hi) / 2; (mid + dt * kap * mid^3 - u0i) > 0 ? (hi = mid) : (lo = mid); end
        Float64((lo + hi) / 2)
    end
    function uref(V)
        v = [vsol(V[:u0][i], V[:kap], V[:dt]) for i in 1:n]; u = copy(v)
        for i in 2:n-1; u[i] = v[i] + V[:nu] * (v[i-1] - 2v[i] + v[i+1]); end
        u
    end
    ref = V -> Dict(:u => uref(V))
    jref = (V, D) -> begin
        v = [vsol(V[:u0][i], V[:kap], V[:dt]) for i in 1:n]
        dv = [D[:u0][i] / (1 + 3 * V[:dt] * V[:kap] * v[i]^2) for i in 1:n]
        du = copy(dv); for i in 2:n-1; du[i] = dv[i] + V[:nu] * (dv[i-1] - 2dv[i] + dv[i+1]); end
        Dict(:u => du)
    end
    return (file = "ph_react_diffuse.jl", int_args = Dict(:n => n), values = vals, frozen = [:tol, :kap, :dt, :nu], ref = ref, jref = jref, outs = [:u], iters = nothing)
end

function ph_spec_heat()
    n = 10; ns = 3; lam = 0.4
    u0 = STADE.val_randn(n)
    vals = Dict{Symbol,Any}(:u => u0, :w => zeros(n), :wn => zeros(n), :lam => lam, :tol => 1e-26, :its => zeros(Int, 1))
    function Afull(lam)
        A = zeros(n, n); A[1, 1] = 1.0; A[n, n] = 1.0
        for i in 2:n-1; A[i, i] = 1 + 2lam; A[i, i-1] = -lam; A[i, i+1] = -lam; end
        A
    end
    ref = V -> (u = copy(V[:u]); A = Afull(V[:lam]); for _ in 1:ns; u = A \ u; end; Dict(:u => u))
    jref = (V, D) -> begin                                       # forward-mode through the dense solves, no AD
        A = Afull(V[:lam]); dA = zeros(n, n)
        for i in 2:n-1; dA[i, i] = 2; dA[i, i-1] = -1; dA[i, i+1] = -1; end
        u = copy(V[:u]); du = copy(D[:u])
        for _ in 1:ns
            un = A \ u; du = A \ (du - D[:lam] * dA * un); u = un
        end
        Dict(:u => du)
    end
    return (file = "ph_heat_steps.jl", int_args = Dict(:n => n, :nsteps => ns), values = vals, frozen = [:tol], ref = ref, jref = jref, outs = [:u], iters = :its)
end

function ph_spec_mg()
    L = 4; nfine = 16; nm = nfine - 1
    h1 = 1.0 / nfine
    f = zeros(nm, L); f[:, 1] = STADE.val_randn(nm)
    vals = Dict{Symbol,Any}(:u => zeros(nm, L), :f => f, :r => zeros(nm, L), :h1 => h1, :tol => 1e-22, :its => zeros(Int, 1))
    A = ph_lap1d(nm)
    ref = V -> (u = zeros(nm, L); u[:, 1] = V[:h1]^2 .* (A \ V[:f][:, 1]); Dict(:u => u))
    # u solves A u = h1^2 f1 (levels above 1 are work space): du = A \ (2 h1 dh1 f1 + h1^2 df1)
    jref = (V, D) -> (u = zeros(nm, L); u[:, 1] = A \ (2 * V[:h1] * D[:h1] .* V[:f][:, 1] .+ V[:h1]^2 .* D[:f][:, 1]); Dict(:u => u))
    return (file = "ph_mg_solve.jl", int_args = Dict(:nfine => nfine, :num_levels => L, :nu1 => 2, :nu2 => 2), values = vals, frozen = [:tol], ref = ref, jref = jref, outs = [:u], iters = :its, level1 = true)
end

# ---------------------------------------------------------------- machinery
function ph_zero_border!(A)               # Dirichlet data are not perturbed
    if A isa AbstractMatrix
        A[1, :] .= 0.0; A[end, :] .= 0.0; A[:, 1] .= 0.0; A[:, end] .= 0.0
    else
        A[1] = 0.0; A[end] = 0.0
    end
    return A
end
ph_proj(spec, kernel, I, V, d) = hasproperty(spec, :border) ?
    (D = STADE.val_unflatten(kernel, I, V, d); for a in spec.border; ph_zero_border!(D[a]); end; STADE.val_flatten(kernel, D)) : d
ph_lv(spec, A) = hasproperty(spec, :level1) ? A[:, 1] : A
ph_relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1e-12)

function ph_masks(kernel, spec)
    m = Float64[]
    for a in STADE.val_float_arg_order(kernel.sig)
        v = spec.values[a]
        append!(m, fill(a in spec.frozen ? 0.0 : 1.0, v isa Number ? 1 : length(v)))
    end
    return m
end

function ph_check(spec; verbose = true)
    Random.seed!(20261110)
    path = joinpath(PH_DIR, spec.file)
    expr = STADE.io_read_corpus_entry(path)
    kernel = STADE.parse_kernel(expr)
    sig = kernel.sig
    I = spec.int_args; V = spec.values
    mask = ph_masks(kernel, spec)
    # Only the declared outputs are compared with finite differences: a work array or an eigenvector that is converged to 1e-8 changes
    # by that much when the trip count changes by one, which a step of 1e-6 turns into a large false difference.
    omask = Bool[]
    for a in STADE.val_float_arg_order(sig)
        o = fill(a in spec.outs, length(V[a]))
        hasproperty(spec, :level1) && a in spec.outs && (o[size(V[a], 1) + 1:end] .= false)    # column 1 only: the other columns are work space
        append!(omask, o)
    end
    res = Dict{Symbol,Any}()
    # (1) primal
    pf = STADE.val_compile(expr)
    args = STADE.val_primal_call_args(sig, I, V)
    Base.invokelatest(pf, args...)
    outv = Dict(a => v for (a, v) in zip(sig.args, args))
    refv = spec.ref(V)
    res[:primal] = maximum(ph_relerr(ph_lv(spec, outv[a]), ph_lv(spec, refv[a])) for a in spec.outs)
    if spec.iters !== nothing; res[:iters] = Int(outv[spec.iters][1]); end
    obs = STADE.val_compile(STADE.val_primal_observing_expr(kernel, expr))
    f_eval = x -> STADE.val_call_primal_observed(obs, kernel, I, STADE.val_unflatten(kernel, I, V, x))
    x0 = STADE.val_flatten(kernel, V)
    # (2) tangent vs independent Jacobian-vector product, (3) tangent vs central FD
    tan = STADE.val_compile(STADE.stade_tangent(expr))
    function jvp(x, d)
        vals = STADE.val_unflatten(kernel, I, V, x); dv = STADE.val_unflatten(kernel, I, STADE.val_zeros_like(kernel, V), d)
        STADE.val_call_tangent(tan, kernel, I, vals, dv)
    end
    e2 = 0.0; e3 = 0.0
    for t in 1:3
        d = ph_proj(spec, kernel, I, V, STADE.val_randn(length(x0)) .* mask)
        D = STADE.val_unflatten(kernel, I, STADE.val_zeros_like(kernel, V), d)
        yd = STADE.val_unflatten(kernel, I, V, jvp(x0, d))
        jr = spec.jref(V, D)
        e2 = max(e2, maximum(ph_relerr(ph_lv(spec, yd[a]), ph_lv(spec, jr[a])) for a in spec.outs))
        h = 1e-6
        fd = (f_eval(x0 .+ h .* d) .- f_eval(x0 .- h .* d)) ./ (2h)
        e3 = max(e3, ph_relerr(jvp(x0, d)[omask], fd[omask]))
    end
    res[:tan_ref] = e2; res[:tan_fd] = e3
    # (4)-(6) adjoint and HVP, stack mode
    ad = STADE.stade_adjoint(expr; keep_push_pop = true)
    adf = STADE.val_compile(ad.adjoint); isf = STADE.val_compile(ad.initstacks)
    hv = STADE.stade_hvp(expr; keep_push_pop = true)
    hvf = STADE.val_compile(hv.hvp)
    seed = STADE.val_random_values_like(kernel, V)
    for a in keys(seed); a in spec.outs || (seed[a] = seed[a] isa Number ? 0.0 : zeros(size(seed[a]))); end
    hasproperty(spec, :level1) && for a in spec.outs; seed[a][:, 2:end] .= 0.0; end
    seedf = STADE.val_flatten(kernel, seed)
    grad(x) = STADE.val_call_adjoint(adf, isf, kernel, I, STADE.val_unflatten(kernel, I, V, x), seed)
    e4 = 0.0
    for t in 1:3
        xd = ph_proj(spec, kernel, I, V, STADE.val_randn(length(x0)) .* mask)
        yb = STADE.val_random_values_like(kernel, V)
        ydv = jvp(x0, xd)
        xb = STADE.val_call_adjoint(adf, isf, kernel, I, STADE.val_unflatten(kernel, I, V, x0), yb)
        lhs = sum(STADE.val_flatten(kernel, yb) .* ydv); rhs = sum(xb .* xd)
        e4 = max(e4, abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0))
    end
    res[:dotprod] = e4
    e5 = 0.0; e6 = 0.0
    g0 = grad(x0)
    for t in 1:3
        d = ph_proj(spec, kernel, I, V, STADE.val_randn(length(x0)) .* mask)
        h = 1e-6
        fd = (sum(f_eval(x0 .+ h .* d) .* seedf) - sum(f_eval(x0 .- h .* d) .* seedf)) / (2h)
        e5 = max(e5, abs(fd - sum(g0 .* d)) / max(abs(fd), 1e-12))
        gfd = (grad(x0 .+ h .* d) .- grad(x0 .- h .* d)) ./ (2h)
        hvv = STADE.val_call_hv(hvf, isf, kernel, I, STADE.val_unflatten(kernel, I, V, x0), seed,
                                STADE.val_unflatten(kernel, I, STADE.val_zeros_like(kernel, V), d), STADE.val_zeros_like(kernel, V))
        e6 = max(e6, maximum(abs.((gfd .- hvv) .* mask)) / max(maximum(abs.(gfd .* mask)), 1e-12))
    end
    res[:grad_fd] = e5; res[:hvp_fd] = e6
    return res
end

ph_all() = [ph_spec_mg, ph_spec_jacobi2d, ph_spec_cg, ph_spec_power, ph_spec_gs1d, ph_spec_kepler, ph_spec_colebrook, ph_spec_saha, ph_spec_react, ph_spec_heat]

if abspath(PROGRAM_FILE) == @__FILE__
    want = Set(ARGS)
    for mk in ph_all()
        spec = mk()
        (isempty(want) || replace(spec.file, ".jl" => "") in want) || continue
        r = try ph_check(spec) catch e; Dict(:error => first(sprint(showerror, e, catch_backtrace()), 600)); end
        println(rpad(spec.file, 22), join(["$k=$(v isa Float64 ? round(v, sigdigits = 3) : v)" for (k, v) in sort(collect(r); by = first)], "  "))
    end
end
