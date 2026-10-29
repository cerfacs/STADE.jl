isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random
# the shim is loaded at the top level so that the functions below are defined in a later world than it (a script run would otherwise hit a world-age error)
isdefined(Main, :CUDA) || include(joinpath(@__DIR__, "ownership_shim.jl"))

# Thread coarsening with merged accumulations (cgen_coarsen_*, `coarsen = 2`).
#
# A coarsened kernel is a second kernel (`..._c!`) emitted beside the plain one; the host launches it only when the number of threads that add to one
# address reaches `coarsen_min`. In it a thread runs two iterations of the offloaded loop at once and sums, in registers, the atomic adds the two
# iterations would issue to one address before issuing a single add. No GPU is needed for the properties checked here:
#   1. UNITS      hand-written kernels. Which ones get a variant, how many sites it merges, that the plain kernels are byte-identical to `coarsen = 1` whatever
#                 the variant does, that the host selects the variant on the right count, and that every refusal guard of `cgen_coarsen_body` is reached by a
#                 case that only it refuses.
#   2. EXECUTION  each unit kernel, forced onto its variant (coarsen_min = 0) at K = 2, 3 and 4, runs on the CPU shim of ownership_shim.jl in ascending and
#                 shuffled thread order over trip counts that are not multiples of 32K (and a zero-trip one); the result must equal the plain kernel to
#                 rounding and no element may be flagged. A huge `coarsen_min` must launch only plain kernels, a zero one only variants.
#   3. CORPUS     for every corpus adjoint and HVP: `coarsen = 1` is byte-identical to the default call, every plain kernel is byte-identical at
#                 `coarsen = 2`, a plan without a variant is byte-identical as a whole, and the kernels the jam must and must not fire on are the ones that
#                 do. The corpus numbers are checked by `validate_ownership(; coarsen = 2)`, which forces every variant and runs every plan on the shim
#                 against the CPU adjoint.
#   4. SABOTAGE   each guard, and each edit of the merge or the selector, removed in a copy of STADE.jl must make a unit or execution check fail.

# (label, expect jammed, merged sites, source, kernel name, sizes, builder)
function crs_units()
    rn(rng, k) = randn(rng, max(k, 0))
    return [
        ("key address under a loop (a weight gradient)", true, 1, """
function jam_wgrad(x, g, W, n, m, nk)
    for idx = 1:n * m
        oc = div(idx - 1, m) + 1
        s = g[idx]
        for kk = 1:nk
            W[(oc - 1) * nk + kk] = W[(oc - 1) * nk + kk] + x[mod(idx - 1 + kk - 1, n * m) + 1] * s
        end
    end
    return nothing
end""", "jam_wgrad", [(3, 7, 4), (4, 40, 3), (5, 64, 2), (2, 33, 5), (1, 1, 1), (6, 100, 3), (0, 5, 3), (2, 16, 2)],
         (rng, (n, m, nk)) -> Any[rn(rng, n * m), rn(rng, n * m), zeros(n * nk), n, m, nk]),
        ("thread-invariant address under a loop (a dense layer)", true, 1, """
function jam_inv(W, y, xb, nr, nc)
    for i = 1:nr
        s = y[i]
        for j = 1:nc
            xb[j] = xb[j] + W[(i - 1) * nc + j] * s
        end
    end
    return nothing
end""", "jam_inv", [(5, 3), (40, 4), (64, 2), (33, 7), (100, 5), (1, 1), (0, 3), (97, 2)],
         (rng, (nr, nc)) -> Any[rn(rng, nr * nc), rn(rng, nr), zeros(nc), nr, nc]),
        ("scatter through an array read stays plain", false, 0, """
function jam_scatter(x, ix, y, n, nk)
    for i = 1:n
        for kk = 1:nk
            j = ix[(i - 1) * nk + kk]
            y[j] = y[j] + x[i]
        end
    end
    return nothing
end""", "jam_scatter", [(10, 3), (70, 2)],
         (rng, (n, nk)) -> Any[rn(rng, n), rand(rng, 1:7, n * nk), zeros(7), n, nk]),
        ("key address under no loop stays plain", false, 0, """
function jam_noloop(g, W, n, m)
    for i = 1:n * m
        oc = div(i - 1, m) + 1
        W[oc] = W[oc] + g[i]
    end
    return nothing
end""", "jam_noloop", [(3, 40), (4, 9)],
         (rng, (n, m)) -> Any[rn(rng, n * m), zeros(n), n, m]),
        ("an add under a data-dependent branch is not mergeable", false, 0, """
function jam_ifcond(x, g, W, n, m, nk)
    for idx = 1:n * m
        oc = div(idx - 1, m) + 1
        for kk = 1:nk
            if g[idx] > 0.0
                W[(oc - 1) * nk + kk] = W[(oc - 1) * nk + kk] + x[idx] * g[idx]
            end
        end
    end
    return nothing
end""", "jam_ifcond", [(3, 40, 2), (2, 9, 3)],
         (rng, (n, m, nk)) -> Any[rn(rng, n * m), rn(rng, n * m), zeros(n * nk), n, m, nk]),
        ("a loop whose bound is read per iteration stays plain", false, 0, """
function jam_ragged(x, cnt, W, n, m)
    for idx = 1:n * m
        oc = div(idx - 1, m) + 1
        for kk = 1:cnt[idx]
            W[(oc - 1) * 4 + kk] = W[(oc - 1) * 4 + kk] + x[idx]
        end
    end
    return nothing
end""", "jam_ragged", [(3, 40), (2, 9)],
         (rng, (n, m)) -> Any[rn(rng, n * m), rand(rng, 0:4, n * m), zeros(n * 4), n, m]),
        ("a hoisted loop-invariant add leaves nothing to merge", false, 0, """
function jam_hoisted(x, W, n, m, nk)
    for idx = 1:n * m
        oc = div(idx - 1, m) + 1
        for kk = 1:nk
            W[oc] = W[oc] + x[(idx - 1) * nk + kk]
        end
    end
    return nothing
end""", "jam_hoisted", [(3, 40, 2), (2, 9, 3)],
         (rng, (n, m, nk)) -> Any[rn(rng, n * m * nk), zeros(n), n, m, nk]),
        ("one merged site beside an owned write and a bounded-index site", true, 1, """
function jam_mixed(x, g, out, W, H, n, m, nk)
    for idx = 1:n * m
        oc = div(idx - 1, m) + 1
        k = mod(idx - 1, m) + 1
        s = g[idx]
        out[idx] = s * 2.0
        for kk = 1:nk
            W[(oc - 1) * nk + kk] = W[(oc - 1) * nk + kk] + x[idx] * s
            H[(kk - 1) * m + k] = H[(kk - 1) * m + k] + s
        end
    end
    return nothing
end""", "jam_mixed", [(3, 40, 2), (2, 9, 3), (4, 33, 2), (1, 70, 2)],
         (rng, (n, m, nk)) -> Any[rn(rng, n * m), rn(rng, n * m), zeros(n * m), zeros(n * nk), zeros(nk * m), n, m, nk]),
        ("a step of 2 and a lower bound above 1", true, 1, """
function jam_step(x, g, W, n, m, nk)
    for idx = 2:2:n * m
        oc = div(idx - 1, m) + 1
        for kk = 1:nk
            W[(oc - 1) * nk + kk] = W[(oc - 1) * nk + kk] + x[idx] * g[idx]
        end
    end
    return nothing
end""", "jam_step", [(3, 40, 2), (2, 9, 3), (5, 64, 2)],
         (rng, (n, m, nk)) -> Any[rn(rng, n * m), rn(rng, n * m), zeros(n * nk), n, m, nk]),
        ("a descending loop", true, 1, """
function jam_down(x, g, W, n, m, nk)
    for idx = n * m:-1:2
        oc = div(idx - 1, m) + 1
        for kk = 1:nk
            W[(oc - 1) * nk + kk] = W[(oc - 1) * nk + kk] + x[idx] * g[idx]
        end
    end
    return nothing
end""", "jam_down", [(3, 40, 2), (2, 9, 3), (5, 64, 2)],
         (rng, (n, m, nk)) -> Any[rn(rng, n * m), rn(rng, n * m), zeros(n * nk), n, m, nk]),
        ("a loop with a per-iteration bound beside a mergeable one", true, 1, """
function jam_ragged_mix(x, g, cnt, W, V, n, m, nk)
    for idx = 1:n * m
        oc = div(idx - 1, m) + 1
        for kk = 1:nk
            W[(oc - 1) * nk + kk] = W[(oc - 1) * nk + kk] + x[idx] * g[idx]
        end
        for kk = 1:cnt[idx]
            V[(oc - 1) * 4 + kk] = V[(oc - 1) * 4 + kk] + g[idx]
        end
    end
    return nothing
end""", "jam_ragged_mix", [(3, 40, 2), (2, 9, 3), (4, 33, 2)],
         (rng, (n, m, nk)) -> Any[rn(rng, n * m), rn(rng, n * m), rand(rng, 0:4, n * m), zeros(n * nk), zeros(n * 4), n, m, nk]),
        ("a data-dependent branch beside a mergeable loop", true, 1, """
function jam_branch_mix(x, g, W, n, m, nk)
    for idx = 1:n * m
        oc = div(idx - 1, m) + 1
        s = g[idx]
        if s > 0.0
            s = s * 2.0
        else
            s = 0.0 - s
        end
        for kk = 1:nk
            W[(oc - 1) * nk + kk] = W[(oc - 1) * nk + kk] + x[idx] * s
        end
    end
    return nothing
end""", "jam_branch_mix", [(3, 40, 2), (2, 9, 3), (4, 33, 2)],
         (rng, (n, m, nk)) -> Any[rn(rng, n * m), rn(rng, n * m), zeros(n * nk), n, m, nk]),
        ("two merged sites in nested loops", true, 2, """
function jam_two(x, g, W, V, n, m, nk, nj)
    for idx = 1:n * m
        oc = div(idx - 1, m) + 1
        for kk = 1:nk
            W[(oc - 1) * nk + kk] = W[(oc - 1) * nk + kk] + x[idx] * g[idx]
            for jj = 1:nj
                V[((oc - 1) * nk + kk - 1) * nj + jj] = V[((oc - 1) * nk + kk - 1) * nj + jj] + g[idx] * jj
            end
        end
    end
    return nothing
end""", "jam_two", [(3, 40, 2, 3), (2, 9, 3, 2), (4, 33, 2, 2)],
         (rng, (n, m, nk, nj)) -> Any[rn(rng, n * m), rn(rng, n * m), zeros(n * nk), zeros(n * nk * nj), n, m, nk, nj]),
        ("a key site and an invariant site in one loop (the larger count selects)", true, 2, """
function jam_twosite(x, g, W, V, n, m, nk)
    for idx = 1:n * m
        oc = div(idx - 1, m) + 1
        for kk = 1:nk
            W[(oc - 1) * nk + kk] = W[(oc - 1) * nk + kk] + x[idx] * g[idx]
            V[kk] = V[kk] + g[idx] * kk
        end
    end
    return nothing
end""", "jam_twosite", [(3, 40, 2), (2, 9, 3), (4, 33, 2), (0, 5, 2)],
         (rng, (n, m, nk)) -> Any[rn(rng, n * m), rn(rng, n * m), zeros(n * nk), zeros(nk), n, m, nk]),
        ("a jammed loop followed by a plain one (launch numbers)", true, 1, """
function jam_seq(x, g, W, y, n, m, nk)
    for idx = 1:n * m
        oc = div(idx - 1, m) + 1
        for kk = 1:nk
            W[(oc - 1) * nk + kk] = W[(oc - 1) * nk + kk] + x[idx] * g[idx]
        end
    end
    for i = 1:n
        y[i] = y[i] + 2.0 * x[i]
    end
    return nothing
end""", "jam_seq", [(3, 40, 2), (2, 9, 3), (4, 33, 2), (0, 5, 2)],
         (rng, (n, m, nk)) -> Any[rn(rng, n * m), rn(rng, n * m), zeros(n * nk), zeros(n), n, m, nk]),
    ]
end

crs_src(e) = STADE.io_expr_to_source(e)
crs_has(e, sub) = e == sub || (e isa Expr && any(a -> crs_has(a, sub), e.args))
crs_jammed(k) = occursin("__jmb", crs_src(k))
crs_plain(ks) = [k for k in ks if !STADE.cgen_is_coarse_variant(k)]
crs_same_list(a, b) = length(a) == length(b) && all(crs_src(x) == crs_src(y) for (x, y) in zip(a, b))
crs_merged(k) = length(collect(eachmatch(r"__jmi\d+_2 == __jmi\d+_1", crs_src(k))))
crs_plan(e; kw...) = STADE.stade_gpu(e, STADE.cgen_backend_cuda(); kw...)

# runs the plan on the shim and the original kernel on the CPU; returns (largest relative difference, flagged elements, elements changed)
function crs_exec(src::String, fname::String, args::Vector, K::Int, order::Symbol; cmin::Int = 0)
    e = Meta.parse(src)
    plan = crs_plan(e; warp_atomics = false, shared_private = false, coarsen = K, coarsen_min = cmin, coarsen_fill = cmin == 0 ? 0 : 65536)
    m = Module(:CrsRun)
    Core.eval(m, :(using Main.CUDA))
    Core.eval(m, :(const CuArray = Main.CUDA.TArray))
    for k in plan.kernels
        Core.eval(m, k)
    end
    Core.eval(m, plan.host)
    ref = Module(:CrsRef)
    Core.eval(ref, e)
    cpu = Any[a isa AbstractArray ? copy(a) : a for a in args]
    Base.invokelatest(getfield(ref, Symbol(fname)), cpu...)
    dev = Any[a isa AbstractArray ? Main.CUDA.TArray(copy(a)) : a for a in args]
    Main.CUDA.ORDER[] = order
    empty!(Main.CUDA.VIOLATIONS)
    empty!(Main.CUDA.LAUNCHED)
    Base.invokelatest(getfield(m, Symbol(fname * "_cuda")), dev...)
    Main.CUDA.ORDER[] = :ascending
    err = 0.0
    changed = 0
    for (a, c, d) in zip(args, cpu, dev)
        a isa AbstractArray || continue
        got = Array(d)
        err = max(err, maximum(abs.(got .- c) ./ max.(abs.(c), 1.0); init = 0.0))
        changed += count(got .!= a)
    end
    return (err = err, flagged = length(Main.CUDA.VIOLATIONS), changed = changed, coarse = count(n -> endswith(n, "_c!"), Main.CUDA.LAUNCHED), launches = length(Main.CUDA.LAUNCHED))
end

function crs_guard_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    wa = STADE.cgen_warp_add_name()
    add(a, i, v) = Expr(:call, wa, a, i, v)
    stmt = (kind = :for, var = :i, lo = 1, hi = :n, step = 1, body = NamedTuple[])
    n_iter = STADE.cgen_trip_count(1, 1, :n)
    fargs = Symbol[:W, :x, :m, :nk, :n]
    good() = Any[:(oc = div(i - 1, m) + 1), Expr(:for, :(kk = 1:nk), Expr(:block, add(:W, :((oc - 1) * nk + kk), :(x[i]))))]
    body(dev; fa = fargs, synth = Dict{Symbol,Any}(), K = 2) = STADE.cgen_coarsen_body(stmt, K, dev, fa, synth, n_iter)
    check(body(good()) !== nothing, "control: a plain weight-gradient body is jammed")
    check(body(good(); K = 1) === nothing, "a factor of 1 emits nothing")
    check(body(Any[:(m = 2), good()...]) === nothing, "a scalar argument assigned in the body refuses (only the argument guard sees it: the count would be taken from the wrong value)")
    check(body(Any[:(kk = 5), good()...]) === nothing, "a loop variable assigned outside its loop refuses (only the loop-variable guard sees it)")
    check(body(Any[:(t = t + 1.0), good()...]) === nothing, "a scalar read before it is assigned refuses")
    check(body(Any[good()..., Expr(:if, :(i > 3), Expr(:block, :(return nothing)))]) === nothing, "a return inside the body refuses")
    check(body(Any[:(__jmx = 1.0), good()..., :(y = __jmx)]) === nothing, "a name with the generated prefix refuses")
    check(body(Any[:(abs = 2.0), good()..., :(y = abs(x[i]))]) === nothing, "a scalar that is also called as a function refuses")
    check(body(vcat(Any[Expr(:(=), Symbol("t", k), :(x[i] + 1.0)) for k in 1:1200], good())) === nothing, "a body over the node cap refuses")
    check(body(Any[:(oc = div(i - 1, m) + 1), :(y = x[oc])]) === nothing, "a body with no atomic leaves the kernel alone")
    check(body(Any[:(oc = i), Expr(:for, :(kk = 1:nk), Expr(:block, add(:W, :(oc + kk), :(x[i]))))]) === nothing, "a wide address under a loop is not worth a jam")
    check(body(good(); synth = Dict{Symbol,Any}(:acc => 0.0)) !== nothing, "a synthesized scalar is renamed and initialised per copy")
    k3 = body(good(); K = 3)
    check(k3 !== nothing && occursin("__jmt3", crs_src(Expr(:block, k3.body...))), "a factor of 3 builds three copies")
    # the threads-per-address count: a division by a kernel argument gives that argument, a thread-invariant address the trip count, the smaller of two
    # the minimum, and a count that cannot be written in the kernel's arguments gives no variant at all
    c1 = body(good())
    check(c1 !== nothing && c1.counts == Any[:m], "a key address counts threads by its divisor ($(c1 === nothing ? "-" : c1.counts))")
    c2 = body(Any[:(oc = i), :(j = 3), Expr(:for, :(kk = 1:nk), Expr(:block, add(:W, :(j + kk), :(x[i]))))])
    check(c2 !== nothing && c2.counts == Any[n_iter], "a thread-invariant address counts every thread")
    c3 = body(Any[good()..., Expr(:for, :(kk = 1:nk), Expr(:block, add(:V, :(kk), :(x[i]))))])
    check(c3 !== nothing && Set(c3.counts) == Set(Any[:m, n_iter]), "two merged sites give two counts")
    c4 = body(Any[:(q = x[1]), :(oc = div(i - 1, q) + 1), Expr(:for, :(kk = 1:nk), Expr(:block, add(:W, :((oc - 1) * nk + kk), :(x[i]))))])
    check(c4 === nothing, "a divisor read from an array cannot be written in the arguments, so there is no variant")
    c5 = body(Any[:(mm = m * 2), :(oc = div(i - 1, mm) + 1), Expr(:for, :(kk = 1:nk), Expr(:block, add(:W, :((oc - 1) * nk + kk), :(x[i]))))])
    check(c5 !== nothing && strip(crs_src(c5.counts[1])) == "m * 2", "a divisor held in a scalar is replaced by its definition ($(c5 === nothing ? "-" : strip(crs_src(c5.counts[1]))))")
    return bad
end

function crs_unit_checks(; execute::Bool = true, only::Union{Nothing,Vector{String}} = nothing)
    isdefined(Main, :CUDA) || include(joinpath(@__DIR__, "ownership_shim.jl"))
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    cuda = STADE.cgen_backend_cuda()
    for (label, jam, merged, src, fname, sizes, mk) in crs_units()
        only !== nothing && !(fname in only) && continue
        e = Meta.parse(src)
        p1 = crs_plan(e; coarsen = 1)
        pd = crs_plan(e)
        p2 = crs_plan(e; coarsen = 2)
        check(crs_src(p1.host) == crs_src(pd.host) && crs_same_list(p1.kernels, pd.kernels), "$fname: coarsen = 1 is the default output")
        check(crs_same_list(crs_plain(p2.kernels), p1.kernels), "$fname: every plain kernel is byte-identical at coarsen = 2")
        vars = [k for k in p2.kernels if STADE.cgen_is_coarse_variant(k)]
        check((length(vars) > 0) == jam && all(crs_jammed, vars), "$fname: $label -- variant = $(length(vars) > 0) (expected $jam)")
        if jam
            k = first(vars)
            parts = STADE.cgen_kernel_parts(k)
            check(parts !== nothing, "$fname: the variant has the standard kernel shape")
            check(crs_merged(k) == merged, "$fname: $(crs_merged(k)) merged site(s) (expected $merged)")
            nt = parts.guard.args[1].args[3]
            check(crs_has(p2.host, nt), "$fname: the launch sizes its blocks from the variant's own thread count")
            check(crs_has(p2.host, k.args[1].args[1]) && count(x -> x isa Expr && x.head == :if, p2.host.args[2].args) >= 1, "$fname: the host launches the variant under a condition")
            hs = crs_src(p2.host)
            check(occursin(">= 256", hs) && occursin(">= 65536", hs), "$fname: the defaults are 256 threads per address and 65536 threads in the launch")
            check(occursin(">= 777", crs_src(crs_plan(e; coarsen = 2, coarsen_min = 777).host)), "$fname: coarsen_min sets the contention threshold")
            check(occursin(">= 555", crs_src(crs_plan(e; coarsen = 2, coarsen_fill = 555).host)), "$fname: coarsen_fill sets the launch-size threshold")
            check(count(">= 0", crs_src(crs_plan(e; coarsen = 2, coarsen_min = 0, coarsen_fill = 0).host)) == 2, "$fname: both thresholds at 0 launch the variant always")
        else
            check(crs_src(p1.host) == crs_src(p2.host) && crs_same_list(p1.kernels, p2.kernels), "$fname: a kernel without a variant gives a byte-identical plan at coarsen = 2")
        end
        if execute
            worst = 0.0; flagged = 0; seen_change = true; forced_ok = true; declined_ok = true
            for K in (2, 3, 4), order in (:ascending, :shuffled), sz in sizes
                rng = MersenneTwister(hash((fname, sz)))
                args = mk(rng, sz)
                r = try crs_exec(src, fname, args, K, order) catch err; (err = Inf, flagged = 0, changed = 0, coarse = 0, launches = 0, msg = first(sprint(showerror, err), 120)) end
                worst = max(worst, r.err); flagged += r.flagged
                sz[1] > 0 && r.changed == 0 && (seen_change = false)
                sz[1] > 0 && jam && r.coarse == 0 && (forced_ok = false)
                (!jam && r.coarse != 0) && (forced_ok = false)
            end
            # a threshold nobody reaches launches plain kernels only, and gives the plain answer
            for sz in sizes
                rng = MersenneTwister(hash((fname, sz)))
                r = try crs_exec(src, fname, mk(rng, sz), 2, :ascending; cmin = 2^30) catch err; (err = Inf, flagged = 0, changed = 0, coarse = 1, launches = 0) end
                r.coarse == 0 && r.err <= 1e-12 || (declined_ok = false)
            end
            check(worst <= 1e-12 && flagged == 0 && seen_change, "$fname: K = 2,3,4 forced, on the shim, both thread orders, $(length(sizes)) sizes: worst relative difference $(round(worst; sigdigits = 2)), $flagged flagged element(s)$(seen_change ? "" : ", a run wrote nothing")")
            check(forced_ok, "$fname: coarsen_min = 0 launches the variant" * (jam ? "" : "-less plan with plain kernels only"))
            check(declined_ok, "$fname: an unreachable coarsen_min launches only plain kernels")
        end
    end

    # a variant is never privatized, the plain kernel still is
    for (label, jam, merged, src, fname, sizes, mk) in crs_units()
        fname == "jam_mixed" || continue
        e = Meta.parse(src)
        k1 = crs_plan(e; coarsen = 1).kernels[1]
        p2 = crs_plan(e; coarsen = 2)
        check(STADE.cgen_private_arrays(k1) == [:H], "jam_mixed: the bounded-index array is privatizable in the plain kernel")
        names = [string(k.args[1].args[1]) for k in p2.kernels]
        check(length(unique(names)) == length(names) && count(n -> endswith(n, "_c!"), names) == 1 && count(n -> endswith(n, "_p!"), names) == 1, "jam_mixed: one variant, one private kernel, no name twice ($(join(names, ", ")))")
        hs = crs_src(p2.host)
        check(occursin(r"_p!.*else.*_c!.*else.*_1!"s, hs), "jam_mixed: the host tries the private kernel first, then the variant, then the plain kernel")
    end
    for (label, jam, merged, src, fname, sizes, mk) in crs_units()
        fname == "jam_twosite" || continue
        hs = crs_src(crs_plan(Meta.parse(src); coarsen = 2).host)
        check(occursin(r"if max\(.*\) >= 256", hs), "jam_twosite: two merged sites select on the larger count ($(join(strip.(filter(l -> occursin("if max(", l), split(hs, "\n"))), "; ")))")
    end
    # two offloaded loops: launch indices skip the variants, so the plain kernels keep the numbers they have at coarsen = 1
    for (label, jam, merged, src, fname, sizes, mk) in crs_units()
        fname in ("jam_two", "jam_seq") || continue
        e = Meta.parse(src)
        p1 = crs_plan(e; coarsen = 1)
        p2 = crs_plan(e; coarsen = 2)
        n1 = [string(k.args[1].args[1]) for k in p1.kernels]
        n2 = [string(k.args[1].args[1]) for k in crs_plain(p2.kernels)]
        check(n1 == n2 && length(unique(n1)) == length(n1) && (fname != "jam_seq" || length(n1) == 2), "$fname: plain kernels keep their names ($(join(n2, ", ")))")
    end
    bad += crs_guard_checks()
    return bad
end
parts_n(k) = STADE.cgen_kernel_parts(k).guard.args[1].args[3]

# corpus kernels the jam must fire on, and ones it must leave alone
const CRS_MUST = ["mlp1d", "mpnn", "transformer", "unet", "selfrace_conv", "remat_cross", "idm_tile", "so_tile"]
const CRS_MUST_NOT = ["dotprod", "ttgc"]

function crs_corpus_checks(dir::String; names = nothing, report::Bool = true)
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    all_names = sort([splitext(f)[1] for f in readdir(dir) if endswith(f, ".jl") && !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl", "_cuda.jl", "_jacc.jl"))])
    names === nothing || (all_names = filter(n -> n in names, all_names))
    fired = Dict{String,Int}()
    for name in all_names
        pe = STADE.io_read_corpus_entry(joinpath(dir, name * ".jl"))
        for (mode, gen) in (("adjoint", STADE.stade_adjoint), ("hvp", STADE.stade_hvp))
            out = try gen(pe; keep_push_pop = false, fuse_ii_loops = true) catch; continue end
            e = mode == "hvp" ? out.hvp : out.adjoint
            p0 = crs_plan(e; coarsen = 1)
            pd = crs_plan(e)
            p2 = crs_plan(e; coarsen = 2)
            same(a, b) = crs_same_list(a.kernels, b.kernels) && crs_src(a.host) == crs_src(b.host)
            check(same(p0, pd), "$name [$mode]: coarsen = 1 equals the default call")
            vars = [k for k in p2.kernels if STADE.cgen_is_coarse_variant(k)]
            nj = length(vars)
            fired[name] = get(fired, name, 0) + nj
            check(crs_same_list(crs_plain(p2.kernels), p0.kernels), "$name [$mode]: $nj variant(s) added, every plain kernel is byte-identical to coarsen = 1")
            nj == 0 && check(same(p0, p2), "$name [$mode]: a plan without a variant is byte-identical at coarsen = 2")
            for k in vars
                check(crs_jammed(k) && STADE.cgen_kernel_parts(k) !== nothing && crs_has(p2.host, parts_n(k)) && crs_has(p2.host, k.args[1].args[1]),
                      "$name [$mode]: a variant is launched, under a condition, with its own thread count")
            end
        end
    end
    for n in CRS_MUST
        n in all_names && check(get(fired, n, 0) > 0, "$n: the jam fires")
    end
    for n in CRS_MUST_NOT
        n in all_names && check(get(fired, n, 0) == 0, "$n: the jam does not fire")
    end
    report && println("jammed kernels by corpus entry: ", join(["$k=$v" for (k, v) in sort(collect(fired)) if v > 0], " "))
    return bad
end

function validate_coarsen(dir::String = joinpath(@__DIR__, "val-corpus"); sabotage::Bool = true)
    isdefined(Main, :CUDA) || include(joinpath(@__DIR__, "ownership_shim.jl"))
    bad = 0
    println("-- unit and execution checks")
    bad += crs_unit_checks()
    println("-- corpus")
    bad += crs_corpus_checks(dir)
    if sabotage
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        repo = abspath(joinpath(@__DIR__, ".."))
        text = read(joinpath(repo, "src", "STADE.jl"), String)
        cases = [
            ("nothing-to-merge gate removed", "    isempty(counts) && return nothing\n", ""),
            ("node cap removed", "cgen_jam_heads!(whole, heads) <= cgen_jam_max_nodes() || return nothing", "cgen_jam_heads!(whole, heads) <= 10^9 || return nothing"),
            ("return check removed", "    cgen_jam_has_head(whole, :return) && return nothing\n", ""),
            ("generated-name check removed", "    any(s -> startswith(string(s), \"__jm\"), cgen_expr_symbols(whole)) && return nothing\n", ""),
            ("loop-variable check removed", "    isempty(intersect(R, lv)) || return nothing\n", ""),
            ("call-position check removed", "    isempty(intersect(R, heads)) || return nothing\n", ""),
            ("argument check removed", "    isempty(intersect(R, Set{Symbol}(fargs))) || return nothing\n", ""),
            ("first-use check removed", "    cgen_jam_first_use!(whole, R, seen) || return nothing\n", ""),
            ("loops with a per-copy bound are merged", "elseif e.head == :for && isempty(intersect(cgen_expr_symbols(e.args[1].args[2]), R))", "elseif e.head == :for"),
            ("branches on a per-copy condition are merged", "elseif e.head == :if && isempty(intersect(cgen_expr_symbols(e.args[1]), R))", "elseif e.head == :if"),
            ("every copy assumed valid", "emit_if(Expr(:call, :<=, Expr(:call, :+, jb, 32 * (K - 1) + 1), n_iter),", "emit_if(Expr(:call, :<=, 0, 1),"),
            ("merged without comparing addresses", "push!(out, emit_if(same, Any[cgen_atomic_rebuild2(e, ik[1], total)]", "push!(out, emit_if(Expr(:call, :(==), 0, 0), Any[cgen_atomic_rebuild2(e, ik[1], total)]"),
            ("a wide address is merged too", "    cgen_jam_class(s.idx, env, denv)[1] === :wide && return Any[cgen_jam_rename(e, maps[k]) for k in 1:K]\n", ""),
            ("too few threads launched", "nthreads = Expr(:call, :*, Expr(:call, :div, Expr(:call, :+, n_iter, 32 * K - 1), 32 * K), 32)", "nthreads = Expr(:call, :*, Expr(:call, :div, Expr(:call, :+, n_iter, 32 * K - 1), 32 * K), 16)"),
            ("tail lanes 31 apart", "Expr(:call, :*, 32, :__jmk)", "Expr(:call, :*, 31, :__jmk)"),
            ("copies not renamed apart", "maps = [Dict{Symbol,Symbol}(s => Symbol(string(s), \"__j\", k) for s in R) for k in 1:K]", "maps = [Dict{Symbol,Symbol}(s => Symbol(string(s), \"__j\", 1) for s in R) for k in 1:K]"),
            ("launch ignores the variant's thread count", "nthreads = cv.nthreads, kname = cv.name)", "kname = cv.name)"),
            ("variant launched always", "cond = Expr(:&&, Expr(:call, :>=, best, cgen_coarsen_min_threads(backend)), Expr(:call, :>=, n_iter, cgen_coarsen_fill_threads(backend))))", "cond = Expr(:call, :>=, 1, 0))"),
            ("variant never launched", "cond = Expr(:&&, Expr(:call, :>=, best, cgen_coarsen_min_threads(backend)), Expr(:call, :>=, n_iter, cgen_coarsen_fill_threads(backend))))", "cond = Expr(:call, :>=, 0, 1))"),
            ("coarsen_fill ignored", "cgen_coarsen_fill_threads(backend) = get(backend, :coarsen_fill, 65536)", "cgen_coarsen_fill_threads(backend) = 65536"),
            ("variant dropped when a private kernel exists", "push!(out, emit_if(cond, Any[private], plain_stmts))", "push!(out, emit_if(cond, Any[private], Any[plain]))"),
            ("coarsen_min ignored", "cgen_coarsen_min_threads(backend) = get(backend, :coarsen_min, 256)", "cgen_coarsen_min_threads(backend) = 256"),
            ("one count for every site", "best = length(jam.counts) == 1 ? jam.counts[1] : Expr(:call, :max, jam.counts...)", "best = jam.counts[1]"),
            ("variant privatized", "            cgen_is_coarse_variant(k) && continue\n", ""),
            ("variant takes a launch number", "cgen_kernel_count(kernels::Vector{Expr}) = count(k -> !cgen_is_coarse_variant(k), kernels)", "cgen_kernel_count(kernels::Vector{Expr}) = length(kernels)"),
            ("divisor not resolved", "        return cgen_jam_resolve(defs[d], defs, fargs, depth + 1)", "        return d"),
            ("invariant address counted as one thread", "d = c === :inv ? n_iter : cgen_jam_resolve(d, defs, fargs, 0)", "d = c === :inv ? 1 : cgen_jam_resolve(d, defs, fargs, 0)"),
        ]
        mktempdir() do tmp
            for (label, old, new) in cases
                n = length(findall(old, text))
                if n != 1
                    check(false, "sabotage \"$label\": anchor found $n times")
                    continue
                end
                path = joinpath(tmp, "STADE_sab.jl")
                write(path, replace(text, old => new; count = 1))
                script = """
                    include($(repr(path)))
                    include($(repr(joinpath(@__DIR__, "ownership_shim.jl"))))
                    include($(repr(joinpath(@__DIR__, "validate_coarsen.jl"))))
                    n = crs_unit_checks()
                    println("RESULT ", n)
                """
                out = try read(Base.pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all coarsening checks passed" : "*** $bad coarsening check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_coarsen() == 0 ? 0 : 1)
end
