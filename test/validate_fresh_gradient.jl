include(joinpath(@__DIR__, "sabotage_run.jl"))
include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random
isdefined(Main, :CUDA) || include(joinpath(@__DIR__, "ownership_shim.jl"))

#=
    validate_fresh_gradient(; only=nothing, sabotage=true)

fresh_gradient (CUDA) turns `g[idx] = g[idx] + e` into `g[idx] = e` for an adjoint array that provably gets one write per element per call. Each case below is a
small kernel in the house style, differentiated by stade_adjoint (keep_push_pop = false) and lowered with and without the keyword.

  1. FLAGGED   the arrays the pass rewrote are exactly the expected ones (the recognizer's list, `plan.fresh`), and the text differs from the plain plan
               at nothing but those stores.
  2. ZERO      from zero gradients the plan with the keyword gives the same arrays as the plan without it, on the CPU shim, for several sizes (an
               empty loop included), with no ownership violation.
  3. NONZERO   from non-zero gradients a flagged array differs (that is the contract: the caller owns the zero state) and every other array agrees.
               A case with nothing flagged agrees everywhere. This is what makes a wrongly flagged accumulation visible: it differs from zero.
  4. OFF       the keyword off is the default output, byte for byte; AMDGPU and Metal never flag.
  5. SABOTAGE  with each guard replaced by `true`, the case built to need it is flagged and check 2 or 3 fails.
  6. CHECK     `fresh_check = true` adds one entry check per flagged array and changes nothing else (the kernels are the same text, the keyword off or without
               `fresh_gradient` is the plain output). From zero gradients it runs and agrees with the plan without the check. From a non-zero gradient, and
               from the gradient a first call left behind, it raises the contract error. An empty array passes and a NaN does not.
=#

fg_src(e) = STADE.io_expr_to_source(e)

# (label, expected flagged, primal source, name, sizes (a NamedTuple of the integer arguments) list)
function fg_cases()
    return [
        ("dense layer: weights and bias", [:wb, :bb], """
function fg_dense(x, w, b, y, n, m)
    for j = 1:m
        s = b[j]
        for i = 1:n
            s += w[(i - 1) * m + j] * x[i]
        end
        y[j] = s
    end
    return nothing
end""", "fg_dense", [(n = 5, m = 4), (n = 3, m = 7), (n = 6, m = 1), (n = 0, m = 3), (n = 3, m = 0)]),
        ("two layers: the intermediate is not flagged", [:w1b, :b1b, :w2b, :b2b], """
function fg_two(x, w1, b1, w2, b2, h, y, n, m, k)
    for j = 1:m
        s = b1[j]
        for i = 1:n
            s += w1[(i - 1) * m + j] * x[i]
        end
        h[j] = s
    end
    for j = 1:k
        s = b2[j]
        for i = 1:m
            s += w2[(i - 1) * k + j] * h[i]
        end
        y[j] = s
    end
    return nothing
end""", "fg_two", [(n = 4, m = 5, k = 3), (n = 2, m = 2, k = 6)]),
        ("a bias beside an input gradient", [:xb, :bb], """
function fg_bias(x, b, y, n)
    for j = 1:n
        y[j] = x[j] + b[j]
    end
    return nothing
end""", "fg_bias", [(n = 6,), (n = 1,)]),
        ("a weight used by two loops is not flagged", Symbol[], """
function fg_shared(x, w, y, z, n)
    for j = 1:n
        s = 0.0
        for i = 1:n
            s += w[(i - 1) * n + j] * x[i]
        end
        y[j] = s
    end
    for j = 1:n
        s = 0.0
        for i = 1:n
            s += w[(i - 1) * n + j] * y[i]
        end
        z[j] = s
    end
    return nothing
end""", "fg_shared", [(n = 4,), (n = 6,)]),
        ("an element reused by every iteration of the inner loop", Symbol[], """
function fg_tsum(x, w, y, n, m)
    for j = 1:m
        s = 0.0
        for i = 1:n
            s += w[j] * x[i]
        end
        y[j] = s
    end
    return nothing
end""", "fg_tsum", [(n = 5, m = 4), (n = 3, m = 3)]),
        ("a stride that is not the thread count", Symbol[], """
function fg_stride(x, w, y, n, m, k)
    for j = 1:m
        s = 0.0
        for i = 1:n
            s += w[(i - 1) * k + j] * x[i]
        end
        y[j] = s
    end
    return nothing
end""", "fg_stride", [(n = 4, m = 5, k = 5), (n = 4, m = 5, k = 0), (n = 3, m = 6, k = 7)]),
        ("a launch inside a host loop", Symbol[], """
function fg_steps(x, w, y, n, ns)
    for t = 1:ns
        for j = 1:n
            y[j] = y[j] + w[j] * x[j]
        end
    end
    return nothing
end""", "fg_steps", [(n = 5, ns = 3), (n = 4, ns = 1)]),
        ("a gathered weight index: the input gradient only", [:xb], """
function fg_gather(x, w, idx, y, n)
    for j = 1:n
        k = idx[j]
        y[j] = x[j] * w[k]
    end
    return nothing
end""", "fg_gather", [(n = 6,)]),
        ("a weight shared by every thread", Symbol[], """
function fg_conv(x, w, y, n, nk)
    for o = 1:n
        s = 0.0
        for k = 1:nk
            s += w[k] * x[o + k - 1]
        end
        y[o] = s
    end
    return nothing
end""", "fg_conv", [(n = 5, nk = 3), (n = 2, nk = 2)]),
        ("an index that a scalar of the loop body makes repeat", Symbol[], """
function fg_hidden(x, w, y, n, m)
    for j = 1:m
        s = 0.0
        for i = 1:n
            c = mod(i, 2) * m
            s += w[(i - 1) * m + c + j] * x[i]
        end
        y[j] = s
    end
    return nothing
end""", "fg_hidden", [(n = 4, m = 3), (n = 5, m = 2)]),
        ("one weight in two statements of one loop", Symbol[], """
function fg_twosites(x, z, w, y, u, n, m)
    for j = 1:m
        s = 0.0
        t = 0.0
        for i = 1:n
            s += w[(i - 1) * m + j] * x[i]
            t += w[(i - 1) * m + j] * z[i] * z[i]
        end
        y[j] = s
        u[j] = t
    end
    return nothing
end""", "fg_twosites", [(n = 4, m = 3), (n = 3, m = 5)]),
    ]
end

# arguments of the adjoint host function: random primal values, zero (or random) adjoints
function fg_args(host::Expr, kinds, mk, sz; adj_start::Symbol = :zero, rng)
    args = Any[]
    prim = Dict{Symbol,Any}()
    for a in host.args[1].args[2:end]
        s = string(a)
        if haskey(kinds, a)
            if kinds[a] in (:scalar_int,)
                v = getfield(sz, a)
            elseif kinds[a] == :array_int
                v = mk(a, rng)
            elseif kinds[a] == :array_float
                v = mk(a, rng)
            else
                v = STADE.val_randn(rng)
            end
            prim[a] = v; push!(args, v)
        elseif endswith(s, "b") && haskey(kinds, Symbol(chop(s)))
            base = prim[Symbol(chop(s))]
            if base isa AbstractArray
                push!(args, adj_start == :zero ? zeros(length(base)) : STADE.val_randn(rng, length(base)))
            else
                push!(args, adj_start == :zero ? 0.0 : STADE.val_randn(rng))
            end
        else
            error("fg_args: no value for $a")
        end
    end
    return args
end

function fg_run(plan, fname, args; order::Symbol = :ascending)
    m = Module(:FgRun)
    Core.eval(m, :(using Main.CUDA))
    Core.eval(m, :(const CuArray = Main.CUDA.TArray))
    for h in plan.helpers
        Core.eval(m, h)
    end
    for k in plan.kernels
        Core.eval(m, k)
    end
    Core.eval(m, plan.host)
    dev = Any[a isa AbstractArray ? Main.CUDA.TArray(copy(a)) : a for a in args]
    Main.CUDA.ORDER[] = order
    empty!(Main.CUDA.VIOLATIONS); empty!(Main.CUDA.LAUNCHED); empty!(Main.CUDA.PROBES)
    ret = Base.invokelatest(getfield(m, Symbol(fname * "_b_cuda")), dev...)
    Main.CUDA.ORDER[] = :ascending
    return (arrays = Any[a isa AbstractArray ? Array(d) : d for (a, d) in zip(args, dev)], ret = ret, flagged = length(Main.CUDA.VIOLATIONS))
end

fg_plan(e; kw...) = STADE.stade_gpu(e, STADE.cgen_backend_cuda(); kw..., warp_atomics = false, shared_private = false, inbounds = false, dead_scalars = false, split_reduce = 0, contraction = 0)

# the small guards on handmade input, so that each one is exercised on a shape the generated kernels do not reach
function fg_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    guard(e) = Expr(:if, Expr(:call, :>, :__tid, e), Expr(:block))
    params = Set{Symbol}([:n, :m, :k])
    gn = guard(:n); gdiv = guard(Expr(:call, :+, Expr(:call, :div, Expr(:call, :-, :n, 1), 1), 1)); grev = guard(Expr(:call, :+, Expr(:call, :div, Expr(:call, :-, 1, :n), -1), 1))
    gother = guard(Expr(:call, :+, :n, :m))
    check(STADE.cgen_fresh_count_positive(gn, :n) && STADE.cgen_fresh_count_positive(gdiv, :n) && STADE.cgen_fresh_count_positive(grev, :n), "a thread count of the three emitted forms proves the count at least 1")
    check(!STADE.cgen_fresh_count_positive(gn, :m) && !STADE.cgen_fresh_count_positive(gother, :n), "another symbol, or another count expression, proves nothing")
    check(STADE.cgen_fresh_coeff_nonzero(3, params, gn) && !STADE.cgen_fresh_coeff_nonzero(0, params, gn), "a literal coefficient is nonzero unless it is 0")
    check(STADE.cgen_fresh_coeff_nonzero(:n, params, gn) && !STADE.cgen_fresh_coeff_nonzero(:k, params, gn) && !STADE.cgen_fresh_coeff_nonzero(:q, params, gn), "a parameter coefficient needs the count proof")
    check(STADE.cgen_fresh_coeff_nonzero(Expr(:call, :*, 2, :n), params, gn) && !STADE.cgen_fresh_coeff_nonzero(Expr(:call, :*, :n, :k), params, gn) && !STADE.cgen_fresh_coeff_nonzero(Expr(:call, :+, :n, 1), params, gn), "a product is nonzero when every factor is, a sum is not accepted")
    site(stmt) = Any[(stmt = stmt, loops = Symbol[])]
    lhs = :(w[i]); good = Expr(:(=), lhs, :(w[i] + e)); other = Expr(:(=), lhs, :(e)); two = Expr(:(=), lhs, :(w[i] + w[i - 1]))
    check(STADE.cgen_fresh_single_store(Expr(:block, good, :(q = w[i])), :w, site(good)) == false, "a second mention of the array refuses")
    check(STADE.cgen_fresh_single_store(good, :w, site(good)) && !STADE.cgen_fresh_single_store(other, :w, site(other)) && !STADE.cgen_fresh_single_store(two, :w, site(two)), "only `w[i] = w[i] + e` with no other mention is accepted")
    check(!STADE.cgen_fresh_single_store(good, :w, vcat(site(good), site(good))), "two stores refuse")
    check(STADE.cgen_fresh_index_names_ok(:((i - 1) * n + j), params, Symbol[:i], Set{Symbol}([:j])) && !STADE.cgen_fresh_index_names_ok(:((i - 1) * n + c), params, Symbol[:i], Set{Symbol}([:j])), "a scalar of the loop body in the index refuses")
    hs = Any[:(a = 1), :(@cuda f(w)), :(g(z))]
    check(STADE.cgen_fresh_host_group(hs, Expr(:block, hs...), :w) == hs[2] && STADE.cgen_fresh_host_group(hs, Expr(:block, hs..., :(h(w))), :w) === nothing && STADE.cgen_fresh_host_group(Any[:(@cuda f(w)), :(@cuda g(w))], Expr(:block, :(@cuda f(w)), :(@cuda g(w))), :w) === nothing, "the host group is the one statement that holds every mention")
    kf = Meta.parse("function cuda_kernel_q_1!(w, n)\n    __tid = 1\n    return nothing\nend")
    kdefs = Dict{Symbol,Expr}(:cuda_kernel_q_1! => kf)
    plainl = Meta.parse("@cuda threads = 256 blocks = 1 cuda_kernel_q_1!(w, n)")
    loopl = Meta.parse("for t = 1:3\n    @cuda threads = 256 blocks = 1 cuda_kernel_q_1!(w, n)\nend")
    check(STADE.cgen_fresh_top_level_launch(plainl, kdefs, Symbol("@cuda")) !== nothing && STADE.cgen_fresh_top_level_launch(loopl, kdefs, Symbol("@cuda")) === nothing, "a launch inside a host loop is not a top-level launch")
    kd = Expr(:function, :(k!(w, e)), Expr(:block, good))
    kd2 = Expr(:function, :(k!(w, e)), Expr(:block, good, :(q = w[1])))
    check(STADE.cgen_fresh_variant_ok(kd, :w, good) && !STADE.cgen_fresh_variant_ok(kd2, :w, good) && !STADE.cgen_fresh_variant_ok(Expr(:function, :(k!(w, e)), Expr(:block, other, :(q = w[1]))), :w, good), "a variant must hold the same one store and nothing else of the array")
    # fresh_check on a handmade host: zero, empty, tiny, NaN and a second array
    hostc = Meta.parse("function fgc_h(wb, bb, n)\n    return nothing\nend")
    chk = STADE.cgen_fresh_check(hostc, Symbol[:wb, :bb])
    mc = Module(:FgCheck); Core.eval(mc, chk)
    runc(a, b) = try Base.invokelatest(getfield(mc, :fgc_h), a, b, 1); "pass" catch ex; ex isa ErrorException && occursin("must be zero on entry", ex.msg) ? "raise" : "other" end
    # the isempty guard cannot be sabotaged here: Base gives maximum(abs, empty) == 0, so the CPU passes without it. It is for the device library, and the V100 run covers it.
    check(runc(zeros(3), zeros(2)) == "pass", "fresh_check: zero arrays pass")
    check(runc(Float64[], zeros(2)) == "pass" && runc(zeros(3), Float64[]) == "pass", "fresh_check: an empty array passes")
    check(runc([0.0, 1e-300, 0.0], zeros(2)) == "raise" && runc(zeros(3), [0.0, -2.0]) == "raise", "fresh_check: a tiny or negative value in either array raises")
    check(runc([0.0, NaN, 0.0], zeros(2)) == "raise" && runc(zeros(3), [NaN, 0.0]) == "raise", "fresh_check: a NaN raises")
    check(STADE.cgen_fresh_check(hostc, Symbol[]) === hostc, "fresh_check: nothing flagged leaves the host as it is")
    check(length(chk.args[2].args) == length(hostc.args[2].args) + 2 && chk.args[2].args[end] == hostc.args[2].args[end], "fresh_check: the checks come first and the body is unchanged")
    return bad
end

function validate_fresh_gradient(; only::Union{Nothing,Vector{String}} = nothing, sabotage::Bool = true, execute::Bool = true)
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    bad += fg_unit_checks()
    for (label, want, src, fname, sizes) in fg_cases()
        only !== nothing && !(fname in only) && continue
        primal = Meta.parse(src)
        kern = STADE.parse_kernel(primal)
        kinds = kern.sig.kinds
        out = STADE.stade_adjoint(primal; keep_push_pop = false)
        e = out.adjoint
        plain = fg_plan(e)
        fresh = fg_plan(e; fresh_gradient = true)
        check(sort(fresh.fresh) == sort(want), "$fname: $label -- flagged $(fresh.fresh) (expected $want)")
        check(isempty(plain.fresh), "$fname: the plan without the keyword flags nothing")
        # 4. OFF: the default output equals the plan with the keyword off
        dflt = STADE.stade_gpu(e, STADE.cgen_backend_cuda())
        offp = STADE.stade_gpu(e, STADE.cgen_backend_cuda(); fresh_gradient = false)
        check(fg_src(dflt.host) == fg_src(offp.host) && length(dflt.kernels) == length(offp.kernels) && all(fg_src(a) == fg_src(b) for (a, b) in zip(dflt.kernels, offp.kernels)), "$fname: fresh_gradient = false is the default output")
        for (lab, be) in (("amdgpu", STADE.cgen_backend_amdgpu()), ("metal", STADE.cgen_backend_metal()))
            pl = try STADE.stade_gpu(e, be) catch; continue end
            check(isempty(pl.fresh), "$fname: $lab output flags nothing")
        end
        # 6. CHECK: text
        fc = fg_plan(e; fresh_gradient = true, fresh_check = true)
        fcoff = fg_plan(e; fresh_gradient = true, fresh_check = false)
        fconly = fg_plan(e; fresh_check = true)
        check(fc.fresh == fresh.fresh, "$fname: fresh_check flags the same arrays")
        check(length(fc.kernels) == length(fresh.kernels) && all(fg_src(a) == fg_src(b) for (a, b) in zip(fc.kernels, fresh.kernels)), "$fname: fresh_check leaves the kernels as they are")
        check(fg_src(fcoff.host) == fg_src(fresh.host), "$fname: fresh_check = false is the plan without the check")
        check(fg_src(fconly.host) == fg_src(plain.host), "$fname: fresh_check without fresh_gradient is the plain output")
        ctext = fg_src(fc.host)
        ncheck = length(collect(eachmatch(r"must be zero on entry", ctext)))
        check(ncheck == length(want) && all(w -> occursin("gradient array $w of", ctext), want), "$fname: one entry check for each flagged array ($ncheck for $(length(want)))")
        check(isempty(want) ? fg_src(fc.host) == fg_src(fresh.host) : fg_src(fc.host) != fg_src(fresh.host), "$fname: the host changes if and only if something is flagged")
        # 1. only the flagged stores differ
        if !isempty(want)
            pt = join(fg_src.(plain.kernels), "\n"); ft = join(fg_src.(fresh.kernels), "\n")
            pl = split(pt, "\n"); fl = split(ft, "\n")
            check(length(pl) == length(fl), "$fname: the same number of lines")
            diffs = [(a, b) for (a, b) in zip(pl, fl) if a != b]
            check(!isempty(diffs) && all(d -> occursin(r"^\s*(\w+)\[(.+)\] = \1\[\2\] \+ ", d[1]) && d[2] == replace(d[1], r"= (\w+)\[(.+)\] \+ " => "= "; count = 1) && any(w -> startswith(strip(d[1]), string(w) * "["), want), diffs), "$fname: $(length(diffs)) changed line(s), each one flagged store")
        end
        execute || continue
        for sz in sizes
            mk = (a, rng) -> kinds[a] == :array_int ? rand(rng, 1:max(1, 6), 6) : STADE.val_randn(rng, fg_len(a, sz, fname))
            rng0 = Random.Xoshiro(7)
            z1 = fg_args(e, kinds, mk, sz; rng = rng0)
            rp = fg_run(plain, fname, z1)
            rf = fg_run(fresh, fname, z1)
            same = all((a isa AbstractArray ? a == b : (a === b || a == b)) for (a, b) in zip(rp.arrays, rf.arrays)) && (rp.ret == rf.ret)
            check(same, "$fname $(sz): from zero gradients the two plans agree")
            check(rp.flagged == 0 && rf.flagged == 0, "$fname $(sz): no ownership violation")
            # 3. non-zero start: a flagged array loses its old value, every other array agrees
            rng1 = Random.Xoshiro(7)
            n1 = fg_args(e, kinds, mk, sz; adj_start = :random, rng = rng1)
            qp = fg_run(plain, fname, n1)
            qf = fg_run(fresh, fname, n1)
            names = Symbol[a for a in e.args[1].args[2:end]]
            ok_other = true; differ = Symbol[]
            for (i, nm) in enumerate(names)
                a, b = qp.arrays[i], qf.arrays[i]
                a isa AbstractArray || continue
                if nm in fresh.fresh
                    a == b || push!(differ, nm)
                else
                    ok_other &= a == b
                end
            end
            check(ok_other, "$fname $(sz): from non-zero gradients the unflagged arrays agree")
            n_active = !isempty(fresh.fresh) && prod(max.(1, collect(values(sz)))) > 0 && !any(==(0), collect(values(sz)))
            n_active && check(sort(differ) == sort(fresh.fresh), "$fname $(sz): a flagged gradient loses its old value ($(differ))")
            # 6. CHECK: execution
            rc = fg_run(fc, fname, z1)
            check(all((a isa AbstractArray ? a == b : (a === b || a == b)) for (a, b) in zip(rf.arrays, rc.arrays)) && rf.ret == rc.ret, "$fname $(sz): from zero gradients the plan with the check agrees")
            fidx = Int[i for (i, nm) in enumerate(names) if nm in fresh.fresh]
            raises(args) = try fg_run(fc, fname, args); "pass" catch ex; ex isa ErrorException && occursin("must be zero on entry", ex.msg) ? "raise" : "other:" * sprint(showerror, ex) end
            nz(arr) = any(i -> any(!iszero, arr[i]) , fidx)
            check(raises(n1) == (nz(n1) ? "raise" : "pass"), "$fname $(sz): a non-zero flagged gradient raises ($(raises(n1)))")
            check(raises(rf.arrays) == (nz(rf.arrays) ? "raise" : "pass"), "$fname $(sz): the gradient a first call left behind raises on a second call without a reset ($(raises(rf.arrays)))")
        end
    end
    if sabotage
        sabs = (("coefficient proof forced", "cgen_fresh_coeff_nonzero(c, params::Set{Symbol}, guard) = true", ["fg_stride"]),
                ("count proof forced", "cgen_fresh_count_positive(guard, s::Symbol) = true", ["fg_stride"]),
                ("zero coefficient ignored", "cgen_split_coeff(e, v::Symbol) = 1", ["fg_tsum"]),
                ("index names forced", "cgen_fresh_index_names_ok(idx, params::Set{Symbol}, loops::Vector{Symbol}, defs::Set{Symbol}) = true", ["fg_hidden"]),
                ("single store forced", "cgen_fresh_single_store(body, p::Symbol, sites::Vector{Any}) = length(sites) >= 1", ["fg_twosites"]),
                ("host group forced", "cgen_fresh_host_group(hs::Vector{Any}, hbody, g::Symbol) = (i = findfirst(st -> cgen_count_symbol(st, g) > 0, hs); i === nothing ? nothing : hs[i])", ["fg_shared"]),
                ("check never raises", "cgen_fresh_check(host::Expr, flagged::Vector{Symbol}) = host", ["fg_dense"]),
                ("only the first array checked", "cgen_fresh_check(host::Expr, flagged::Vector{Symbol}) = (isempty(flagged) ? host : Expr(:function, host.args[1], Expr(:block, Expr(:if, Expr(:&&, Expr(:call, :!, Expr(:call, Expr(:., :Base, QuoteNode(:isempty)), flagged[1])), Expr(:call, :!=, Expr(:call, Expr(:., :Base, QuoteNode(:maximum)), Expr(:., :Base, QuoteNode(:abs)), flagged[1]), 0)), Expr(:call, :error, \"fresh_gradient: the gradient array \" * string(flagged[1]) * \" of \" * string(host.args[1].args[1]) * \" must be zero on entry.\")), host.args[2].args...)))", ["fg_dense"]),
                ("NaN passes", "cgen_fresh_check(host::Expr, flagged::Vector{Symbol}) = (isempty(flagged) ? host : Expr(:function, host.args[1], Expr(:block, [Expr(:if, Expr(:&&, Expr(:call, :!, Expr(:call, Expr(:., :Base, QuoteNode(:isempty)), g)), Expr(:call, :>, Expr(:call, Expr(:., :Base, QuoteNode(:maximum)), Expr(:., :Base, QuoteNode(:abs)), g), 0)), Expr(:call, :error, \"fresh_gradient: the gradient array \" * string(g) * \" of \" * string(host.args[1].args[1]) * \" must be zero on entry.\")) for g in flagged]..., host.args[2].args...)))", Symbol[]),
                ("variant check forced", "cgen_fresh_variant_ok(kd, p::Symbol, stmt) = true", Symbol[]),
                ("top-level launch forced", "cgen_fresh_top_level_launch(grp, kdefs::Dict{Symbol,Expr}, lm) = (names = cgen_split_launched!(grp, lm, Set{Symbol}()); base = first(Set(cgen_split_base_name(n) for n in names)); (names = names, base = base, call = cgen_split_first_call(grp, lm, base)))", ["fg_bias"]))
        for (label, redef, cases) in sabs
            names = isempty(cases) ? ["fg_bias"] : cases
            script = """
                include($(repr(joinpath(@__DIR__, "validate_fresh_gradient.jl"))))
                Core.eval(STADE, Meta.parse($(repr(redef))))
                println("RESULT ", try validate_fresh_gradient(; only = $(repr(names)), sabotage = false) catch; 1 end)
            """
            o = sab_read(script)
            m = match(r"RESULT (\d+)", o)
            check(m !== nothing && parse(Int, m.captures[1]) > 0, "sabotage \"$label\": $(join(names, ", ")) fails a check")
        end
    end
    println("\n", bad == 0 ? "all fresh-gradient checks passed" : "*** $bad fresh-gradient check(s) failed ***")
    return bad
end

# length of the float array `a` for the sizes of a case; the cases use the layouts below
function fg_len(a, sz, fname)
    s = string(a)
    n(k) = getfield(sz, k)
    fname == "fg_dense" && return a == :x ? n(:n) : a == :w ? n(:n) * n(:m) : (a == :b || a == :y) ? n(:m) : error(a)
    fname == "fg_two" && return a == :x ? n(:n) : a == :w1 ? n(:n) * n(:m) : (a == :b1 || a == :h) ? n(:m) : a == :w2 ? n(:m) * n(:k) : (a == :b2 || a == :y) ? n(:k) : error(a)
    fname == "fg_bias" && return n(:n)
    fname == "fg_shared" && return a == :w ? n(:n)^2 : n(:n)
    fname == "fg_tsum" && return a == :x ? n(:n) : a == :w ? n(:m) : n(:m)
    fname == "fg_stride" && return a == :x ? n(:n) : a == :w ? max(1, (n(:n) - 1) * n(:k) + n(:m)) : n(:m)
    fname == "fg_steps" && return n(:n)
    fname == "fg_gather" && return n(:n)
    fname == "fg_hidden" && return a == :x ? n(:n) : a == :w ? (n(:n) + 1) * n(:m) : n(:m)
    fname == "fg_twosites" && return a == :w ? n(:n) * n(:m) : (a == :x || a == :z) ? n(:n) : n(:m)
    fname == "fg_conv" && return a == :x ? n(:n) + n(:nk) : a == :w ? n(:nk) : n(:n)
    error(fname)
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_fresh_gradient(; only = isempty(ARGS) ? nothing : ARGS) == 0 ? 0 : 1)
end
