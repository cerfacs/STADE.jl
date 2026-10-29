# ==================== validate_split.jl =============================
# Checks the split reduction (`split_reduce`, v0.4.33): the twin kernel that gives each output L lanes on its inner loops.
#
#   julia validate_split.jl                        units, refusals, sabotage
#   validate_split(; phases = (:units,))
#
# Every case is a small kernel in the house style. Each one is run through the CPU shim of validate_ownership (ownership_shim.jl: blocks of tasks,
# barriers, a record of who touched what) twice, with the twin forced on and with `split_reduce = 0`, from the same random inputs.
#   - a case that must split: a twin exists, the two runs agree to 1e-12 (the sum order differs), the shim reports no conflict between threads
#     that no barrier orders, and the twin actually ran;
#   - a case that must not split: no twin exists, and the two plans are the very same text.
# The runs use sizes that are not multiples of 32 (a partial last block) and one case whose weight stride is 0 at run time, where the host test
# must send the launch to the plain kernel.
# SABOTAGE edits of the split code, each of which one case must catch. One guard has no sabotage entry: the test that a plain-stored array
# appears nowhere outside its own loop (`n_all == inside`). Every shape tried to reach it (a second update loop reading the array at another
# index, a reduction reading it before the store, a shifted index) is already refused by the plain-write gate that builds the kernel, or is
# race-free behind the barriers, so with the guard removed no case changes. It stays as a second line of defense and is not claimed as tested.

isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

# (name, source, entry, args, must_split)
# args: a function seed -> Vector of arguments; arrays are Float64 vectors, integers stay integers
const SPL_CASES = [
    ("matvec with tanh", """function spl_mv(w, x, b, y, n, m)
        for j = 1:n
            s = b[j]
            for i = 1:m
                s += w[(i - 1) * n + j] * x[i]
            end
            y[j] = tanh(s)
        end
        return nothing
    end""", :spl_mv, sd -> (r = MersenneTwister(sd); n = 37; m = 19; Any[randn(r, n * m), randn(r, m), randn(r, n), zeros(n), n, m]), true),
    ("subtracted sum", """function spl_sub(w, x, y, n, m)
        for j = 1:n
            s = 1.5
            for i = 1:m
                s -= w[(i - 1) * n + j] * x[i]
            end
            y[j] = s
        end
        return nothing
    end""", :spl_sub, sd -> (r = MersenneTwister(sd); n = 33; m = 11; Any[randn(r, n * m), randn(r, m), zeros(n), n, m]), true),
    ("two accumulators in one loop", """function spl_two(x, y, n, m)
        for j = 1:n
            s = 0.0
            q = 0.0
            for i = 1:m
                a = x[(j - 1) * m + i]
                s += a
                q += a * a
            end
            y[j] = q / m - (s / m) * (s / m)
        end
        return nothing
    end""", :spl_two, sd -> (r = MersenneTwister(sd); n = 40; m = 13; Any[randn(r, n * m), zeros(n), n, m]), true),
    ("two reduction loops", """function spl_loops(x, y, n, m)
        for j = 1:n
            s = 0.0
            for i = 1:m
                s += x[(j - 1) * m + i]
            end
            q = 0.5
            for i = 1:m
                q += x[(j - 1) * m + i] * s
            end
            y[j] = q
        end
        return nothing
    end""", :spl_loops, sd -> (r = MersenneTwister(sd); n = 35; m = 9; Any[randn(r, n * m), zeros(n), n, m]), true),
    ("descending inner loop", """function spl_rev(w, x, y, n, m)
        for j = 1:n
            s = 0.0
            for i = m:-1:1
                s += w[(i - 1) * n + j] * x[i]
            end
            y[j] = s
        end
        return nothing
    end""", :spl_rev, sd -> (r = MersenneTwister(sd); n = 34; m = 17; Any[randn(r, n * m), randn(r, m), zeros(n), n, m]), true),
    ("reduction, epilogue and update loop", """function spl_upd(w, x, g, gx, b, n, m)
        for j = 1:n
            s = b[j]
            for i = 1:m
                s += w[(i - 1) * n + j] * x[i]
            end
            t = tanh(s)
            for i = 1:m
                g[(i - 1) * n + j] = g[(i - 1) * n + j] + x[i] * t
                gx[i] = gx[i] + w[(i - 1) * n + j] * t
            end
        end
        return nothing
    end""", :spl_upd, sd -> (r = MersenneTwister(sd); n = 37; m = 21; Any[randn(r, n * m), randn(r, m), randn(r, n * m), randn(r, m), randn(r, n), n, m]), true),
    ("weight stride zero at run time", """function spl_ld(w, x, g, b, n, m, ld)
        for j = 1:n
            s = b[j]
            for i = 1:m
                s += w[j] * x[i]
            end
            for i = 1:m
                g[(i - 1) * ld + j] = x[i] * s
            end
        end
        return nothing
    end""", :spl_ld, sd -> (r = MersenneTwister(sd); n = 35; m = 6; Any[randn(r, n), randn(r, m), zeros(n * m), randn(r, n), n, m, n]), true),
    ("store, then reread it in a later reduction", """function spl_reread2(w, x, y, z, n, m)
        for j = 1:n
            s = 0.0
            for i = 1:m
                s += w[(i - 1) * n + j] * x[i]
            end
            y[j] = tanh(s)
            q = 0.0
            for i = 1:m
                q += w[(i - 1) * n + j] * y[j]
            end
            z[j] = q
        end
        return nothing
    end""", :spl_reread2, sd -> (r = MersenneTwister(sd); n = 37; m = 15; Any[randn(r, n * m), randn(r, m), zeros(n), zeros(n), n, m]), true),
    # ---- refusals ----
    ("loop temporary read after the loop", """function spl_tempafter(x, y, n, m)
        for j = 1:n
            s = 0.0
            a = 0.0
            for i = 1:m
                a = x[(j - 1) * m + i]
                s += a
            end
            y[j] = s + a
        end
        return nothing
    end""", :spl_tempafter, sd -> (r = MersenneTwister(sd); n = 35; m = 9; Any[randn(r, n * m), zeros(n), n, m]), false),
    ("second update loop reads the first one's array at another index", """function spl_twoupd(w, x, y, g, n, m)
        for j = 1:n
            s = 0.0
            for i = 1:m
                s += w[(i - 1) * n + j] * x[i]
            end
            t = tanh(s)
            for i = 1:m
                y[(i - 1) * n + j] = x[i] * t
            end
            for i = 1:m
                g[(i - 1) * n + j] = y[(m - i) * n + j]
            end
        end
        return nothing
    end""", :spl_twoupd, sd -> (r = MersenneTwister(sd); n = 33; m = 8; Any[randn(r, n * m), randn(r, m), zeros(n * m), zeros(n * m), n, m]), false),
    ("carried scalar that is not a sum", """function spl_carry(x, y, n, m)
        for j = 1:n
            s = 0.0
            for i = 1:m
                s = s * 0.9 + x[(j - 1) * m + i]
            end
            y[j] = s
        end
        return nothing
    end""", :spl_carry, sd -> (r = MersenneTwister(sd); n = 33; m = 9; Any[randn(r, n * m), zeros(n), n, m]), false),
    ("running sum feeding a second sum", """function spl_prefix(x, y, n, m)
        for j = 1:n
            s = 0.0
            q = 0.0
            for i = 1:m
                s += x[(j - 1) * m + i]
                q += s
            end
            y[j] = q
        end
        return nothing
    end""", :spl_prefix, sd -> (r = MersenneTwister(sd); n = 33; m = 9; Any[randn(r, n * m), zeros(n), n, m]), false),
    ("integer counter", """function spl_count(x, y, n, m)
        for j = 1:n
            c = 0
            for i = 1:m
                c += 1
            end
            y[j] = x[j] * c
        end
        return nothing
    end""", :spl_count, sd -> (r = MersenneTwister(sd); n = 33; m = 9; Any[randn(r, n), zeros(n), n, m]), false),
    ("branch in the inner loop", """function spl_branch(x, y, n, m)
        for j = 1:n
            s = 0.0
            for i = 1:m
                if x[(j - 1) * m + i] > 0.0
                    s += x[(j - 1) * m + i]
                end
            end
            y[j] = s
        end
        return nothing
    end""", :spl_branch, sd -> (r = MersenneTwister(sd); n = 33; m = 9; Any[randn(r, n * m), zeros(n), n, m]), false),
    ("triangular inner bound", """function spl_tri(x, y, n)
        for j = 1:n
            s = 0.0
            m = n - j
            for i = 1:m
                s += x[i]
            end
            y[j] = s
        end
        return nothing
    end""", :spl_tri, sd -> (r = MersenneTwister(sd); n = 36; Any[randn(r, n), zeros(n), n]), false),
    ("last store wins", """function spl_last(x, y, n, m)
        for j = 1:n
            for i = 1:m
                y[j] = x[i] * 2.0
            end
        end
        return nothing
    end""", :spl_last, sd -> (r = MersenneTwister(sd); n = 33; m = 7; Any[randn(r, m), zeros(n), n, m]), false),
    ("array stored in the loop read elsewhere", """function spl_reread(w, x, y, z, n, m)
        for j = 1:n
            s = 0.0
            for i = 1:m
                s += w[(i - 1) * n + j] * x[i]
            end
            for i = 1:m
                y[(i - 1) * n + j] = x[i] * s
            end
            z[j] = y[j] + s
        end
        return nothing
    end""", :spl_reread, sd -> (r = MersenneTwister(sd); n = 33; m = 7; Any[randn(r, n * m), randn(r, m), zeros(n * m), zeros(n), n, m]), false),
]

const SPL_INIT = Ref(false)
function spl_init!()
    SPL_INIT[] && return
    include(joinpath(@__DIR__, "ownership_shim.jl"))
    Core.eval(Main, :(using .CUDA))
    Core.eval(Main, :(const CuArray = CUDA.TArray))
    SPL_INIT[] = true
end

# builds the plan, evaluates its functions into Main, runs the host function; returns (outputs, kernels launched, conflicts, plan)
function spl_run(src::String, entry::Symbol, args, split::Int; maxt = nothing)
    spl_init!()
    return Base.invokelatest(spl_run_body, src, entry, args, split, maxt)
end
function spl_run_body(src::String, entry::Symbol, args, split::Int, maxt)
    Main.CUDA.MAXT[] = maxt === nothing ? (name -> 1024) : maxt
    cuda = STADE.cgen_backend_cuda()
    plan = STADE.stade_gpu(Meta.parse(src), cuda; warp_atomics = false, shared_private = false, inbounds = false,
                           split_reduce = split, split_min_inner = 0, split_max_out = 10^9)
    for k in plan.kernels; Core.eval(Main, k); end
    Core.eval(Main, plan.host)
    host = plan.host.args[1].args[1]
    dev = Any[a isa AbstractArray ? Main.CUDA.TArray(copy(a)) : a for a in args]
    empty!(Main.CUDA.LAUNCHED); empty!(Main.CUDA.VIOLATIONS)
    Base.invokelatest(getfield(Main, host), dev...)
    outs = Any[a isa Main.CUDA.TArray ? copy(a.data) : a for a in dev]
    return (outs = outs, launched = copy(Main.CUDA.LAUNCHED), conflicts = copy(Main.CUDA.VIOLATIONS), plan = plan)
end

function spl_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for (name, src, entry, mk, must) in SPL_CASES
        try
            args = mk(1)
            a = spl_run(src, entry, args, 4)
            b = spl_run(src, entry, args, 0)
            twins = count(k -> endswith(string(k.args[1].args[1]), "_s!"), a.plan.kernels)
            ran = any(n -> endswith(n, "_s!"), a.launched)
            if must
                err = maximum(maximum(abs.(x .- y) ./ max.(abs.(x), abs.(y), 1.0)) for (x, y) in zip(a.outs, b.outs) if x isa AbstractArray; init = 0.0)
                ld0 = name == "weight stride zero at run time"
                if ld0
                    # the stride is the loop's `ld`; with ld = 0 the host test must choose the plain kernel
                    a0 = spl_run(src, entry, Any[args[1:end-1]..., 0], 4)
                    b0 = spl_run(src, entry, Any[args[1:end-1]..., 0], 0)
                    same0 = all(x == y for (x, y) in zip(a0.outs, b0.outs) if x isa AbstractArray)
                    check(twins >= 1 && !any(n -> endswith(n, "_s!"), a0.launched) && same0 && isempty(a0.conflicts), "$name: stride 0 at run time takes the plain kernel (twins $twins)")
                    check(ran && err <= 1e-12 && isempty(a.conflicts), "$name: stride nonzero runs the twin, error $(round(err, sigdigits = 3)), conflicts $(length(a.conflicts))")
                else
                    check(twins >= 1 && ran && err <= 1e-12 && isempty(a.conflicts), "$name: twins $twins, ran $ran, error $(round(err, sigdigits = 3)), conflicts $(length(a.conflicts))")
                end
            else
                same_text = STADE.io_expr_to_source(a.plan.host) == STADE.io_expr_to_source(b.plan.host) && twins == 0
                check(same_text && !ran, "$name: refused, plan unchanged (twins $twins)")
            end
        catch e
            check(false, "$name: error " * first(split(sprint(showerror, e), "\n")))
        end
    end
    # the block of 32 x L threads must fit the twin's register limit: a limit below it sends the launch to the plain kernel
    try
        name, src, entry, mk, _ = SPL_CASES[1]
        args = mk(1)
        lim = spl_run(src, entry, args, 4; maxt = nm -> endswith(nm, "_s!") ? 64 : 1024)
        base = spl_run(src, entry, args, 0)
        same = all(x == y for (x, y) in zip(lim.outs, base.outs) if x isa AbstractArray)
        check(!any(n -> endswith(n, "_s!"), lim.launched) && same, "$name: a register limit below 32 x L takes the plain kernel")
        ok4 = spl_run(src, entry, args, 4; maxt = nm -> endswith(nm, "_s!") ? 128 : 1024)
        check(any(n -> endswith(n, "_s!"), ok4.launched), "$name: a limit of exactly 32 x L runs the twin")
    catch e
        check(false, "register limit: error " * first(split(sprint(showerror, e), "\n")))
    end
    return bad
end

function validate_split(; phases = (:units, :sabotage), slice = 1:typemax(Int))
    bad = 0
    if :units in phases
        println("-- units")
        bad += spl_unit_checks()
    end
    if :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        text = read(joinpath(abspath(joinpath(@__DIR__, "..")), "src", "STADE.jl"), String)
        cases = [
            ("no barrier between groups", "prev !== nothing && g !== prev && push!(body, Expr(:call, :sync_threads))", "false && push!(body, Expr(:call, :sync_threads))"),
            ("no barrier after the partial sums", "            push!(body, Expr(:call, :sync_threads))\n            for (k, (v, p)) in enumerate(accs)\n                terms", "            for (k, (v, p)) in enumerate(accs)\n                terms"),
            ("lane stride is wrong", "Expr(:call, :(:), first_, lp.step * L, lp.hi)", "Expr(:call, :(:), first_, lp.step * (L - 1), lp.hi)"),
            ("partial sum starts from the pre-value", "push!(body, Expr(:(=), p, Expr(:call, :zero, v)))", "push!(body, Expr(:(=), p, v))"),
            ("every lane runs the top-level stores", "Expr(:if, Expr(:&&, act, Expr(:call, :(==), ly, 1)), Expr(:block, s))", "Expr(:if, act, Expr(:block, s))"),
            ("every lane starts at the first iteration", "first_ = lp.step == 1 ? Expr(:call, :+, lp.lo, Expr(:call, :-, ly, 1)) : Expr(:call, :-, lp.lo, Expr(:call, :-, ly, 1))", "first_ = lp.lo"),
            ("the stride test is dropped", "        cond = Expr(:&&, cond, Expr(:call, :!=, c, 0))", "        cond = cond"),
            ("a store index without the loop variable is accepted", "(c === nothing || c === 0) && return nothing", "c === nothing && return nothing"),
            ("an accumulator may feed another statement", "                            v in cgen_split_vars(st2.args[2]) && return nothing", "                            false && return nothing"),
            ("a loop temporary may be read after the loop", "                        v in cgen_split_vars(st.args[2]) && return nothing\n                        (v in later) && return nothing\n                        push!(defined, v)\n                    end\n                    for r in cgen_split_vars(st.args[2])\n                        (r in assigned && !haskey(accs, r) && !(r in defined)) && return nothing", "                        v in cgen_split_vars(st.args[2]) && return nothing\n                        push!(defined, v)\n                    end\n                    for r in cgen_split_vars(st.args[2])\n                        (r in assigned && !haskey(accs, r) && !(r in defined)) && return nothing"),
            ("an integer start is accepted", "(items[k][2].args[2] isa Integer) && return nothing", "false && return nothing"),
            ("an inner bound computed in the kernel is accepted", "issubset(cgen_split_vars(lp.lo), params) && issubset(cgen_split_vars(lp.hi), params) || return nothing", "true"),
            ("the register limit is not checked", "fits = Expr(:call, :>=, Expr(:call, Expr(:., :CUDA, QuoteNode(:maxthreads)), probe), 32 * t.lanes)", "fits = true"),
            ("the clamp of the thread index is dropped", "Expr(:(=), :__tid, Expr(:call, :min, :__tidr, n_iter))]", "Expr(:(=), :__tid, :__tidr)]"),
            ("update loops run for inactive threads", "push!(body, Expr(:if, act, Expr(:block, lane_loop(lp, lp.body))))", "push!(body, lane_loop(lp, lp.body))"),
        ]
        mktempdir() do tmp
            for (label, old, new) in cases[intersect(eachindex(cases), slice)]
                n = length(findall(old, text))
                if n != 1
                    check(false, "sabotage \"$label\": anchor found $n times")
                    continue
                end
                path = joinpath(tmp, "STADE_sab.jl")
                write(path, replace(text, old => new; count = 1))
                script = """
                    include($(repr(path)))
                    cd($(repr(@__DIR__)))
                    include($(repr(joinpath(@__DIR__, "validate_split.jl"))))
                    n = spl_unit_checks()
                    println("RESULT ", n)
                """
                out = try read(Base.pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all split reduction checks passed" : "*** $bad split reduction check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_split() == 0 ? 0 : 1)
end
