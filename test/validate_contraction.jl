# ==================== validate_contraction.jl =============================
# Checks the contraction twin (`contraction`, v0.4.34): a tiled kernel for a loop over the elements of a matrix whose inner loop sums products.
#
#   julia validate_contraction.jl                  units, refusals, sabotage
#   validate_contraction(; phases = (:units,))
#
# Every case is a small kernel in the house style, run through the CPU shim of validate_ownership (ownership_shim.jl: blocks of tasks, barriers, shared memory,
# a record of who touched what) twice, with the twin forced on (`contraction = 1`) and with `contraction = 0`, from the same random inputs.
#   - a case that must tile: a twin exists, the two runs agree to 1e-12 (the sum order differs), the shim reports no conflict, and the twin ran;
#   - a case that must not tile: no twin exists and the two plans are the very same text.
# The sizes are not multiples of the 64 x 64 tile or of the 16-wide step of the inner loop, and the grid has several blocks in both directions.
# SABOTAGE edits of the contraction code, each of which one case must catch.

isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

# (name, source, entry, args, must_tile)   must_tile = :plain: a twin is built but the host test keeps the plain kernel
# sizes: n columns, m rows, k inner length
function con_args(r; n = 70, m = 67, k = 19)
    (r = r, n = n, m = m, k = k)
end
const CON_CASES = [
    ("bias product", """function con_bias(x, w, b, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = b[o]
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_bias, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), randn(r, n), zeros(m * n), m, n, k]), true),
    ("residual, tanh and two stores", """function con_res(x, w, b, res, y, z, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = b[o] + res[idx]
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = tanh(s)
            z[idx] = s
        end
        return nothing
    end""", :con_res, sd -> (r = MersenneTwister(sd); n = 66; m = 65; k = 33; Any[randn(r, m * k), randn(r, k * n), randn(r, n), randn(r, m * n), zeros(m * n), zeros(m * n), m, n, k]), true),
    ("subtracted product", """function con_sub(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.5
            for c = 1:k
                s = s - x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_sub, sd -> (r = MersenneTwister(sd); n = 71; m = 66; k = 17; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k]), true),
    ("operands in the other order", """function con_swap(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + w[(c - 1) * n + o] * x[(t - 1) * k + c]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_swap, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 21; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k]), true),
    ("loop starts at a parameter", """function con_lo(x, w, y, m, n, k, c0)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = c0:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_lo, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 25; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k, 4]), true),
    ("one row", """function con_row(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_row, sd -> (r = MersenneTwister(sd); n = 70; m = 1; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k]), true),
    ("same array on both sides", """function con_gram(x, y, m, k)
        for idx = 1:m * m
            o = mod(idx - 1, m) + 1
            t = div(idx - 1, m) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * x[(o - 1) * k + c]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_gram, sd -> (r = MersenneTwister(sd); m = 69; k = 19; Any[randn(r, m * k), zeros(m * m), m, k]), true),
    ("the start value is read from an array of the output index", """function con_start(x, w, b, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = b[idx] * 2.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_start, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), randn(r, m * n), zeros(m * n), m, n, k]), true),
    # ---- refusals ----
    ("left operand depends on the column", """function con_adep(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(o - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_adep, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, max(n, m) * k), randn(r, k * n), zeros(m * n), m, n, k]), false),
    ("right operand depends on the row", """function con_bdep(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o + t]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_bdep, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n + m), zeros(m * n), m, n, k]), false),
    ("a parameter has the loop variable's name and is read after the loop", """function con_cafter(x, w, y, m, n, k, c)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s + c
        end
        return nothing
    end""", :con_cafter, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k, 0.25]), false),
    ("a parameter has the loop variable's name and is read before the loop", """function con_cbefore(x, w, y, m, n, k, c)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = c
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_cbefore, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k, 0.25]), false),
    ("sum of one array, no product", """function con_sum(x, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_sum, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), zeros(m * n), m, n, k]), false),
    ("product of three", """function con_three(x, w, v, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o] * v[c]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_three, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), randn(r, k), zeros(m * n), m, n, k]), false),
    ("a partial last row", """function con_part(x, w, y, m, n, k)
        for idx = 1:m * n + 5
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_part, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, (m + 1) * k), randn(r, k * n), zeros(m * n + 5), m, n, k]), :plain),
    ("a factor that is a scalar", """function con_scal(x, w, y, m, n, k, a)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * a
            end
            y[idx] = s
        end
        return nothing
    end""", :con_scal, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k, 0.5]), false),
    ("a factor that is a function of an element", """function con_fn(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + tanh(x[(t - 1) * k + c]) * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_fn, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k]), false),
    ("two inner loops", """function con_two(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_two, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k]), false),
    ("branch in the inner loop", """function con_branch(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                if x[(t - 1) * k + c] > 0.0
                    s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
                end
            end
            y[idx] = s
        end
        return nothing
    end""", :con_branch, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k]), false),
    ("descending inner loop", """function con_down(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = k:-1:1
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_down, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k]), false),
    ("operand array stored in the kernel", """function con_stored(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            for c = 1:k
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            x[(t - 1) * k + 1] = s
        end
        return nothing
    end""", :con_stored, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k]), false),
    ("matrix-vector loop, no flat index", """function con_mv(w, x, y, n, k)
        for o = 1:n
            s = 0.0
            for c = 1:k
                s = s + w[(c - 1) * n + o] * x[c]
            end
            y[o] = s
        end
        return nothing
    end""", :con_mv, sd -> (r = MersenneTwister(sd); n = 70; k = 19; Any[randn(r, k * n), randn(r, k), zeros(n), n, k]), false),
    ("inner bound computed in the kernel", """function con_kin(x, w, y, m, n, k)
        for idx = 1:m * n
            o = mod(idx - 1, n) + 1
            t = div(idx - 1, n) + 1
            s = 0.0
            kk = k - 2
            for c = 1:kk
                s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
            end
            y[idx] = s
        end
        return nothing
    end""", :con_kin, sd -> (r = MersenneTwister(sd); n = 70; m = 67; k = 19; Any[randn(r, m * k), randn(r, k * n), zeros(m * n), m, n, k]), false),
]

# The recognizer on hand-written kernel text, for the guards that STADE's own gates make unreachable from source: a store into an operand array, and a parameter that has the
# loop variable's name. Each variant must be refused, the control must not.
con_kernel(pre, post; extra = "", first = "idx = 1 + (__tid - 1)") = Meta.parse("""function cuda_kernel_dir!(x, w, y, m, n, k, c$extra)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(m * n - 1, 1) + 1
        return nothing
    end
    $first
    o = mod(idx - 1, n) + 1
    t = div(idx - 1, n) + 1
    $pre
    for c = 1:k
        s = s + x[(t - 1) * k + c] * w[(c - 1) * n + o]
    end
    $post
    return nothing
end""")
const CON_DIRECT = [
    ("control", "s = 0.0", "y[idx] = s", true),
    ("a store into an operand array", "s = 0.0", "x[idx] = s", false),
    ("a store into the other operand array", "s = 0.0", "w[idx] = s", false),
    ("the loop variable's name read after the loop", "s = 0.0", "y[idx] = s + c", false),
    ("the loop variable's name read before the loop", "s = c", "y[idx] = s", false),
    ("the column is changed after its decode", "s = 0.0\n    o = o + 1", "y[idx] = s", false),
    ("the row is changed in the epilogue", "s = 0.0", "t = t + 1\n    y[idx] = s", false),
    ("an atomic store in the epilogue", "s = 0.0", "CUDA.@atomic y[idx] += s", false),
    ("the flat index is not the thread index", "s = 0.0", "y[idx] = s", false),
]
function con_direct_checks()
    bad = 0
    for (name, pre, post, must) in CON_DIRECT
        kd = con_kernel(pre, post; first = startswith(name, "the flat index") ? "idx = 2 + (__tid - 1)" : "idx = 1 + (__tid - 1)")
        t = try STADE.cgen_contraction_kernel(kd) catch e; e end
        ok = must ? (t isa NamedTuple) : t === nothing
        println(rpad(ok ? "ok" : "FAIL", 6), "direct, $name: ", must ? "a twin is built" : "refused")
        ok || (bad += 1)
    end
    return bad
end

function con_init!()
    isdefined(Main, :CUDA) || include(joinpath(@__DIR__, "ownership_shim.jl"))
    isdefined(Main, :CuArray) || Core.eval(Main, :(const CuArray = CUDA.TArray))
    Core.eval(Main, :(using .CUDA))
end

# builds the plan, evaluates its functions into Main, runs the host function; returns (outputs, kernels launched, conflicts, plan)
function con_run(src::String, entry::Symbol, args, contraction::Int; maxt = nothing)
    con_init!()
    return Base.invokelatest(con_run_body, src, entry, args, contraction, maxt)
end
function con_run_body(src::String, entry::Symbol, args, contraction::Int, maxt)
    Main.CUDA.MAXT[] = maxt === nothing ? (name -> 1024) : maxt
    cuda = STADE.cgen_backend_cuda()
    plan = STADE.stade_gpu(Meta.parse(src), cuda; warp_atomics = false, shared_private = false, inbounds = false, split_reduce = 0, contraction = contraction)
    for k in plan.kernels; Core.eval(Main, k); end
    Core.eval(Main, plan.host)
    host = plan.host.args[1].args[1]
    dev = Any[a isa AbstractArray ? Main.CUDA.TArray(copy(a)) : a for a in args]
    empty!(Main.CUDA.LAUNCHED); empty!(Main.CUDA.VIOLATIONS)
    Base.invokelatest(getfield(Main, host), dev...)
    outs = Any[a isa Main.CUDA.TArray ? copy(a.data) : a for a in dev]
    return (outs = outs, launched = copy(Main.CUDA.LAUNCHED), conflicts = copy(Main.CUDA.VIOLATIONS), plan = plan)
end

con_relerr(a, b) = maximum(maximum(abs.(x .- y) ./ max.(abs.(x), abs.(y), 1.0)) for (x, y) in zip(a.outs, b.outs) if x isa AbstractArray; init = 0.0)

function con_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for (name, src, entry, mk, must) in CON_CASES
        try
            args = mk(1)
            a = con_run(src, entry, args, 1)
            b = con_run(src, entry, args, 0)
            twins = count(k -> endswith(string(k.args[1].args[1]), "_t!"), a.plan.kernels)
            ran = any(n -> endswith(n, "_t!"), a.launched)
            if must === :plain
                # a twin exists but the host test sends the launch to the plain kernel (a loop that does not make whole rows)
                check(twins >= 1 && !ran && con_relerr(a, b) == 0.0 && any(x != y for (x, y) in zip(a.outs, args) if x isa AbstractArray), "$name: a twin exists, the plain kernel runs, outputs equal (twins $twins)")
            elseif must
                err = con_relerr(a, b)
                changed = any(x != y for (x, y) in zip(a.outs, args) if x isa AbstractArray)
                check(twins >= 1 && ran && changed && err <= 1e-12 && isempty(a.conflicts), "$name: twins $twins, ran $ran, error $(round(err, sigdigits = 3)), conflicts $(length(a.conflicts))")
            else
                same_text = STADE.io_expr_to_source(a.plan.host) == STADE.io_expr_to_source(b.plan.host) && twins == 0
                check(same_text && !ran, "$name: refused, plan unchanged (twins $twins)")
            end
        catch e
            check(false, "$name: error " * first(split(sprint(showerror, e), "\n")))
        end
    end
    bad += con_direct_checks()
    # thresholds: a size below `contraction` runs the plain kernel; an inner length of 0 does too; the answer is the start value
    try
        name, src, entry, mk, _ = CON_CASES[1]
        args = mk(1)                       # n = 70, m = 67, k = 19
        above = con_run(src, entry, args, 19)
        below = con_run(src, entry, args, 20)
        check(any(n -> endswith(n, "_t!"), above.launched) && !any(n -> endswith(n, "_t!"), below.launched), "$name: the twin runs at the threshold (19) and not below it (20)")
        k0 = Any[args[1:end-1]..., 0]
        z = con_run(src, entry, k0, 1)
        zb = con_run(src, entry, k0, 0)
        check(!any(n -> endswith(n, "_t!"), z.launched) && con_relerr(z, zb) == 0.0, "$name: an inner length of 0 takes the plain kernel")
        lim = con_run(src, entry, args, 1; maxt = nm -> endswith(nm, "_t!") ? 128 : 1024)
        base = con_run(src, entry, args, 0)
        check(!any(n -> endswith(n, "_t!"), lim.launched) && con_relerr(lim, base) == 0.0, "$name: a register limit below 256 threads takes the plain kernel")
    catch e
        check(false, "thresholds: error " * first(split(sprint(showerror, e), "\n")))
    end
    return bad
end

function validate_contraction(; phases = (:units, :sabotage), slice = 1:typemax(Int))
    bad = 0
    if :units in phases
        println("-- units")
        bad += con_unit_checks()
    end
    if :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        text = read(joinpath(abspath(joinpath(@__DIR__, "..")), "src", "STADE.jl"), String)
        cases = [
            ("no barrier after the loads", "Expr(:block, loadA..., loadB..., call(:sync_threads),", "Expr(:block, loadA..., loadB...,"),
            ("no barrier after the products", "Expr(:for, asg(:__ct_kk, call(:(:), 0, TK - 1)), Expr(:block, mac...)),\n                      call(:sync_threads)))", "Expr(:for, asg(:__ct_kk, call(:(:), 0, TK - 1)), Expr(:block, mac...))))"),
            ("the tile is padded with ones", "zfill = asg(:__ct_v, 0.0)", "zfill = asg(:__ct_v, 1.0)"),
            ("the row of a load is not bounded", "Expr(:&&, call(:<, cadd(:__ct_r0, :__ct_rr), :__ct_M), call(:<, cadd(:__ct_k0, :__ct_kk), :__ct_K))", "call(:<, cadd(:__ct_k0, :__ct_kk), :__ct_K)"),
            ("the column of a load is not bounded", "Expr(:&&, call(:<, cadd(:__ct_c0, :__ct_cc), :__ct_N), call(:<, cadd(:__ct_k0, :__ct_kk), :__ct_K))", "call(:<, cadd(:__ct_k0, :__ct_kk), :__ct_K)"),
            ("the inner index of a load is not bounded", "Expr(:&&, call(:<, cadd(:__ct_r0, :__ct_rr), :__ct_M), call(:<, cadd(:__ct_k0, :__ct_kk), :__ct_K))", "call(:<, cadd(:__ct_r0, :__ct_rr), :__ct_M)"),
            ("the outputs are not bounded", "Expr(:if, Expr(:&&, call(:<, :__ct_row, :__ct_M), call(:<, :__ct_col, :__ct_N)), Expr(:block, inner...))", "Expr(:if, true, Expr(:block, inner...))"),
            ("the loop start is ignored", "push!(loadA, asg(cv, cadd(cadd(:__ct_lo, :__ct_k0), :__ct_kk)))", "push!(loadA, asg(cv, cadd(cadd(1, :__ct_k0), :__ct_kk)))"),
            ("a descending loop is accepted", "(lp !== nothing && lp.step == 1 && length(lp.body) == 1) || return nothing", "(lp !== nothing && length(lp.body) == 1) || return nothing"),
            ("operands that depend on the wrong index are accepted", "issubset(va, union(params, Set([tv, cv]))) && issubset(vb, union(params, Set([ov, cv]))) || return nothing", "true"),
            ("an operand of the product that is stored is accepted", "(fa.arr in stored || fb.arr in stored) && return nothing", "false && return nothing"),
            ("the loop variable may be read after the loop", "    for st in post\n        cv in cgen_split_vars(st) && return nothing\n    end\n", ""),
            ("the loop variable may be read before the loop", "    any(cv in cgen_split_vars(st) for st in pre_rest) && return nothing\n", ""),
            ("an inner bound computed in the kernel is accepted", "issubset(cgen_split_vars(lp.lo), params) && issubset(cgen_split_vars(lp.hi), params) || return nothing\n    ac = ", "true\n    ac = "),
            ("a loop after the first is allowed in the epilogue", "        (cgen_split_is_scalar(st) || (cgen_split_is_store(st) && cgen_atomic_site(st) === nothing)) || return nothing", "        true"),
            ("a product test is dropped", "(term isa Expr && term.head === :call && term.args[1] === :* && length(term.args) == 3) || return nothing", "true"),
            ("a factor that is not an array element is accepted", "(f isa Expr && f.head === :ref && length(f.args) == 2 && f.args[1] isa Symbol) || return nothing", "true"),
            ("the column or row may be reassigned", "        count(st -> st isa Expr && st.head === :(=) && st.args[1] === v, Any[pre_rest..., post...]) == 1 || return nothing\n", ""),
            ("the flat index may be offset", "(first_st isa Expr && first_st.head === :(=) && first_st.args[1] isa Symbol && first_st.args[2] == :(1 + (__tid - 1))) || return nothing", "(first_st isa Expr && first_st.head === :(=) && first_st.args[1] isa Symbol) || return nothing"),
            ("the register limit is not checked", "fits = Expr(:call, :>=, Expr(:call, Expr(:., :CUDA, QuoteNode(:maxthreads)), probe), 256)", "fits = true"),
            ("the whole-rows test is dropped", "    cond = Expr(:&&, cond, call(:(==), call(:mod, n_iter_total(n_iter), N), 0))\n", ""),
            ("the size test is dropped", "cond = Expr(:&&, Expr(:&&, Expr(:&&, call(:>=, N, cmin), call(:>=, Mh, cmin)), call(:>=, Kh, cmin)), call(:>, N, 0))", "cond = call(:>, N, 0)"),
            ("the launch asks for too little shared memory", "shmem = 8 * ((TM + 1) * TK + TM * TK))", "shmem = 8 * ((TM + 1) * TK + TM * TK) - 8)"),
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
                    include($(repr(joinpath(@__DIR__, "validate_contraction.jl"))))
                    n = con_unit_checks()
                    println("RESULT ", n)
                """
                out = try read(Base.pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all contraction checks passed" : "*** $bad contraction check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_contraction() == 0 ? 0 : 1)
end
