include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_jacc_reduction()

Check the JACC idiomatic-reduction path: that `JACC.@parallel_reduce` reads the elements the loop
actually visited, and that it cannot be launched with an empty range.

Two defects are gated here, both found on a Tesla V100 during benchmarking and both invisible to every
other script in this suite.

**Wrong elements.** JACC's closure takes the POSITION in `1:range`, not the loop's own value. The two
coincide only for a loop starting at 1 with step 1. `jgen_idiomatic_reduction_value` used to name the
closure's first parameter after the loop variable and leave `arr[loopvar]` as written, so any other
lower bound, stride, or direction silently summed elements `1 .. trip_count`. With `u[i] = 0.5 +
sin(0.1i)`: `2:i_n` at n = 100 returned 90.974668 against CUDA's exact 90.616806, and `3:3:i_n` was out
by 45%. A wrong number, never an error.

**Empty range.** `JACC.@parallel_reduce range = 0` is not an empty sum but an error -- "Grid dimensions
CuDim3(0x0, 0x1, 0x1) are not positive". CUDA's `mapreduce` carries an `init` and returns zero.

Neither is reachable from the corpus at its normal settings: the reduce branch is taken only when the
trip count is at least `CGEN_REDUCTION_THRESHOLD_DEFAULT` (32768) and corpus baselines draw integers
from 3..5, so every corpus run takes the atomic branch instead. The same invisibility hid the
cross-thread write-overlap race. These checks therefore force the branch with `reduction_threshold = 0`.

The index check does not match on syntax. It EVALUATES the emitted index expression for every position
in `1:range` at a concrete `i_n` and compares the resulting index list against `collect(lo:step:hi)`,
in order. A syntactic test ("does the closure still use the bare loop variable") would pass any rewrite
that is merely different; this one passes only a rewrite that is right.

`validate_backend_agreement.jl` owns the corresponding structural property across the whole corpus:
every `@parallel_reduce` must sit inside a `> 0` guard.

No GPU is needed; this is codegen only.
"""

# (name, source, i_n => the range the loop runs). `byte_identical` marks the loops whose emitted closure
# must not change at all, because the position-to-value map is the identity for them.
function vjr_cases()
    return [
        ("red_lo2", """
         function red_lo2(u, loss, i_n)
             for i_x = 2:i_n
                 loss[1] = loss[1] + u[i_x] ^ 2
             end
             return nothing
         end
         """, n -> 2:n, false),
        ("red_step2", """
         function red_step2(u, loss, i_n)
             for i_x = 1:2:i_n
                 loss[1] = loss[1] + u[i_x] ^ 2
             end
             return nothing
         end
         """, n -> 1:2:n, false),
        ("red_rev3", """
         function red_rev3(u, loss, i_n)
             for i_x = i_n:-1:3
                 loss[1] = loss[1] + u[i_x] ^ 2
             end
             return nothing
         end
         """, n -> n:-1:3, false),
        ("red_lo3s3", """
         function red_lo3s3(u, loss, i_n)
             for i_x = 3:3:i_n
                 loss[1] = loss[1] + u[i_x] ^ 2
             end
             return nothing
         end
         """, n -> 3:3:n, false),
        # controls: the position IS the loop value, so these must come out byte-identical
        ("red_unit", """
         function red_unit(u, loss, i_n)
             for i_x = 1:i_n
                 loss[1] = loss[1] + u[i_x] ^ 2
             end
             return nothing
         end
         """, n -> 1:n, true),
        # the shift lives inside the term, not in the bounds -- still a `1:i_n` loop
        ("red_shifted", """
         function red_shifted(u, loss, alpha, beta, i_n)
             for i_x = 1:i_n
                 loss[1] = loss[1] + (alpha * u[i_x] + beta) ^ 2
             end
             return nothing
         end
         """, n -> 1:n, true),
    ]
end

# the `JACC.@parallel_reduce` macrocall anywhere in `e`, or nothing
function vjr_find_reduce(e)
    if e isa Expr
        if e.head === :macrocall && any(a -> a isa Expr && a.head === :. &&
               a.args[2] isa QuoteNode && a.args[2].value === Symbol("@parallel_reduce"), e.args)
            return e
        end
        for a in e.args
            r = vjr_find_reduce(a)
            r === nothing || return r
        end
    end
    return nothing
end

# (range_expr, closure_param, index_expr) from a `@parallel_reduce(range = N, (closure)(args...))`
function vjr_parts(red::Expr)
    rng = nothing
    closure = nothing
    for a in red.args
        a isa Expr || continue
        a.head === :(=) && a.args[1] === :range && (rng = a.args[2])
        if a.head === :call && a.args[1] isa Expr
            inner = a.args[1]
            while inner isa Expr && inner.head === :block && length(inner.args) == 1
                inner = inner.args[1]
            end
            inner isa Expr && inner.head === :-> && (closure = inner)
        end
    end
    (rng === nothing || closure === nothing) && return nothing
    param = closure.args[1].args[1]
    idxs = Any[]
    vjr_collect_indices(closure.args[2], idxs)
    return (rng, param, idxs)
end

function vjr_collect_indices(e, acc)
    if e isa Expr
        if e.head === :ref && length(e.args) == 2
            push!(acc, e.args[2])
        end
        foreach(a -> vjr_collect_indices(a, acc), e.args)
    end
    return acc
end

# true iff a `@parallel_reduce` sits anywhere in `e` that is NOT inside a `... > 0` branch
function vjr_unguarded_reduce(e, guarded = false)
    if e isa Expr
        if e.head === :macrocall && any(a -> a isa Expr && a.head === :. &&
               a.args[2] isa QuoteNode && a.args[2].value === Symbol("@parallel_reduce"), e.args)
            return !guarded
        end
        if e.head === :if && e.args[1] isa Expr && e.args[1].head === :call && e.args[1].args[1] === :>
            vjr_unguarded_reduce(e.args[2], true) && return true
            return length(e.args) > 2 && vjr_unguarded_reduce(e.args[3], guarded)
        end
        return any(a -> vjr_unguarded_reduce(a, guarded), e.args)
    end
    return false
end

function validate_jacc_reduction()
    bad = 0
    checks = 0
    for (name, src, rangeof, byte_identical) in vjr_cases()
        expr = Meta.parse(src)
        ja = STADE.stade_jacc(expr; reduction_threshold = 0)
        cu = STADE.stade_gpu(expr, STADE.cgen_backend_cuda(); reduction_threshold = 0)
        red = vjr_find_reduce(ja.host)

        checks += 1
        if red === nothing
            println(rpad("$(name) [reduce emitted]", 34), " FAIL  no @parallel_reduce in the JACC host")
            bad += 1
            continue
        end
        parts = vjr_parts(red)
        if parts === nothing
            println(rpad("$(name) [reduce shape]", 34), " FAIL  could not read range/closure out of the macrocall")
            bad += 1
            continue
        end
        rng, param, idxs = parts

        # --- the elements actually read, for concrete sizes ---
        for n in (100, 1000, 7)
            checks += 1
            want = collect(rangeof(n))
            trip = Core.eval(Main, Expr(:let, Expr(:(=), :i_n, n), rng))
            got = Any[]
            for k in 1:trip
                for ix in idxs
                    push!(got, Core.eval(Main, Expr(:let, Expr(:block, Expr(:(=), :i_n, n), Expr(:(=), param, k)), ix)))
                end
            end
            # every array in the term is indexed identically, so collapse the per-ref repeats
            per = isempty(idxs) ? 0 : length(idxs)
            got = per <= 1 ? got : got[1:per:end]
            if trip != length(want)
                println(rpad("$(name) [n=$(n) trip]", 34), " FAIL  range=", trip, " but the loop runs ", length(want), " times")
                bad += 1
            elseif got != want
                println(rpad("$(name) [n=$(n) elements]", 34), " FAIL  reads ", first(got, 4), "... want ", first(want, 4), "...")
                bad += 1
            else
                println(rpad("$(name) [n=$(n) elements]", 34), " ok  ", trip, " positions map onto ", first(want, 3), "...")
            end
        end

        # --- the zero-trip guard ---
        checks += 1
        if vjr_unguarded_reduce(ja.host)
            println(rpad("$(name) [zero-trip guard]", 34), " FAIL  @parallel_reduce is not inside a `> 0` branch")
            bad += 1
        else
            println(rpad("$(name) [zero-trip guard]", 34), " ok")
        end

        # --- the identity case must not be rewritten at all ---
        if byte_identical
            checks += 1
            if param !== :i_x || any(ix -> ix !== :i_x, idxs)
                println(rpad("$(name) [unchanged]", 34), " FAIL  a `1:n` step-1 loop was rewritten: param=", param, " idx=", idxs)
                bad += 1
            else
                println(rpad("$(name) [unchanged]", 34), " ok  closure still `(i_x, ...) -> ... u[i_x] ...`")
            end
        end

        # --- CUDA must be untouched by any of this: it reduces over an explicit view ---
        checks += 1
        csrc = STADE.io_expr_to_source(cu.host)
        if !occursin("view(", csrc)
            println(rpad("$(name) [cuda view]", 34), " FAIL  CUDA host lost its explicit reduction view")
            bad += 1
        else
            println(rpad("$(name) [cuda view]", 34), " ok")
        end
    end

    println("\n", checks - bad, "/", checks, " checks passed",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end

validate_jacc_reduction()
