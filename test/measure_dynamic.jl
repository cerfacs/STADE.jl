include(joinpath(ARGS[1], "src", "STADE.jl"))
using Random

# Counts float operations ACTUALLY EXECUTED by a generated tangent or adjoint, rather than counted
# once per line in the source. Every claim made about this pass so far has rested on a static count,
# which weights a saving inside a triply nested loop the same as one at the top level -- fine for
# ordering two variants against each other, useless for saying what the pass is worth.
#
# Each leaf statement is prefixed with an increment of its own static float-op cost, so the running
# total is exactly sum over executed statements of (ops in that statement). An `if` condition is
# charged before the branch, where it is evaluated once per arrival. Integer index arithmetic is
# excluded, matching what the pass is allowed to touch.

const OPS = Ref(0)

is_float_op(op) = op in (:+, :-, :*, :/, :^, :sqrt, :exp, :log, :sin, :cos, :tan, :tanh,
                         :abs, :max, :min, :inv)
# ALL_OPS=true counts every arithmetic node including integer index math, which the CSE pass is
# forbidden to touch; the gap between the two totals is the size of the prize this pass cannot reach.
const ALL_OPS = get(ENV, "ALL_OPS", "0") == "1"

# Ops in one expression, skipping anything under an array subscript (index arithmetic is Int).
function static_ops(e)
    e isa Expr || return 0
    if e.head == :call
        n = (e.args[1] isa Symbol && (ALL_OPS || is_float_op(e.args[1]))) ? 1 : 0
        return n + sum(Int[static_ops(a) for a in e.args[2:end]]; init = 0)
    elseif e.head == :ref
        return ALL_OPS ? sum(Int[static_ops(a) for a in e.args[2:end]]; init = 0) : 0
    elseif e.head == :(=)
        lhs = e.args[1]
        idx = lhs isa Expr && lhs.head == :ref ? 0 : 0
        return static_ops(e.args[2]) + idx
    end
    return sum(Int[static_ops(a) for a in e.args]; init = 0)
end

bump(n) = n == 0 ? nothing : Expr(:call, :+=, :(Main.OPS[]), n)

function instrument(stmts)
    out = Any[]
    for s in stmts
        s isa LineNumberNode && continue
        if s isa Expr && s.head == :for
            push!(out, Expr(:for, s.args[1], Expr(:block, instrument(s.args[2].args)...)))
        elseif s isa Expr && s.head == :if
            c = static_ops(s.args[1])
            c > 0 && push!(out, :(Main.OPS[] += $c))
            args = Any[s.args[1]]
            for a in s.args[2:end]
                push!(args, a isa Expr && a.head == :block ? Expr(:block, instrument(a.args)...) :
                            Expr(:block, instrument(Any[a])...))
            end
            push!(out, Expr(:if, args...))
        else
            n = static_ops(s)
            n > 0 && push!(out, :(Main.OPS[] += $n))
            push!(out, s)
        end
    end
    return out
end

instrument_fn(f) = Expr(:function, f.args[1], Expr(:block, instrument(f.args[2].args)...))

dir = joinpath(ARGS[1], "test", "val-corpus")
label = ARGS[2]
rows = Tuple{String,Int}[]

for f in sort(filter(f -> endswith(f, ".jl") &&
                          !(endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl")),
                     readdir(dir)))
    name = splitext(f)[1]
    path = joinpath(dir, f)
    yp = STADE.io_default_yaml_path(path)
    STADE.io_path_exists(yp) || STADE.stade_generate_baseline_file(path; yaml_path = yp, int_lo = 3, int_hi = 5)
    baseline = STADE.io_read_baseline_yaml(yp)
    primal = STADE.io_read_corpus_entry(path)
    kernel = STADE.parse_kernel(primal)
    STADE.val_coerce_int_arrays!(kernel, baseline.values)

    Random.seed!(hash(name))
    seed = STADE.val_random_values_like(kernel, baseline.values)

    # tangent
    try
        tan = STADE.stade_tangent(primal)
        fn = STADE.val_compile(instrument_fn(tan))
        OPS[] = 0
        STADE.val_call_tangent(fn, kernel, baseline.int_args, deepcopy(baseline.values), seed)
        push!(rows, (name * " [tangent]", OPS[]))
    catch e
        push!(rows, (name * " [tangent]", -1))
    end

    # adjoint
    try
        adj = STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true)
        fn = STADE.val_compile(instrument_fn(adj.adjoint))
        init = STADE.val_compile(adj.initstacks)
        OPS[] = 0
        STADE.val_call_adjoint(fn, init, kernel, baseline.int_args, deepcopy(baseline.values), seed;
                               stack_arg_names = STADE.val_def_arg_names(adj.initstacks))
        push!(rows, (name * " [adjoint]", OPS[]))
    catch e
        println("  adjoint failed for ", name, ": ", e)
        push!(rows, (name * " [adjoint]", -1))
    end
end

open("/tmp/dyn_" * label * ".txt", "w") do io
    for r in rows
        println(io, r[1], "\t", r[2])
    end
end
ok = [r for r in rows if r[2] >= 0]
println(label, ": ", length(ok), "/", length(rows), " measured, total executed float ops = ",
        sum(r[2] for r in ok))
