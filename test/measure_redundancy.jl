include(joinpath(ARGS[1], "src", "STADE.jl"))
using Random

# Sizes the work the CSE pass is currently forbidden to do, in EXECUTED operations rather than
# source lines. The float gate in emit_cse_touches_float exists because binding integer index
# arithmetic hid the thread variable from cgen_ and cost four mpnn loops their offload. That gate is
# also why unet -- 77% of the corpus's executed work -- gains nothing from the pass. Before widening
# it, the question is how much redundant integer arithmetic actually runs, and how much of that
# could be named WITHOUT touching a loop variable.
#
# Method: value-numbering with kills over each straight-line run (identical bookkeeping to
# emit_cse_run), counting occurrences that a local CSE could have reused. Each run is then charged
# its redundancy once per execution, by an inserted counter, so the totals are dynamic.
#
# Three buckets:
#   float  -- what the pass already removes. Should be near zero on the current revision; a large
#             number here would mean the pass is missing its own targets.
#   int_safe -- redundant integer arithmetic mentioning NO enclosing loop variable. Naming these
#             cannot hide a thread variable, so this is the subset a widened gate could take safely.
#   int_var  -- redundant integer arithmetic that does mention a loop variable. Off limits.

const R_FLOAT = Ref(0)
const R_INT_SAFE = Ref(0)
const R_INT_VAR = Ref(0)

reads!(e, acc) = (e isa Symbol && push!(acc, e);
                  e isa Expr && e.head == :ref && e.args[1] isa Symbol && push!(acc, e.args[1]);
                  e isa Expr && foreach(a -> reads!(a, acc),
                                        e.args[((e.head == :call || e.head == :ref) ? 2 : 1):end]); acc)

pure_ops() = Set(keys(STADE.der_table()))

candidate(e, pure) = e isa Expr &&
    ((e.head == :call && !isempty(e.args) && e.args[1] isa Symbol && e.args[1] in pure) ||
     (e.head == :ref && !isempty(e.args) && e.args[1] isa Symbol))

touches_float(e, floats) = any(v in floats for v in reads!(e, Set{Symbol}()))
touches_loopvar(e, loopvars) = any(v in loopvars for v in reads!(e, Set{Symbol}()))

key_of(e, gen, barrier) = (e, barrier[],
    Tuple(sort([(v, get(gen, v, 0)) for v in reads!(e, Set{Symbol}())]; by = first)))

function bump_stacks!(e, gen)
    if e isa Expr
        e.head == :call && length(e.args) >= 2 && e.args[1] in (:push!, :pop!) &&
            e.args[2] isa Symbol && (gen[e.args[2]] = get(gen, e.args[2], 0) + 1)
        foreach(a -> bump_stacks!(a, gen), e.args)
    end
end

function apply_writes!(s, gen, barrier)
    bump_stacks!(s, gen)
    if s isa Expr && s.head == :(=)
        lhs = s.args[1]
        v = lhs isa Symbol ? lhs : (lhs isa Expr && lhs.head == :ref ? lhs.args[1] : nothing)
        v !== nothing && (gen[v] = get(gen, v, 0) + 1; return nothing)
    end
    barrier[] += 1
    return nothing
end

function scan_expr!(e, gen, barrier, seen, tally, pure, floats, loopvars)
    if candidate(e, pure)
        k = key_of(e, gen, barrier)
        if k in seen
            b = touches_float(e, floats) ? :float : (touches_loopvar(e, loopvars) ? :int_var : :int_safe)
            tally[b] += 1
        else
            push!(seen, k)
        end
    end
    if e isa Expr
        for a in e.args[((e.head == :call || e.head == :ref) ? 2 : 1):end]
            scan_expr!(a, gen, barrier, seen, tally, pure, floats, loopvars)
        end
    end
end

# Redundancy of one straight-line run, as a (float, int_safe, int_var) triple.
function run_tally(run, pure, floats, loopvars)
    gen = Dict{Symbol,Int}(); barrier = Ref(0); seen = Set{Any}()
    tally = Dict(:float => 0, :int_safe => 0, :int_var => 0)
    for s in run
        if s isa Expr && s.head == :(=)
            scan_expr!(s.args[2], gen, barrier, seen, tally, pure, floats, loopvars)
            s.args[1] isa Expr && for a in s.args[1].args[2:end]
                scan_expr!(a, gen, barrier, seen, tally, pure, floats, loopvars)
            end
        else
            scan_expr!(s, gen, barrier, seen, tally, pure, floats, loopvars)
        end
        apply_writes!(s, gen, barrier)
    end
    return (tally[:float], tally[:int_safe], tally[:int_var])
end

function counter_stmts(t)
    out = Any[]
    t[1] > 0 && push!(out, :(Main.R_FLOAT[] += $(t[1])))
    t[2] > 0 && push!(out, :(Main.R_INT_SAFE[] += $(t[2])))
    t[3] > 0 && push!(out, :(Main.R_INT_VAR[] += $(t[3])))
    return out
end

function instrument(stmts, pure, floats, loopvars)
    out = Any[]; run = Any[]
    flush!() = (append!(out, counter_stmts(run_tally(run, pure, floats, loopvars))); append!(out, run); empty!(run))
    for s in stmts
        s isa LineNumberNode && continue
        if s isa Expr && s.head == :for
            flush!()
            v = s.args[1].args[1]
            push!(out, Expr(:for, s.args[1],
                            Expr(:block, instrument(s.args[2].args, pure, floats, union(loopvars, Set([v])))...)))
        elseif s isa Expr && s.head == :if
            flush!()
            args = Any[s.args[1]]
            for a in s.args[2:end]
                push!(args, Expr(:block, instrument(a isa Expr && a.head == :block ? a.args : Any[a],
                                                    pure, floats, loopvars)...))
            end
            push!(out, Expr(:if, args...))
        else
            push!(run, s)
        end
    end
    flush!()
    return out
end

instrument_fn(f, pure, floats) =
    Expr(:function, f.args[1], Expr(:block, instrument(f.args[2].args, pure, floats, Set{Symbol}())...))

# Float-valued names, from kinds plus the float stacks. Branch and trip-count stacks hold Int64, and
# the Tier B prefix/val tables are integer offsets, so all three stay out.
function float_names(sig, body)
    names = Set{Symbol}()
    for (v, k) in sig.kinds
        k in (:scalar_float, :array_float) || continue
        for nm in (v, Symbol(string(v) * "b"), Symbol(string(v) * "d"))
            push!(names, nm); push!(names, Symbol(string(nm) * "d"))
        end
    end
    for nm in collect_arrays(body)
        s = string(nm)
        endswith(s, "_stack") || endswith(s, "_stack_d") || continue
        (startswith(s, "branch_") || startswith(s, "tripcount_") ||
         startswith(s, "prefix_") || startswith(s, "val_")) && continue
        push!(names, nm)
    end
    return names
end

function collect_arrays(e, acc = Set{Symbol}())
    if e isa Expr
        e.head == :ref && e.args[1] isa Symbol && push!(acc, e.args[1])
        foreach(a -> collect_arrays(a, acc), e.args)
    end
    return acc
end

dir = joinpath(ARGS[1], "test", "val-corpus")
pure = pure_ops()
rows = Tuple{String,Int,Int,Int}[]

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

    for mode in (:tangent, :adjoint)
        try
            if mode == :tangent
                body = STADE.stade_tangent(primal)
                fl = float_names(kernel.sig, body)
                fn = STADE.val_compile(instrument_fn(body, pure, fl))
                R_FLOAT[] = 0; R_INT_SAFE[] = 0; R_INT_VAR[] = 0
                STADE.val_call_tangent(fn, kernel, baseline.int_args, deepcopy(baseline.values), seed)
            else
                adj = STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true)
                fl = float_names(kernel.sig, adj.adjoint)
                fn = STADE.val_compile(instrument_fn(adj.adjoint, pure, fl))
                init = STADE.val_compile(adj.initstacks)
                R_FLOAT[] = 0; R_INT_SAFE[] = 0; R_INT_VAR[] = 0
                STADE.val_call_adjoint(fn, init, kernel, baseline.int_args, deepcopy(baseline.values), seed;
                                       stack_arg_names = STADE.val_def_arg_names(adj.initstacks))
            end
            push!(rows, (name * " [" * string(mode) * "]", R_FLOAT[], R_INT_SAFE[], R_INT_VAR[]))
        catch e
            println("  failed: ", name, " [", mode, "] ", e)
        end
    end
end

tf = sum(r[2] for r in rows); ts = sum(r[3] for r in rows); tv = sum(r[4] for r in rows)
println("\nexecuted redundant operations across ", length(rows), " (kernel, mode) pairs")
println("  float (pass already removes these) : ", tf)
println("  int, no loop variable  (SAFE)      : ", ts)
println("  int, mentions a loop variable      : ", tv)
sort!(rows, by = r -> -(r[3] + r[4]))
println("\ntop kernels by redundant integer work:")
for r in rows[1:min(10, end)]
    println("  ", rpad(r[1], 28), " float ", lpad(r[2], 7), "  int_safe ", lpad(r[3], 8), "  int_var ", lpad(r[4], 8))
end
