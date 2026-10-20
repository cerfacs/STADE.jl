include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    scan_if_rewrite(paths...)

List kernels whose adjoint or HVP v0.4.5 and earlier generated wrongly because of the branch-scalar hoist: a scalar assigned in an
`if` arm that a later statement of the same block writes again. `agen_backward_body` used to recompute such a scalar from the branch
flag at the top of the block, which overwrote the later write's value for the reads after it. The tangent was never affected.

Exact with respect to the old guard: a kernel is reported when a variable that an `if` arm assigns purely because it is read elsewhere
(the set the hoist selected) is assigned again by a later statement of the same block. 0.4.10 removed the hoist, so the selection logic
now lives in this file. Each argument is a kernel file or a directory of them.
A reported kernel must have its adjoint and HVP regenerated with 0.4.6 or later.
"""
# The hoist's candidate set, copied from the 0.4.9 source: float scalars an `if` arm assigns without reading them in the same statement,
# for which a snapshot is needed.
function branch_scalar_vars(stmt, kinds, value_needed)
    vars = Symbol[]
    for sub in vcat(stmt.then, stmt.els)
        sub.kind == :assign || continue
        var = sub.lhs isa Symbol ? sub.lhs : sub.lhs.args[1]
        (sub.lhs isa Symbol && kinds[var] == :scalar_float) || continue
        STADE.agen_count_var_refs(sub.tree.expr, var) == 0 || continue
        STADE.agen_needs_snapshot(sub.lhs, sub.tree.expr, var, value_needed) || continue
        var in vars || push!(vars, var)
    end
    return vars
end

function scan_if_rewrite(paths::String...)
    files = String[]
    for p in paths
        isdir(p) ? append!(files, sort([joinpath(p, f) for f in readdir(p) if endswith(f, ".jl") && !(endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl"))])) : push!(files, p)
    end
    hits = Tuple{String,Any}[]
    scanned = 0
    function walk(plan, body, kinds, vn, acc)
        for (idx, st) in enumerate(plan)
            if st.kind == :if
                vars = branch_scalar_vars(st, kinds, vn)
                later = Set{Symbol}()
                foreach(j -> STADE.agen_stmt_assigned!(body[j], later), idx + 1:length(body))
                for v in vars
                    v in later && push!(acc, v)
                end
                walk(st.then, body[idx].then, kinds, vn, acc); walk(st.els, body[idx].els, kinds, vn, acc)
            elseif st.kind == :for || st.kind == :while
                walk(st.body, body[idx].body, kinds, vn, acc)
            end
        end
    end
    for f in files
        try
            kernels = STADE.io_read_kernel_corpus(f)
            name = STADE.io_corpus_entry_name(f, kernels)
            inl = STADE.inl_inline_calls(kernels)
            kernel = STADE.norm_insert_dead_entry_resets(STADE.parse_kernel(inl[name]))
            plan = STADE.lin_build(kernel, STADE.act_analyze(kernel))
            acc = Symbol[]
            walk(plan, kernel.body, kernel.sig.kinds, STADE.agen_value_needed_vars(kernel), acc)
            scanned += 1
            isempty(acc) || push!(hits, (f, sort(unique(acc))))
        catch e
            println("skip ", f, ": ", first(split(sprint(showerror, e), "\n"))[1:min(end, 100)])
        end
    end
    println("scanned ", scanned, " kernels, ", length(hits), " affected")
    for (f, v) in hits
        println("  AFFECTED ", f, "  scalar(s): ", join(v, ", "))
    end
    return hits
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(isempty(scan_if_rewrite(ARGS...)) ? 0 : 1)
end
