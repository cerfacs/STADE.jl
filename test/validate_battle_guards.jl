# Battle test, families F7 to F10: guards and safety of the generated device code (no device run).
#   G1  every launch survives a zero-trip loop (CUDA: guard in the kernel, JACC: guard in the host)
#   G2  no call to a banned vendor routine in any kernel or host
#   G3  generation is deterministic: two runs give the same text
#   G4  CUDA and JACC agree on which loops leave the host (K or R on CUDA, K on JACC)
#   G5  size: generated nodes against primal nodes, by loop depth
# Reads generated code only.  See STADE_offload_battle_plan_2026-11-09.md.
include(joinpath(@__DIR__, "validate_battle_status.jl"))

bt_nodes(e) = e isa Expr ? 1 + sum(bt_nodes, e.args; init = 0) : 1
const BT_BANNED = Set([:dot, :norm, :nrm2, :axpy!, :scal!, :BLAS])

bt_calls(e, acc = Symbol[]) = (e isa Expr && (e.head === :call && e.args[1] isa Symbol && push!(acc, e.args[1]); foreach(a -> bt_calls(a, acc), e.args)); acc)

# CUDA: a kernel guard is an `if` with a return as one of the first statements.
function bt_cuda_guarded(k::Expr)
    body = bt_stmts(k.args[2])
    for s in body[1:min(end, 8)]
        if s isa Expr && s.head === :if
            txt = string(s)
            occursin("return", txt) && return true
        end
    end
    return false
end
# JACC: every parallel launch sits inside an `if` that compares a range with zero.
function bt_jacc_unguarded(host)
    bad = String[]
    function walk(e, guarded)
        e isa Expr || return
        if e.head === :macrocall && any(a -> a isa Expr && a.head === :., [e.args[1]]) && occursin("parallel_", string(e.args[1]))
            guarded || push!(bad, first(string(e), 60))
        end
        if e.head === :macrocall && e.args[1] isa Symbol && e.args[1] in (Symbol("@parallel_for"), Symbol("@parallel_reduce"))
            guarded || push!(bad, first(string(e), 60))
        end
        if e.head === :macrocall && e.args[1] isa Expr && occursin("parallel_", string(e.args[1]))
            guarded || push!(bad, first(string(e), 60))
        end
        if e.head === :if
            cond = string(e.args[1])
            g = guarded || (occursin("> 0", cond) || occursin("0 <", cond) || occursin("!= 0", cond))
            walk(e.args[1], guarded)
            for a in e.args[2:end]; walk(a, g); end
        else
            foreach(a -> walk(a, guarded), e.args)
        end
    end
    walk(host, false)
    return bad
end

function validate_battle_guards(files = bt_subject_files(["f1_", "f2_", "f3_", "f4_", "f5_", "f6_"]);
                                modes = (:primal, :adjoint, :hvp), verbose = true)
    bad = 0; checks = 0
    sizes = Dict{Int,Vector{Float64}}()
    for f in files
        path = joinpath(BT_DIR, f)
        prim = STADE.io_read_corpus_entry(path)
        tab = bt_read_table(path)
        depth = length(tab.loops) + length(tab.whiles)
        pn = bt_nodes(prim)
        for mode in modes
            expr = try bt_mode_expr(prim, mode; fuse = true) catch e; continue end   # refusals are the status family's job
            plans = Dict{Symbol,Any}()
            for be in (:cuda, :jacc)
                plans[be] = try bt_plan(expr, be, :pinned) catch e; (verbose && println("PLAN-ERROR ", f, " ", mode, " ", be, " ", first(sprint(showerror, e), 100))); bad += 1; continue end
            end
            length(plans) == 2 || continue
            # G1
            checks += 1
            ng = [k for k in plans[:cuda].kernels if !bt_cuda_guarded(k)]
            if !isempty(ng); bad += 1; verbose && println("G1 NOT OK ", f, " ", mode, " cuda: ", length(ng), " kernel(s) without a guard: ", join([string(k.args[1].args[1]) for k in ng], ",")); end
            checks += 1
            ug = bt_jacc_unguarded(plans[:jacc].host)
            if !isempty(ug); bad += 1; verbose && println("G1 NOT OK ", f, " ", mode, " jacc: unguarded launch ", ug[1]); end
            # G2
            for be in (:cuda, :jacc)
                checks += 1
                calls = Set(bt_calls(plans[be].host)); for k in plans[be].kernels; union!(calls, bt_calls(k)); end
                hit = intersect(calls, BT_BANNED)
                if !isempty(hit); bad += 1; verbose && println("G2 NOT OK ", f, " ", mode, " ", be, ": ", collect(hit)); end
            end
            # G3
            for be in (:cuda, :jacc)
                checks += 1
                p2 = bt_plan(expr, be, :pinned)
                if bt_hash(plans[be]) != bt_hash(p2); bad += 1; verbose && println("G3 NOT OK ", f, " ", mode, " ", be, ": two generations differ"); end
            end
            # G4
            checks += 1
            vars = collect(keys(tab.loops)); wv = collect(keys(tab.whiles))
            oc, _ = bt_observe(plans[:cuda], vars, wv); oj, _ = bt_observe(plans[:jacc], vars, wv)
            offl(o) = Set(v for v in vars if !isempty(intersect(keys(o[v]), (:K, :R))))
            if offl(oc) != offl(oj)
                bad += 1
                verbose && println("G4 NOT OK ", f, " ", mode, ": offloaded on cuda ", sort(collect(offl(oc))), " on jacc ", sort(collect(offl(oj))))
            end
            # G5
            push!(get!(sizes, depth, Float64[]), (bt_nodes(plans[:cuda].host) + sum(bt_nodes, plans[:cuda].kernels; init = 0)) / pn)
        end
    end
    verbose && for d in sort(collect(keys(sizes)))
        v = sizes[d]; println("G5 loops=", d, " n=", length(v), " growth min/median/max = ", round(minimum(v), digits = 1), " / ", round(sort(v)[cld(end, 2)], digits = 1), " / ", round(maximum(v), digits = 1))
    end
    println(bad == 0 ? "battle guards: all $checks checks pass" : "battle guards: $bad of $checks checks NOT OK")
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    pre = length(ARGS) >= 1 ? split(ARGS[1], ",") : ["f1_", "f2_", "f3_", "f4_", "f5_", "f6_"]
    files = bt_subject_files(pre)
    if length(ARGS) >= 3
        md = parse(Int, ARGS[2]); rs = parse(Int, ARGS[3]); files = [f for (i, f) in enumerate(files) if i % md == rs]
    end
    validate_battle_guards(files) == 0 || exit(1)
end
