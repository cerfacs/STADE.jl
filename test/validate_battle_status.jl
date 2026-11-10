# Battle test, family F1 to F6: does each loop get the status the oracle expects?
# Reads generated code only.  See STADE_offload_battle_plan_2026-11-09.md.
include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
include(joinpath(@__DIR__, "battle_lib.jl"))

const BT_DIR = joinpath(@__DIR__, "battle-corpus")

function bt_subject_files(prefixes)
    sort([f for f in readdir(BT_DIR) if endswith(f, ".jl") && any(p -> startswith(f, p), prefixes)])
end

# one row per (subject, mode, backend, config, loop)
function bt_status_rows(file::String; modes = (:primal, :tangent, :adjoint, :hvp), backends = (:cuda, :jacc),
                        configs = (:pinned,), fuses = (true, false))
    path = joinpath(BT_DIR, file)
    f = STADE.io_read_corpus_entry(path)
    tab = bt_read_table(path)
    classes = Dict(v => c for (v, (c, e)) in tab.loops)
    exp_oracle = bt_expected(f, classes)
    rows = NamedTuple[]
    synt = bt_classify_subject(f)
    for (v, (c, e)) in tab.loops
        sc = get(synt, v, :U)
        push!(rows, (subject = file, kind = :oracle, loop = v, expected = e, oracle = get(exp_oracle, v, :?), table_class = c, synt = sc))
    end
    for mode in modes, fuse in fuses
        (mode === :primal && !fuse) && continue
        expr = try bt_mode_expr(f, mode; fuse = fuse) catch e; push!(rows, (subject = file, kind = :gen_error, mode = mode, fuse = fuse, msg = first(sprint(showerror, e), 120))); continue end
        for be in backends, cf in configs
            pl = try bt_plan(expr, be, cf) catch e; push!(rows, (subject = file, kind = :plan_error, mode = mode, backend = be, msg = first(sprint(showerror, e), 120))); continue end
            obs, allow = bt_observe(pl, collect(keys(tab.loops)), collect(keys(tab.whiles)))
            for (v, (c, e)) in tab.loops
                st = obs[v]
                push!(rows, (subject = file, kind = :status, mode = mode, backend = be, config = cf, fuse = fuse, loop = v,
                             expected = e, observed = st, kernels = length(pl.kernels), allow = allow))
            end
        end
    end
    return rows
end

# severity of one status row.  expected K and observed R (CUDA reduction twin) is a match by design.
function bt_verdict(expected::Symbol, observed::Dict{Symbol,Int}, backend::Symbol)
    obs = Set(keys(observed))
    # a lane-strided inner loop (:L) belongs to a twin variant of a kernel: it does not change the decision of the loop
    length(obs) > 1 && :L in obs && (obs = setdiff(obs, [:L]))
    (expected === :K && obs == Set([:R])) && return :ok      # a host loop below the threshold, a device reduction above it (CUDA mapreduce, JACC @parallel_reduce)
    expected === :K && :R in obs && (obs = Set(o === :R ? :K : o for o in obs))   # a reduction twin is a thread loop above the threshold
    obs == Set([expected]) && return :ok
    :X in obs && return :absent
    if expected === :H
        :K in obs && return :false_offload                  # S1: loop the oracle keeps sequential runs as threads
        return :other
    elseif expected === :K
        :H in obs && return :missed_offload                 # S2
        :T in obs && return :missed_offload
        return :other
    else  # expected T
        :K in obs && return :extra_offload
        :H in obs && return :missed_offload
        return :other
    end
end

function bt_run_status(files; out::String, modes = (:primal, :tangent, :adjoint, :hvp), configs = (:pinned,))
    open(out, "w") do io
        println(io, join(["subject", "mode", "fuse", "backend", "config", "loop", "expected", "observed", "verdict", "kernels", "allow"], "\t"))
        for f in files
            rows = bt_status_rows(f; modes = modes, configs = configs)
            for r in rows
                if r.kind === :oracle
                    r.expected == r.oracle || println(io, join([r.subject, "-", "-", "-", "-", r.loop, r.expected, r.oracle, "oracle_mismatch", 0, 0], "\t"))
                    (r.synt in (:I, :S) && r.synt != r.table_class) && println(io, join([r.subject, "-", "-", "-", "-", r.loop, r.table_class, r.synt, "class_mismatch", 0, 0], "\t"))
                elseif r.kind === :status
                    println(io, join([r.subject, r.mode, r.fuse, r.backend, r.config, r.loop, r.expected,
                                      join(sort([string(k, "=", v) for (k, v) in r.observed]), ","), bt_verdict(r.expected, r.observed, r.backend), r.kernels, r.allow], "\t"))
                else
                    println(io, join([r.subject, r.mode, get(r, :fuse, "-"), get(r, :backend, "-"), "-", "-", "-", r.msg, r.kind, 0, 0], "\t"))
                end
            end
            flush(io)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    # usage: julia validate_battle_status.jl OUT PREFIXES [MODES] [CONFIGS] [MODULO RESIDUE]
    out = ARGS[1]; prefixes = split(ARGS[2], ",")
    modes = length(ARGS) >= 3 ? Tuple(Symbol.(split(ARGS[3], ","))) : (:primal, :tangent, :adjoint, :hvp)
    configs = length(ARGS) >= 4 ? Tuple(Symbol.(split(ARGS[4], ","))) : (:pinned,)
    files = bt_subject_files(prefixes)
    if length(ARGS) >= 6
        md = parse(Int, ARGS[5]); rs = parse(Int, ARGS[6])
        files = [f for (i, f) in enumerate(files) if i % md == rs]
    end
    bt_run_status(files; out = out, modes = modes, configs = configs)
    println("wrote ", out, " for ", length(files), " subjects")
end
