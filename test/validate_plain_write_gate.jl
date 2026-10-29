include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_plain_write_gate(dir="val-corpus")

Regression gate for the two plain-write hazards that `validate_ownership.jl` found in unmodified STADE 0.4.4 (ttgc and
idm_refuse_cond_read; since the array snapshot, asnap_readbefore witnesses the first), and for the three guards in `cgen_split_loop_gate` that close them:

  * `cgen_overlap_proven_arrays` refuses a loop whose plain writes to one array provably land on one element from two threads
    (`v[i]` and `v[i+1]`). An index it cannot analyse stays trusted, so mpnn's three slices of one buffer still offload.
  * `cgen_rmw_hazard_arrays` refuses a read-modify-write that no atomic add repairs, or an undo from a stack, through an index
    gathered from an array, and an array that is both accumulated and plainly written through gathered indices.
  * `cgen_fold_accumulates` turns a CSE'd accumulation (`t = A[e]; A[e] = t + x`, t used once) back into `A[e] += x`, so it
    offloads as an atomic add instead of staying a plain read-modify-write.

It also covers the save-only scalar relaxation (`cgen_saveonly_synth`), which lets a register-tiled loop offload although its stack
saves read a scalar before the loop assigns it. Its three guards (`cgen_ends_with_push`, `cgen_saveonly_closed`, and the live-out
check) are sabotaged like the others; the controls so_inner, so_hostread and so_liveout each break exactly one condition.

Three groups of checks, no GPU needed:

  1. VERDICTS: the loop the guard must refuse is refused with the named reason and array; the loop that must offload offloads,
     and the CUDA of `idm_cse_acc` carries an atomic add on `m`.
  2. KEPT: kernels whose plain writes the guards must NOT touch still offload (mpnn, stencil_loss, transformer, idm_dup_write).
     An over-eager guard hands every such kernel to the host, which is correct and very slow on a device, and nothing else
     would notice.
  3. SABOTAGE: in a fresh process each guard is replaced by a version that finds nothing. The ownership check must then flag the
     kernel the guard protects. A guard whose removal changes nothing is not a guard.
"""
function validate_plain_write_gate(dir::String = joinpath(@__DIR__, "val-corpus"))
    bad = 0
    check(ok, msg) = (println(rpad(msg, 78), ok ? " ok" : " FAIL"); ok || (bad += 1); ok)
    work = mktempdir()
    function report(name)
        for sfx in (".jl",)
            cp(joinpath(dir, name * sfx), joinpath(work, name * sfx); force = true)
        end
        STADE.stade_adjoint_file(joinpath(work, name * ".jl"), joinpath(work, name * "_b.jl"); keep_push_pop = false, fuse_ii_loops = true)
        return STADE.stade_gpu_report_file(joinpath(work, name * "_b.jl"), STADE.cgen_backend_cuda())
    end
    has(recs, verdict, reason, arr) = any(r -> r.verdict == verdict && r.reason == reason && r.array == arr, recs)

    # 1. verdicts
    r = report("idm_overlap_plain");  check(has(r, :host, :overlap, :a), "idm_overlap_plain: plain writes a[i], a[i+1] -> host (overlap, a)")
    r = report("idm_refuse_cond_read"); check(has(r, :host, :overlap, :v), "idm_refuse_cond_read: restores v[i], v[i+1] -> host (overlap, v)")
    r = report("idm_gather_rmw");     check(has(r, :host, :rmw, :m), "idm_gather_rmw: m[n] = m[n]*0.5 + x through a gather -> host (rmw, m)")
    # the array snapshot takes the saved old values off ttgc's scatters; asnap_readbefore reads the array before its writes, keeps them, and stays on the host
    r = report("asnap_readbefore");   check(any(x -> x.verdict == :host && x.reason == :rmw, r), "asnap_readbefore: scatter with saved old value, and its undo -> host (rmw)")
    r = report("ttgc");               check(!any(x -> x.reason == :rmw, r), "ttgc: no saved old value left on the scatters (array snapshot) -> no rmw refusal")
    r = report("idm_cse_acc")
    check(any(x -> x.verdict == :kernel && x.reason == :ok, r), "idm_cse_acc: CSE'd accumulation -> offloads")
    check(has(r, :host, :rmw, :mb), "idm_cse_acc: its reverse zeroes and re-adds mb through a gather -> host (rmw, mb)")
    STADE.stade_cuda_file(joinpath(work, "idm_cse_acc_b.jl"), joinpath(work, "idm_cse_acc_b_cuda.jl"); warp_atomics = false, inbounds = false, dead_scalars = false)
    check(occursin(r"@atomic m\[i_node\] \+=", read(joinpath(work, "idm_cse_acc_b_cuda.jl"), String)), "idm_cse_acc: forward scatter into m emits an atomic add")

    # the save-only scalar relaxation: the tile kernel offloads completely, each control keeps its refused loop on the host
    for mode in ("adjoint",)
        r = report("so_tile"); check(!any(x -> x.verdict == :host, r), "so_tile: both tile loops and every other loop offload, no host loop ($(count(x -> x.verdict == :host, r)) found)")
        r = report("so_hostread"); check(any(x -> x.verdict == :host && x.reason == :scalar && x.loop_var == :t, r), "so_hostread: wv read outside the tile loop -> tile loop stays on the host (scalar)")
        r = report("so_liveout"); check(any(x -> x.verdict == :host && x.reason == :liveout, r), "so_liveout: wv read after the tile loop -> tile loop stays on the host (liveout)")
        r = report("so_inner"); check(any(x -> x.verdict == :host && x.reason == :scalar && x.loop_var == :ci && x.depth == 1, r), "so_inner: tap loop without an end-of-body push -> stays on the host (scalar)")
    end

    # 2. kept on the device
    for (name, ceiling) in (("mpnn", 0), ("stencil_loss", 0), ("transformer", 31), ("idm_dup_write", 0), ("cellscatter", 3))
        r = report(name)
        host = count(x -> x.verdict == :host, r)
        check(!any(x -> x.reason in (:overlap, :rmw), r), "$name: no loop refused by the plain-write guards (host loops: $host, ceiling $ceiling)")
    end

    # 3. sabotage, one fresh process per guard
    src = joinpath(@__DIR__, "validate_ownership.jl")
    guards = (
        ("overlap", "cgen_overlap_proven_arrays(body::Vector{NamedTuple}, thread_var::Symbol) = Set{Symbol}()", ["idm_overlap_plain", "idm_refuse_cond_read"]),
        ("rmw", "cgen_rmw_hazard_arrays(body::Vector{NamedTuple}) = Set{Symbol}()", ["idm_gather_rmw", "asnap_readbefore"]),
        # the three guards of the save-only scalar relaxation; each control breaks exactly one of its conditions
        ("saveonly push", "cgen_ends_with_push(body::Vector{NamedTuple}, v::Symbol) = true", ["so_inner"]),
        ("saveonly closed", "cgen_saveonly_closed(body::Vector{NamedTuple}, v::Symbol, in_writer::Bool = false) = true", ["so_hostread"]),
        ("saveonly liveout", "cgen_liveout_is_zeroed(stmt, root_body, loop_vars, synth = Dict{Symbol,Any}(), saveonly = Set{Symbol}()) = true", ["so_liveout"]),
    )
    for (nm, def, kernels) in guards
        script = """
        include($(repr(src)))
        Core.eval(STADE, Meta.parse($(repr(def))))
        r = validate_ownership(; only = $(repr(kernels)), known = Dict{String,String}(), verbose = false)
        println("SABOTAGE_BAD=", r.bad)
        """
        out = read(`$(Base.julia_cmd()) --startup-file=no -e $script`, String)
        m = match(r"SABOTAGE_BAD=(\d+)", out)
        n = m === nothing ? -1 : parse(Int, m.captures[1])
        check(n > 0, "sabotage $nm guard: ownership check flags $(join(kernels, ", ")) ($n of $(2 * length(kernels)) rows)")
    end
    # the fold guard is what lets the CSE'd scatter offload: without it the same loop is refused as a read-modify-write
    script = """
    include($(repr(joinpath(@__DIR__, "..", "src", "STADE.jl"))))
    Core.eval(STADE, Meta.parse("cgen_fold_accumulates(body::Vector{NamedTuple}, root::Vector{NamedTuple}) = body"))
    cp($(repr(joinpath(dir, "idm_cse_acc.jl"))), $(repr(joinpath(work, "sab.jl"))); force = true)
    STADE.stade_adjoint_file($(repr(joinpath(work, "sab.jl"))), $(repr(joinpath(work, "sab_b.jl"))); keep_push_pop = false, fuse_ii_loops = true)
    recs = STADE.stade_gpu_report_file($(repr(joinpath(work, "sab_b.jl"))), STADE.cgen_backend_cuda())
    println("SABOTAGE_FOLD=", count(r -> r.verdict == :host && r.reason == :rmw && r.array == :m, recs))
    """
    out = read(`$(Base.julia_cmd()) --startup-file=no -e $script`, String)
    m = match(r"SABOTAGE_FOLD=(\d+)", out)
    check(m !== nothing && parse(Int, m.captures[1]) > 0, "sabotage fold: without it the forward scatter into m is refused (host, rmw, m)")
    println("\n", bad == 0 ? "all plain-write gate checks passed" : "*** $bad NOT OK ***")
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_plain_write_gate() == 0 ? 0 : 1)
end
