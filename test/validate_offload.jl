include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_offload(dir="val-corpus")

Check how much of each corpus kernel's adjoint and HVP reaches the GPU.

`validate_corpus.jl` checks that generated code computes the right numbers. It
says nothing about whether that code can run on a device, so a change that
silently pushes a loop back onto the host passes every oracle. This script
measures the property directly: for each kernel it generates the adjoint and
the HVP with `keep_push_pop = false` (the only GPU-eligible mode), converts
each with the CUDA backend, and counts `for` loops before and after.

Both modes are measured against the same ceiling, because an HVP forward sweep
is the adjoint's forward sweep with every statement doubled by
`hvp_double_stmt` -- same loops, same bounds, same snapshot sites -- so it must
offload exactly as well. Measuring only the adjoint hid a real defect for as
long as this file existed: `cgen_elide_snapshot_saves_body` tested the
statement immediately after a snapshot save for the covering overwrite, which
is where an adjoint puts it and is two statements early for an HVP, whose
interleaved shadow twin sits in between. mpnn's adjoint offloaded every loop
while its HVP left the loops over graph edges and over graph nodes on the host,
launching a kernel per edge and per node instead of one kernel over all of
them.

The measure is **loops left on the host**, never device-kernel count. Adjacent
splittable loops merge into one kernel, so fewer kernels can mean more
offloaded, and a kernel count moves for reasons that have nothing to do with
coverage.

`EXPECTED` below is a ceiling per kernel, not an equality. A kernel that
offloads more than recorded passes and prints a note, so an improvement never
fails the build. A kernel that offloads less fails.

This script needs no GPU. `stade_gpu` is codegen, and CUDA is never loaded.
"""
function validate_offload(dir::String = joinpath(@__DIR__, "val-corpus"))
    # max host-side loops allowed per kernel. `:refused` = cgen_ingest cannot take
    # this adjoint at all, which is correct when a stack stays growable.
    EXPECTED = Dict{String,Any}(
    # bnd_carried, ii_readnested and ttgc sit above what they used to reach, deliberately.
    # cgen_liveout_is_zeroed refuses to split a loop whose local scalars a later HOST
    # statement reads -- those reads saw a stale value, since the split moved the
    # assignment onto the device. Their gradients were correct anyway, but only because
    # every stale value happened to land in a snapshot slot whose restore was dead, which
    # nothing enforced. These three are the residue after agen_boundary_push_redundant
    # cleared the rest; each one's scalar is genuinely read after the loop, so the split
    # cannot be made safe by removing a redundant push. Six host loops corpus-wide is the
    # measured price of the guard.
        # The ddp_* subjects for the bgen_ stage. Every one offloads fully,
        # adjoint and HVP alike. bgen_ adds no code path to this script, so
        # any movement in these rows is a regression in cgen_, not in bgen_.
        # This script counts host loops PRESENT, not host loops TAKEN. A
        # reduction into a local now emits both a host loop and a
        # mapreduce behind a run-time trip-count test, so the loop is in
        # the source whichever branch executes. The count is therefore
        # back where it was before the reduction lowering existed.
        "mlp1d" => 2,
        # Reduction stress subjects. Ceilings measured with the script's own
        # counter, which generates with the default reduction_threshold.
        # inl_ argument-relaxation witnesses: an expression and a literal
        # passed for a read-only scalar parameter.
        "inl_argexpr" => 0, "inl_arglit" => 0,
        "partialdot" => 0,   # witness for the view-bounded reduction
        "red_single_cube" => 0, "red_three_arrays" => 0, "red_subtract" => 0,
        "red_reverse" => 0, "red_strided" => 0, "red_refuse_invariant" => 0,
        "red_refuse_barevar" => 0, "red_nested_inner" => 0,
        # Two top-level reductions into locals, one of them guarded (see
        # mlp1d above). The other never lowers: CSE splits its body into
        # two statements and cgen_idiomatic_scalar_reduction requires one.
        "red_two_locals" => 6,
        "ddp_affine" => 0,
        "ddp_ambiguous_reduce" => 0,
        "ddp_batch_hist" => 0,
        "ddp_branch" => 0,
        "ddp_ragged" => 1,   # Tier B, same as raggedii
        "ddp_tangent_input" => 0,
        "ddp_twoparam" => 0,
        "ddp_written_per_sample" => 0,
        # Array-snapshot witnesses. Each of these scatters into an array that the loop reads after every write, so the array snapshot
        # replaces the per-write saves and the scatter loops leave the host. The v0.4.7 ceilings were 7 (10 for nested, 9 for ii).
        "asnap_ok" => 3, "asnap_2d" => 3, "asnap_ifwrite" => 3, "asnap_twoarrays" => 3, "asnap_zerocover" => 3, "asnap_zerotrip" => 3,
        "asnap_outside" => 3, "asnap_nested" => 6, "asnap_nestedboth" => 6, "asnap_ii" => 3,
        # Kernels the snapshot must refuse. These keep their host loops. asnap_nameclash keeps the scatter into m on the host,
        # because the name of its snapshot stack would collide with the array m_snap.
        "asnap_readbefore" => 7, "asnap_selfnl" => 9, "asnap_readbetween" => 11, "asnap_nameclash" => 7, "asnap_ragged" => 11,
        # Cost refusals: the snapshot would be correct and would not pay (guards G5, G10, G13).
        "asnap_condread" => 11, "asnap_direct" => 2, "asnap_slice" => 2,
        "advection" => 3, "advection_multi" => 3, "affine_loss" => 0,
        "bilinear" => 0, "bnd_branch" => 3, "bnd_carried" => 1,
        "bnd_nested_only" => 5, "bnd_readfirst" => 6,
        "branchsel" => 0, "cascadic_mg_prolong" => 22,
        "cellscatter" => 3,
        "clamped_sumsq" => 0, "coarsen_retire" => 3, "cond_field_choice" => 0,
        "cond_loop_choice" => 0, "dotprod" => 0,
    # cse_intoffset, mg_vcycle and mg_vcycle_multi each offload one more loop since the split gate
    # accepts accumulate-only and consume-only reverse loops (idiom relaxation).
    # CSE coverage kernels, at the loop counts they reach today. cse_branch and
    # cse_intoffset keep only their row loops on the host. cse_zerotrip keeps 6 of 7,
    # the same shape as entry_empty (7): its window is a runtime scalar reassigned each
    # pass, so cgen_last_assign_is_zero cannot prove the zeroing write happens and
    # declines to split. Each kernel's HVP matches its adjoint, which is the property
    # this file exists to hold.
        "cse_branch" => 1, "cse_zerotrip" => 6, "cse_intoffset" => 1,
    # gather_alias keeps all three of its loops on the host: the gathered read makes the sweep
    # sequential, so cgen_ cannot split it, and the stack-init loop goes with it.
        "gather_alias" => 3,
    # entry_empty and mpnn sit one or two loops above what they reached before
    # cgen_last_assign_is_zero started requiring the zeroing write to be GUARANTEED. For
    # entry_empty the split it gave up was unsafe: the shadow is zeroed inside a loop that
    # runs zero times once the window retires, and the GPU adjoint was wrong by 1.43 while
    # the CPU one was correct. mpnn is the conservative residue of the same rule.
        "entry_branch" => 3, "entry_dead" => 2, "entry_empty" => 7,
        "geomrecur" => 3,
    # Negative control, and the one entry here that is a REFUSAL rather than a
    # budget. The outer loop carries a read-after-write -- each cell's dense
    # layer consumes the previous cell's slice -- so all three outer loops in
    # the bundle (forward sweep, reverse sweep, and the re-emitted primal) must
    # stay on the host. A drop below 3 is not an improvement here; it means an
    # unsafe loop reached the device. Do NOT lower this baseline.
    "halo_assembly" => 3,
        "fixed_sweeps" => 3, "ii_kill" => 6, "ii_readbefore" => 4, "ii_readnested" => 2,
        "matvec_loss" => 0, "mg_vcycle" => 27, "mg_vcycle_multi" => 27,
        "mpnn" => 0, "normcomp" => 0, "pipeline" => 0, "prefixscan" => 3,
        "quadloss" => 0, "raggedii" => 1, "raggedind" => 1, "red_escape" => 5,
        "retire_empty" => 4, "retire_nested" => 11, "retire_if" => 6,
    # Witnesses for the nested-`if` and branch-scalar shapes. The two in-place stencils (inplace_stencil_if, nest_if_scalar) read the array
    # they overwrite, so every loop stays on the host. Each HVP matches its adjoint.
        "nest_if" => 0, "nest_if_scalar" => 6, "if_scalar_then_if" => 3, "if_scalar_then_loop_if" => 3, "retire_in_if" => 5,
        "inplace_stencil_if" => 6, "if_top_scalar" => 1, "if_top_two" => 1, "if_in_outer_loop" => 2, "early_stage_offset" => 13,
        "relu_field" => 0, "richardson_substep" => 7, "stencil_loss" => 0,
        "sumsq_shifted" => 0, "transformer" => 31, "ttgc" => 6,
        "two_field_loss" => 0, "unet" => 0, "weightedsumsq" => 0,
        "windowed_relax_retire" => 7,
    # idm_*: the idiom relaxation of the GPU split gate (cgen_split_loop_gate). The four
    # offloading kernels reach 0 host loops in the adjoint AND the HVP. The refuse_* kernels
    # are negative controls: their ceilings are budgets, and REFUSALS below pins WHICH loop
    # is refused and why, since an improvement there would mean an unsafe loop reached the device.
        "idm_slots" => 0, "idm_zerotrip" => 0, "idm_acc_two" => 0, "idm_tile" => 0,
        "idm_dup_write" => 0,
        # loop-invariant atomic hoisting: ragged per-row loops that accumulate into a shared bin; the one host loop of hoist_sentinel is the prefix table of its Tier B trip counts in initstacks_*
        "hoist_bins" => 0, "hoist_nested" => 0, "hoist_sentinel" => 1,
        "idm_cse_acc" => 2, "ii_multiversion" => 0, "idm_src_consume" => 2, "idm_gather_rmw" => 6, "idm_overlap_plain" => 3,
        "idm_refuse_cond_read" => 3, "idm_refuse_scalar_read" => 3, "idm_refuse_consume_gap" => 3,
        # idm_refuse_scalar_read_nl: the nonlinear twin of idm_refuse_scalar_read, kept so the scalar gate still has a witness once linear uses stopped saving the scalar
        "idm_refuse_scalar_read_nl" => 3,
        # lin_*: in-place linear loops (v0.4.27 value-needed rule); the carried and ping-pong loops are order-dependent and must stay on the host
        "lin_pingpong" => 3, "lin_quotient" => 3, "lin_then_quad" => 3, "lin_quad_inside" => 3, "lin_guard_products" => 3,
        "lin_carried_sum" => 3, "lin_carried_scale" => 3,
        "so_tile" => 0, "so_hostread" => 4, "so_liveout" => 9, "so_carry" => 5, "so_inner" => 5,
        # correctness kernels for the rematerialized-restore guards and the branch-scalar rewrite; not offload targets
        "if_scalar_rewrite" => 0, "remat_shapes" => 20,
        "liveout_red_nest" => 0, "liveout_red_fused" => 0, "liveout_red_top" => 3, "liveout_local_last" => 0, "remat_cross" => 6,
        "cross_readbefore" => 0, "cross_entry" => 0, "cross_after" => 0, "cross_ifwrite" => 0, "cross_nested" => 0, "cross_branchlocal" => 2,
        "cross_mixed" => 6, "cross_argread" => 3, "selfrace_conv" => 0, "selfrace_layout" => 0,
    )
    # kernel => decisions that MUST appear in the offload report of its adjoint: (verdict, reason, array).
    # A listed loop that reaches a kernel instead is a failure, not an improvement.
    REFUSALS = Dict{String,Vector{Tuple{Symbol,Symbol,Any}}}(
        "idm_refuse_cond_read" => [(:host, :array, :v), (:host, :overlap, :v)],
        "idm_gather_rmw" => [(:host, :rmw, :m)],
        "idm_cse_acc" => [(:host, :rmw, :mb)],
        "idm_overlap_plain" => [(:host, :overlap, :a)],
        # a scalar used only linearly is no longer saved (v0.4.27), so this loop is now refused on the array alone; idm_refuse_scalar_read_nl pins the scalar gate
        "idm_refuse_scalar_read" => [(:host, :array, :vb), (:host, :array, :v)],
        "idm_refuse_scalar_read_nl" => [(:host, :scalar, nothing), (:host, :array, :v)],
        "idm_refuse_consume_gap" => [(:host, :scalar, nothing), (:host, :array, :u)],
        # so_*: the save-only scalar relaxation (cgen_saveonly_synth). so_tile offloads; each control breaks one of its four conditions
        "so_hostread" => [(:host, :scalar, nothing)],
        "so_liveout" => [(:host, :liveout, nothing)],
        "so_carry" => [(:host, :scalar, nothing), (:host, :scope, nothing)],
        "so_inner" => [(:host, :scalar, nothing)],
    )
    # idiom acceptance MUST appear for these (the loop is offloaded only because of the idiom)
    IDIOM_ACCEPTED = Dict{String,Int}("idm_slots" => 1, "idm_tile" => 1)

    count_for(e) = e isa Expr ? (e.head == :for) + sum(Int[count_for(a) for a in e.args]; init = 0) : 0

    kernels = sort(filter(readdir(dir)) do f
        endswith(f, ".jl") && !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl"))
    end)

    MODES = ((:adjoint, STADE.stade_adjoint_file, "_b.jl"),
             (:hvp,     STADE.stade_hvp_file,     "_hv.jl"))

    bad = 0
    checks_extra = Ref(0)
    improved = String[]
    for f in kernels
        name = splitext(f)[1]
        haskey(EXPECTED, name) || (println(rpad(name, 26), " NO BASELINE -- add one"); bad += 1; continue)
        for (mode, gen_fn, suffix) in MODES
            out = joinpath(dir, name * suffix)
            total = 0
            host = 0
            status = :ok
            try
                gen_fn(joinpath(dir, f), out; keep_push_pop = false, fuse_ii_loops = true)
                for e in STADE.io_read_kernel_bundle(out)
                    total += count_for(e)
                    host += count_for(STADE.stade_gpu(e, STADE.cgen_backend_cuda()).host)
                end
            catch
                status = :refused
            end
            want = EXPECTED[name]
            label = rpad("$(name) [$(mode)]", 34)
            if status == :refused
                ok = want === :refused
                println(label, ok ? " ok" : " FAIL", "  ingest refused (expected $(want))")
                ok || (bad += 1)
            elseif want === :refused
                println(label, " FAIL  now ingests (baseline says refused) -- update the baseline")
                bad += 1
            else
                println(label, host <= want ? " ok" : " FAIL",
                        "  host_left=", host, "/", total, " (max ", want, ")")
                host > want && (bad += 1)
                host < want && push!(improved, "$(name) [$(mode)]: $(want) -> $(host)")
            end
        end
    end

    # Refusal pins and idiom acceptance, read from the offload report (cgen_decision_record).
    for f in kernels
        name = splitext(f)[1]
        (haskey(REFUSALS, name) || haskey(IDIOM_ACCEPTED, name)) || continue
        recs = NamedTuple[]
        try
            recs = STADE.stade_gpu_report_file(joinpath(dir, name * "_b.jl"), STADE.cgen_backend_cuda())
        catch e
            println(rpad(name, 34), " FAIL  report: ", first(split(sprint(showerror, e), "\n")))
            bad += 1
            continue
        end
        checks_extra[] += 1
        for (verdict, reason, arr) in get(REFUSALS, name, Tuple{Symbol,Symbol,Any}[])
            if !any(r -> r.verdict == verdict && r.reason == reason && r.array == arr, recs)
                println(rpad(name, 34), " FAIL  no ", verdict, " decision with reason ", reason, " array ", arr)
                bad += 1
            end
        end
        if haskey(IDIOM_ACCEPTED, name)
            n = count(r -> r.verdict == :kernel && r.via_idiom, recs)
            if n < IDIOM_ACCEPTED[name]
                println(rpad(name, 34), " FAIL  expected >= ", IDIOM_ACCEPTED[name], " loop(s) accepted via an idiom, found ", n)
                bad += 1
            end
        end
    end

    # Backend agreement used to be checked here too. It now lives in
    # validate_backend_agreement.jl, which compares loop shapes and kernel arities
    # rather than bare counts. Two implementations of one rule drift apart; this one
    # was the weaker, so it is gone rather than duplicated.
    # No generated host function may CALL a plain name it also ASSIGNS as a scalar.
    # A shadow name is its primal name plus b/d/bd, so a primal scalar `cl` becomes
    # `cld` and shadows Base.cld -- which cgen_launch_expr calls in that same scope.
    # mg_vcycle's HVP died on a live GPU with "objects of type Float64 are not
    # callable" while its adjoint (suffix `b`) and its JACC build (launch uses div)
    # both passed. Checking the property beats blacklisting names: it catches
    # `mo` -> `mod` and `fl` -> `fld` too, and anything future codegen starts calling.
    for f in kernels
        name = splitext(f)[1]
        primal = STADE.io_read_corpus_entry(joinpath(dir, f))
        for (mode, _, _) in MODES
            gen = mode == :hvp ? STADE.stade_hvp(primal; keep_push_pop = false, fuse_ii_loops = true) :
                                 STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true)
            expr = mode == :hvp ? gen.hvp : gen.adjoint
            for (label, host) in (("cuda", STADE.stade_gpu(expr, STADE.cgen_backend_cuda()).host),
                                  ("jacc", STADE.stade_jacc(expr).host))
                assigned = Set{Symbol}()
                collect_assigned(e) = e isa Expr &&
                    (e.head === :(=) && e.args[1] isa Symbol && push!(assigned, e.args[1]);
                     foreach(collect_assigned, e.args))
                collect_assigned(host)
                shadowed = Symbol[]
                scan(e) = e isa Expr &&
                    (e.head === :call && e.args[1] isa Symbol && e.args[1] in assigned &&
                        push!(shadowed, e.args[1]);
                     foreach(scan, e.args))
                scan(host)
                checks_extra[] += 1
                if !isempty(shadowed)
                    println(rpad("$(name) [$(mode)/$(label)]", 34),
                            " FAIL  calls a shadowed name: ", join(sort(unique(shadowed)), ", "))
                    bad += 1
                end
            end
        end
    end

    checks = length(kernels) * length(MODES) + checks_extra[]
    isempty(improved) || println("\nimproved (lower the baseline): ", join(improved, ", "))
    println("\n", checks - bad, "/", checks, " checks within budget",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_offload() == 0 ? 0 : 1)
end
