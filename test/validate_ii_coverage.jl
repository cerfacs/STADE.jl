include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_ii_coverage(dir="val-corpus")

Check how many loops `fuse_ii_loops` still classifies in each corpus kernel.

`validate_corpus.jl` checks that generated code computes the right numbers, and a
change that refuses every classification passes every oracle -- fusion is an
optimization, so losing it is silent. `validate_offload.jl` does not see it
either: classification changes stack traffic, not which loops reach the device.
This script measures the property directly, by counting `stade_ii_plan_check`
entries per kernel.

`EXPECTED` below is a FLOOR per kernel, the mirror of validate_offload's ceiling.
A kernel that classifies more than recorded passes and prints a note, so a
genuine improvement never fails the build. A kernel that classifies fewer fails:
that is either a real loss of fusion, or an eligibility gate that has started
refusing something it used to accept.

Note what a floor does NOT check: that each classification is still CORRECT.
Only the oracles in validate_corpus.jl do that, and only for the shapes the
corpus contains. Raise a number here only alongside a kernel that exercises it.

This script needs no GPU and generates no code; it runs the classifier alone.
"""
function validate_ii_coverage(dir::String = joinpath(@__DIR__, "val-corpus"))
    # minimum classified loops per kernel. 0 means "nothing here is eligible",
    # which is the correct answer for most of the corpus -- those entries exist
    # so a kernel added later cannot silently skip this check.
    # CEILINGS pin loops that fuse_ii_loops must NOT classify. idm_src_consume is independent in fact, but only the
    # idiom rule of the GPU split gate can see that; the strict rule that fuse_ii_loops uses must keep refusing it.
    # If the strict rule starts accepting idioms, this number rises above 0 and the AD output changes.
    # ii_multiversion: 0.4.4 classified two of its loops and produced a wrong adjoint (the weight scalar is reassigned per tap).
    CEILINGS = Dict{String,Int}("idm_src_consume" => 0, "ii_multiversion" => 0)
    EXPECTED = Dict{String,Int}(
        "idm_slots" => 0, "idm_zerotrip" => 0, "idm_acc_two" => 0, "idm_tile" => 2, "idm_dup_write" => 0,
        "idm_refuse_cond_read" => 0, "idm_refuse_scalar_read" => 0, "idm_refuse_consume_gap" => 0,
        "idm_src_consume" => 0, "idm_cse_acc" => 0, "idm_gather_rmw" => 0, "idm_overlap_plain" => 0, "ii_multiversion" => 0,
        "so_carry" => 0, "so_hostread" => 0, "so_inner" => 0, "so_liveout" => 0, "so_tile" => 0,
        "if_scalar_rewrite" => 0, "remat_shapes" => 0,
        # liveout_*: an accumulator read after its loop is deferred to the backward position (:mixed), a live-out plain local is not covered
        "liveout_red_nest" => 2, "liveout_red_fused" => 2, "liveout_red_top" => 1, "liveout_local_last" => 0,
        "remat_cross" => 5, "cross_readbefore" => 0, "cross_entry" => 0, "cross_after" => 0, "cross_ifwrite" => 0, "cross_nested" => 0,
        "cross_branchlocal" => 0, "cross_mixed" => 2, "cross_argread" => 0, "selfrace_conv" => 0, "selfrace_layout" => 0,
        # baselines added for kernels that had none (measured at the revision that added them)
        "ddp_affine" => 0, "ddp_ambiguous_reduce" => 0, "ddp_batch_hist" => 0, "ddp_branch" => 0, "ddp_ragged" => 0, "ddp_tangent_input" => 0,
        "ddp_twoparam" => 0, "ddp_written_per_sample" => 0, "inl_argexpr" => 0, "inl_arglit" => 0, "mlp1d" => 3, "partialdot" => 0,
        "red_nested_inner" => 2, "red_refuse_barevar" => 0, "red_refuse_invariant" => 0, "red_reverse" => 0, "red_single_cube" => 0,
        "red_strided" => 0, "red_subtract" => 0, "red_three_arrays" => 0, "red_two_locals" => 2,
        "advection" => 0, "advection_multi" => 0, "affine_loss" => 0,
        "bilinear" => 0, "bnd_branch" => 2, "bnd_carried" => 1,
        "bnd_nested_only" => 0, "bnd_readfirst" => 1, "branchsel" => 0,
        "cascadic_mg_prolong" => 0, "cellscatter" => 1, "clamped_sumsq" => 0,
        "coarsen_retire" => 0, "cond_field_choice" => 0, "cond_loop_choice" => 0,
        # CSE coverage kernels. cse_branch's row loop is iteration-independent and
        # classifies; the other two are sequential (a retiring window, a ragged
        # accumulator) and correctly classify nothing.
        "cse_branch" => 1, "cse_zerotrip" => 0, "cse_intoffset" => 0,
        # gather_alias's loop is sequential -- it reads y at a gathered index an earlier
        # iteration may have written -- so nothing here is eligible for fusion.
        "gather_alias" => 0,
        "dotprod" => 0, "fixed_sweeps" => 0, "geomrecur" => 0,
        "halo_assembly" => 0,
        "entry_branch" => 0, "entry_dead" => 2, "entry_empty" => 0,
        "ii_kill" => 0, "ii_readbefore" => 1, "ii_readnested" => 0,
        "matvec_loss" => 0, "mg_vcycle" => 0,
        "mg_vcycle_multi" => 0, "mpnn" => 0, "normcomp" => 0, "pipeline" => 0,
        "prefixscan" => 0, "quadloss" => 0, "raggedii" => 0, "raggedind" => 0,
        "red_escape" => 0, "retire_empty" => 0,
        "relu_field" => 0, "richardson_substep" => 0, "stencil_loss" => 0,
        "sumsq_shifted" => 0, "transformer" => 13, "ttgc" => 6,
        "two_field_loss" => 0, "unet" => 2, "weightedsumsq" => 0,
        "windowed_relax_retire" => 0,
        # baselines added for 30 kernels that had none (measured with v0.4.14); asnap_* kernels pin array-snapshot shapes, whose loops mostly stay unclassified
        "asnap_2d" => 0, "asnap_condread" => 0, "asnap_direct" => 0, "asnap_ifwrite" => 0, "asnap_ii" => 1, "asnap_nameclash" => 0,
        "asnap_nested" => 0, "asnap_nestedboth" => 0, "asnap_ok" => 0, "asnap_outside" => 0, "asnap_ragged" => 0, "asnap_readbefore" => 0,
        "asnap_readbetween" => 0, "asnap_selfnl" => 0, "asnap_slice" => 0, "asnap_twoarrays" => 0, "asnap_zerocover" => 0, "asnap_zerotrip" => 0,
        "early_stage_offset" => 0, "if_in_outer_loop" => 1, "if_scalar_then_if" => 0, "if_scalar_then_loop_if" => 0, "if_top_scalar" => 0,
        "if_top_two" => 0, "inplace_stencil_if" => 0, "nest_if" => 0, "nest_if_scalar" => 0, "retire_if" => 0, "retire_in_if" => 0, "retire_nested" => 0,
    )

    kernels = sort(filter(f -> endswith(f, ".jl") &&
                               !(endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl")),
                          readdir(dir)))
    bad = 0
    improved = String[]
    for f in kernels
        name = splitext(f)[1]
        if !haskey(EXPECTED, name)
            println(rpad(name, 26), " NO BASELINE -- add one")
            bad += 1
            continue
        end
        got = try
            # stade_ii_plan_check runs both duplicated implementations and asserts
            # they agree, so a one-sided edit fails here as loudly as anywhere else
            length(STADE.stade_ii_plan_check(STADE.parse_kernel(STADE.io_read_corpus_entry(joinpath(dir, f)))))
        catch e
            println(rpad(name, 26), " ERROR  ", sprint(showerror, e)[1:min(end, 90)])
            bad += 1
            continue
        end
        want = EXPECTED[name]
        println(rpad(name, 26), got >= want ? " ok" : " FAIL", "  classified=", got, " (min ", want, ")")
        got < want && (bad += 1)
        if haskey(CEILINGS, name) && got > CEILINGS[name]
            println(rpad(name, 26), " FAIL  classified ", got, " > ceiling ", CEILINGS[name], " (the strict gate must not accept idioms, and a loop with several live scalar versions must not be classified)")
            bad += 1
        end
        got > want && push!(improved, "$(name): $(want) -> $(got)")
    end

    isempty(improved) || println("\nimproved (raise the baseline): ", join(improved, ", "))
    println("\n", length(kernels) - bad, "/", length(kernels), " kernels at or above floor",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end

validate_ii_coverage()
