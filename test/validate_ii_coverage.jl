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
    EXPECTED = Dict{String,Int}(
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
        got > want && push!(improved, "$(name): $(want) -> $(got)")
    end

    isempty(improved) || println("\nimproved (raise the baseline): ", join(improved, ", "))
    println("\n", length(kernels) - bad, "/", length(kernels), " kernels at or above floor",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end

validate_ii_coverage()
