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
        "advection" => 3, "advection_multi" => 3, "affine_loss" => 0,
        "bilinear" => 0, "bnd_branch" => 3, "bnd_carried" => 1,
        "bnd_nested_only" => 5, "bnd_readfirst" => 6,
        "branchsel" => 0, "cascadic_mg_prolong" => 22,
        "cellscatter" => 3,
        "clamped_sumsq" => 0, "coarsen_retire" => 4, "cond_field_choice" => 0,
        "cond_loop_choice" => 0, "dotprod" => 0,
    # entry_empty and mpnn sit one or two loops above what they reached before
    # cgen_last_assign_is_zero started requiring the zeroing write to be GUARANTEED. For
    # entry_empty the split it gave up was unsafe: the shadow is zeroed inside a loop that
    # runs zero times once the window retires, and the GPU adjoint was wrong by 1.43 while
    # the CPU one was correct. mpnn is the conservative residue of the same rule.
        "entry_branch" => 3, "entry_dead" => 2, "entry_empty" => 7,
        "geomrecur" => 3,
        "fixed_sweeps" => 3, "ii_kill" => 6, "ii_readbefore" => 4, "ii_readnested" => 2,
        "matvec_loss" => 0, "mg_vcycle" => 28, "mg_vcycle_multi" => 28,
        "mpnn" => 0, "normcomp" => 0, "pipeline" => 0, "prefixscan" => 3,
        "quadloss" => 0, "raggedii" => 1, "raggedind" => 1, "red_escape" => 5,
        "retire_empty" => 4,
        "relu_field" => 0, "richardson_substep" => 7, "stencil_loss" => 0,
        "sumsq_shifted" => 0, "transformer" => 31, "ttgc" => 6,
        "two_field_loss" => 0, "unet" => 0, "weightedsumsq" => 0,
        "windowed_relax_retire" => 7,
    )

    count_for(e) = e isa Expr ? (e.head == :for) + sum(Int[count_for(a) for a in e.args]; init = 0) : 0

    kernels = sort(filter(readdir(dir)) do f
        endswith(f, ".jl") && !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl"))
    end)

    MODES = ((:adjoint, STADE.stade_adjoint_file, "_b.jl"),
             (:hvp,     STADE.stade_hvp_file,     "_hv.jl"))

    bad = 0
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

    checks = length(kernels) * length(MODES)
    isempty(improved) || println("\nimproved (lower the baseline): ", join(improved, ", "))
    println("\n", checks - bad, "/", checks, " checks within budget",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end
