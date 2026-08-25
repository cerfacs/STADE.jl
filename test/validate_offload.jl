include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_offload(dir="val-corpus")

Check how much of each corpus kernel's adjoint reaches the GPU.

`validate_corpus.jl` checks that generated code computes the right numbers. It
says nothing about whether that code can run on a device, so a change that
silently pushes a loop back onto the host passes every oracle. This script
measures the property directly: for each kernel it generates the adjoint with
`keep_push_pop = false` (the only GPU-eligible mode), converts it with the CUDA
backend, and counts `for` loops before and after.

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
        "advection" => 3, "advection_multi" => 3, "affine_loss" => 0,
        "bilinear" => 0, "branchsel" => 0, "cascadic_mg_prolong" => 22,
        "clamped_sumsq" => 0, "coarsen_retire" => 4, "cond_field_choice" => 0,
        "cond_loop_choice" => 0, "dotprod" => 0, "geomrecur" => 3,
        "matvec_loss" => 0, "mg_vcycle" => 29, "mg_vcycle_multi" => 29,
        "mpnn" => 0, "normcomp" => 0, "pipeline" => 0, "prefixscan" => 3,
        "quadloss" => 0, "raggedii" => :refused, "raggedind" => 3,
        "relu_field" => 0, "richardson_substep" => 7, "stencil_loss" => 0,
        "sumsq_shifted" => 0, "transformer" => 63, "ttgc" => 18,
        "two_field_loss" => 0, "unet" => 0, "weightedsumsq" => 0,
        "windowed_relax_retire" => 7,
    )

    count_for(e) = e isa Expr ? (e.head == :for) + sum(Int[count_for(a) for a in e.args]; init = 0) : 0

    kernels = sort(filter(readdir(dir)) do f
        endswith(f, ".jl") && !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl"))
    end)

    bad = 0
    improved = String[]
    for f in kernels
        name = splitext(f)[1]
        haskey(EXPECTED, name) || (println(rpad(name, 26), " NO BASELINE -- add one"); bad += 1; continue)
        out = joinpath(dir, name * "_b.jl")
        total = 0
        host = 0
        status = :ok
        try
            STADE.stade_adjoint_file(joinpath(dir, f), out; keep_push_pop = false, fuse_ii_loops = true)
            for e in STADE.io_read_kernel_bundle(out)
                total += count_for(e)
                host += count_for(STADE.stade_gpu(e, STADE.cgen_backend_cuda()).host)
            end
        catch
            status = :refused
        end
        want = EXPECTED[name]
        if status == :refused
            ok = want === :refused
            println(rpad(name, 26), ok ? " ok" : " FAIL", "  ingest refused (expected $(want))")
            ok || (bad += 1)
        elseif want === :refused
            println(rpad(name, 26), " FAIL  now ingests (baseline says refused) -- update the baseline")
            bad += 1
        else
            println(rpad(name, 26), host <= want ? " ok" : " FAIL",
                    "  host_left=", host, "/", total, " (max ", want, ")")
            host > want && (bad += 1)
            host < want && push!(improved, "$(name): $(want) -> $(host)")
        end
    end

    isempty(improved) || println("\nimproved (lower the baseline): ", join(improved, ", "))
    println("\n", length(kernels) - bad, "/", length(kernels), " kernels within budget",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end
