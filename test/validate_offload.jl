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
    # CSE coverage kernels, at the loop counts they reach today. cse_branch and
    # cse_intoffset keep only their row loops on the host. cse_zerotrip keeps 6 of 7,
    # the same shape as entry_empty (7): its window is a runtime scalar reassigned each
    # pass, so cgen_last_assign_is_zero cannot prove the zeroing write happens and
    # declines to split. Each kernel's HVP matches its adjoint, which is the property
    # this file exists to hold.
        "cse_branch" => 1, "cse_zerotrip" => 6, "cse_intoffset" => 2,
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

    # Backend agreement used to be checked here too. It now lives in
    # validate_backend_agreement.jl, which compares loop shapes and kernel arities
    # rather than bare counts and knows that JACC's splat ceiling makes one pair
    # legitimately diverge. Two implementations of one rule drift apart; this one
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
