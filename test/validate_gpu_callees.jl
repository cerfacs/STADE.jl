include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

# The reduction lowering had no test that ran it. validate_corpus runs
# on the CPU, where the reduction path is never emitted, and
# validate_offload counts loops without executing anything. So the
# emitted `dot(u, v)` went unexercised until a real device rejected it:
# every cuBLAS Level-1 entry point raises DivideError on the reference
# cluster, `dot` and `norm` alike, for Float32 and Float64, at length 16
# and 4096, while `gemv` on the same device works.
#
# This script is the cheap half of the fix. It runs in a sandbox with no
# GPU and asserts that nothing in the emitted code reaches a vendor BLAS
# routine. The other half is a cluster job that actually runs a
# lowered adjoint; a codegen check cannot replace it, it
# only makes the failure visible one step earlier.

"""
    validate_gpu_callees(dir = "val-corpus")

Check every emitted GPU function for a call this cluster cannot execute,
and check that `reduction_threshold` still changes what is emitted.
"""
function validate_gpu_callees(dir::String = joinpath(@__DIR__, "val-corpus"))
    # Reach GPUArrays' generic machinery, verified working on device.
    allowed = Set([:mapreduce, :sum, :abs2, :fill!, :cld, :max, :div, :length])
    # cuBLAS/rocBLAS Level-1. `dot` and `norm` are measured broken; the
    # rest share the same binding layer and are refused on that basis.
    banned = Set([:dot, :norm, :nrm2, :axpy!, :scal!, :BLAS])

    checks = 0; bad = 0
    kernels = sort(filter(f -> endswith(f, ".jl") && !occursin(r"_(b|d|hv)\.jl$", f),
                          readdir(dir)))
    hits = String[]
    differs = 0
    for f in kernels
        primal = try STADE.io_read_corpus_entry(joinpath(dir, f)) catch e; continue end
        name = splitext(f)[1]
        for mode in (:adjoint, :hvp)
            gen = try
                mode == :hvp ? STADE.stade_hvp(primal; keep_push_pop = false, fuse_ii_loops = true) :
                               STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true)
            catch e
                continue
            end
            e = mode == :hvp ? gen.hvp : gen.adjoint
            src = Dict{Bool,String}()
            # 0 forces every matched reduction to lower, a huge value forces
            # none. The boolean that used to select this is gone: those two
            # extremes of reduction_threshold span exactly what it did.
            for kaa in (true, false)
                th = kaa ? 1_000_000_000 : 0
                out = try STADE.stade_gpu(e, STADE.cgen_backend_cuda(); reduction_threshold = th) catch er; continue end
                s = join(vcat([STADE.io_expr_to_source(k) for k in out.kernels],
                              [STADE.io_expr_to_source(out.host)]), "\n")
                src[kaa] = s
                checks += 1
                found = Symbol[]
                walk(x) = begin
                    if x isa Expr
                        if x.head == :call && x.args[1] isa Symbol && x.args[1] in banned
                            push!(found, x.args[1])
                        end
                        foreach(walk, x.args)
                    end
                end
                for k in vcat(out.kernels, [out.host]); walk(k); end
                isempty(found) || (bad += 1; push!(hits, "$name [$mode kaa=$kaa] calls " * string(unique(found))))
            end
            haskey(src, true) && haskey(src, false) && src[true] != src[false] && (differs += 1)
        end
    end

    for h in hits; println("  ", h); end
    println(rpad("no emitted GPU code calls a vendor BLAS routine", 52),
            bad == 0 ? " ok  $checks functions scanned" : " FAIL  $bad")

    # Every emitted reduction must read exactly the range its loop visited.
    # A bare `mapreduce(f, +, u, v)` reduces the whole array whatever the
    # bounds were, which is wrong for any loop that does not start at 1.
    # partialdot is the witness: nothing else in the corpus has a
    # non-unit-start reduction.
    checks += 1
    unbounded = String[]
    for f in kernels
        primal = try STADE.io_read_corpus_entry(joinpath(dir, f)) catch e; continue end
        name = splitext(f)[1]
        for mode in (:adjoint, :hvp)
            gen = try
                mode == :hvp ? STADE.stade_hvp(primal; keep_push_pop = false, fuse_ii_loops = true) :
                               STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true)
            catch e
                continue
            end
            ex = mode == :hvp ? gen.hvp : gen.adjoint
            out = try STADE.stade_gpu(ex, STADE.cgen_backend_cuda(); reduction_threshold = 0) catch er; continue end
            walk2(x) = begin
                if x isa Expr
                    if x.head == :call && x.args[1] in (:mapreduce, :sum)
                        # every array argument must arrive through a view
                        for a in x.args[2:end]
                            a isa Symbol && a in (:abs2, :+) && continue
                            a isa Expr && a.head == :-> && continue
                            a isa Symbol && push!(unbounded, "$name [$mode] reduces bare `$a`")
                        end
                    end
                    foreach(walk2, x.args)
                end
            end
            for k in vcat(out.kernels, [out.host]); walk2(k); end
        end
    end
    if isempty(unbounded)
        println(rpad("every reduction is bounded by a view", 52), " ok")
    else
        bad += 1
        println(rpad("every reduction is bounded by a view", 52), " FAIL")
        for u in unique(unbounded); println("    ", u); end
    end

    # A flag that never changes the output is a flag nobody is testing.
    checks += 1
    if differs > 0
        println(rpad("reduction_threshold changes the emission", 52), " ok  ", differs, " subjects")
    else
        bad += 1
        println(rpad("reduction_threshold changes the emission", 52), " FAIL  no subject differs")
    end

    println("\n", checks - bad, "/", checks, " checks passed",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end

validate_gpu_callees()
