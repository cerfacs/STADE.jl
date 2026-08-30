include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_gpu_arglayout(dir="val-corpus")

Check that every GPU parity harness builds its call in the argument order the
generated function actually declares, for both `:adjoint` and `:hvp`.

Three independent pieces have to agree on one ordering rule:

  * `STADE.val_gpu_call_args`  -- names emitted into a submittable script
  * `gval_build_call`          -- values built in-process by validate_corpus_gpu.jl
  * the generated `_b`/`_hv` function's own parameter list

Nothing enforced that agreement before. It matters most for the HVP, whose
layout is the whole adjoint list followed by one `(tangent, tangent-of-shadow)`
pair per float argument -- a shape that cannot be derived from `sig.kinds`
alone, so both consumers reimplement it. Two arrays of the same length swapped
between those slots does not raise anywhere: it runs, and produces wrong
numbers that only a live GPU comparison would catch.

Every slot is filled with a unique sentinel, so a mismatch reports which
argument and which role landed in the wrong position rather than just a count.

This script needs no GPU and never loads CUDA or JACC -- `gval_build_call` is
lifted out of validate_corpus_gpu.jl as source text, since that file's own
top-level `using CUDA` would otherwise be required.
"""
function validate_gpu_arglayout(dir::String = joinpath(@__DIR__, "val-corpus"))
    src = read(joinpath(@__DIR__, "validate_corpus_gpu.jl"), String)
    i = findfirst("function gval_build_call", src)[1]
    j = findfirst("\nend\n", src[i:end])[1] + i
    build = Core.eval(Main, Meta.parse(src[i:j+3]))

    kernels = sort(filter(readdir(dir)) do f
        endswith(f, ".jl") && !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl", "_cuda.jl", "_jacc.jl"))
    end)

    bad = 0
    checks = 0
    for f in kernels
        name = splitext(f)[1]
        primal = STADE.io_read_corpus_entry(joinpath(dir, f))
        sig = STADE.parse_kernel(primal).sig

        tag = Dict{Float64,String}()
        int_args = Dict{Symbol,Any}()
        values = Dict{Symbol,Any}(); seed = Dict{Symbol,Any}()
        dvalues = Dict{Symbol,Any}(); dseed = Dict{Symbol,Any}()
        for (idx, a) in enumerate(sig.args)
            k = sig.kinds[a]
            if k == :scalar_int
                int_args[a] = 900000 + idx
                tag[Float64(900000 + idx)] = string(a)
                continue
            end
            for (off, role, dct) in ((1, "_val", values), (2, "_sh", seed),
                                     (3, "_tan", dvalues), (4, "_shtan", dseed))
                id = off * 100000 + idx
                dct[a] = k in (:array_float, :array_int) ? [Float64(id)] : Float64(id)
                tag[Float64(id)] = string(a) * role
            end
        end

        for mode in (:adjoint, :hvp)
            want = [replace(x, "_cpu" => "") for x in STADE.val_gpu_call_args(sig, :cpu; mode)]
            gen = mode == :hvp ? STADE.stade_hvp(primal; keep_push_pop = false) :
                                 STADE.stade_adjoint(primal; keep_push_pop = false)
            declared = Symbol[a for a in (mode == :hvp ? gen.hvp : gen.adjoint).args[1].args[2:end]]
            r = gen.initstacks.args[2].args[end].args[1]
            n_stack = (r === nothing || r === :nothing) ? 0 : (r isa Symbol ? 1 : length(r.args))
            n_declared = length(declared) - n_stack

            args, _ = Base.invokelatest(build, sig, int_args, values, seed, identity;
                                        mode = mode, dvalues = dvalues, dseed = dseed)
            got = [tag[x isa AbstractArray ? x[1] : Float64(x)] for x in args]

            checks += 1
            label = rpad("$(name) [$(mode)]", 34)
            if got != want
                println(label, " FAIL  live runner disagrees with the script generator")
                for (g, w) in zip(got, want)
                    g == w || println("      live=", g, "   generator=", w)
                end
                bad += 1
            elseif length(want) != n_declared
                println(label, " FAIL  arity ", length(want), " but the generated function declares ", n_declared)
                bad += 1
            else
                println(label, " ok  ", length(want), " args + ", n_stack, " stacks")
            end
        end
    end

    println("\n", checks - bad, "/", checks, " (kernel, mode) call layouts agree",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end

validate_gpu_arglayout()
