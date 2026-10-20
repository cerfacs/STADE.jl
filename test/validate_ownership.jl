include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

# kernel => why it is flagged. Empty: the two kernels that 0.4.4 flagged (ttgc, idm_refuse_cond_read) are now kept on the host by the
# split gate (cgen_rmw_hazard_arrays, cgen_overlap_proven_arrays). A new entry here is a finding, not a place to park a failure.
own_known() = Dict{String,String}()

"""
    validate_ownership(dir="val-corpus"; only=nothing, int_lo=3, int_hi=6)

CPU check of the one assumption STADE's GPU code generator makes without proof:
that a plain (non-atomic) array write inside a kernel is disjoint across threads.

Every generated CUDA plan (adjoint and HVP) is run on the CPU through
`ownership_shim.jl`, which records which thread read, wrote, or atomically added
each element of each launch. An element is reported when two threads plain-write
it, when one thread plain-writes it while another reads or adds to it, or when
one thread reads it while another adds to it. The result does not depend on
thread order, so it also catches a hazard a real warp would hide.

The same run compares the emulated plan against the CPU adjoint, reusing the
`gval_*` helpers of validate_corpus_gpu.jl as source text (that file loads CUDA
at top level). Both must hold: values agree AND no unsafe element.

`idm_dup_write` is the documented counter-example: its source writes the same
element from two iterations, so the check MUST flag it. A run that does not flag
it fails, since a check that cannot see a known fault proves nothing.

No GPU, no CUDA package. Works on a scratch copy of the corpus.

Findings are listed in `own_known()`, each with its reason. A kernel there may be flagged without
failing the run; it is still printed, so the finding stays visible. The table is empty: the two hazards
reproduced on the unmodified 0.4.4 source (ttgc, idm_refuse_cond_read) are fixed in the gate.
"""
function validate_ownership(dir::String = joinpath(@__DIR__, "val-corpus");
                            only::Union{Vector{String},Nothing} = nothing,
                            int_lo::Int = 3, int_hi::Int = 6, trials::Int = 2,
                            modes::Vector{Symbol} = [:adjoint, :hvp],
                            orders::Vector{Symbol} = [:ascending, :shuffled],
                            redraws::Int = 12,
                            order_rtol::Float64 = 1e-9,
                            known::Dict{String,String} = own_known(),
                            shim::String = get(ENV, "OWN_SHIM", joinpath(@__DIR__, "ownership_shim.jl")),
                            expected_flagged::Vector{String} = ["idm_dup_write"],
                            verbose::Bool = true)
    include(shim)
    Core.eval(Main, :(using .CUDA))
    Core.eval(Main, :(const CuArray = CUDA.TArray))
    src = read(joinpath(@__DIR__, "validate_corpus_gpu.jl"), String)
    i = findfirst("# ---- live-execution helpers", src)[1]
    j = findfirst("# ---- top-level driver", src)[1]
    Core.eval(Main, Meta.parse("begin\n" * src[i:j-1] * "\nend"))

    work = mktempdir()
    for f in readdir(dir)
        endswith(f, ".jl") || continue
        any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl", "_cuda.jl", "_jacc.jl")) && continue
        cp(joinpath(dir, f), joinpath(work, f))
    end
    names = sort([splitext(f)[1] for f in readdir(work) if endswith(f, ".jl")])
    only !== nothing && (names = filter(n -> n in only, names))

    backend = STADE.cgen_backend_cuda()

    # One draw of inputs, then one emulated run per thread order. Violations come from the first order:
    # they describe the access pattern, which does not depend on order.
    function one_draw(name, mode, path)
        yp = STADE.io_gpu_yaml_path(path)
        isfile(yp) || STADE.stade_generate_baseline_file(path; yaml_path = yp, int_lo = int_lo, int_hi = int_hi, grow_max = 512)
        primal = STADE.io_read_corpus_entry(path)
        kernel = STADE.parse_kernel(primal)
        baseline = STADE.io_read_baseline_yaml(yp)
        STADE.val_coerce_int_arrays!(kernel, baseline.values)
        gen = mode == :hvp ? STADE.stade_hvp(primal; keep_push_pop = false) :
                             STADE.stade_adjoint(primal; keep_push_pop = false)
        gexpr = mode == :hvp ? gen.hvp : gen.adjoint
        stack_names = Symbol[a for a in gen.initstacks.args[1].args[2:end]]
        seeds = [STADE.val_random_values_like(kernel, baseline.values) for _ in 1:trials]
        tans  = mode == :hvp ? [STADE.val_random_values_like(kernel, baseline.values) for _ in 1:trials] : Any[]
        shts  = mode == :hvp ? [STADE.val_random_values_like(kernel, baseline.values) for _ in 1:trials] : Any[]
        pinit = STADE.stade_gpu(gen.initstacks, backend)
        pgen  = STADE.stade_gpu(gexpr, backend)
        res = Dict{Symbol,Any}()
        viol = NamedTuple[]
        for order in orders
            Main.CUDA.ORDER[] = order
            empty!(Main.CUDA.VIOLATIONS)
            res[order] = Base.invokelatest(Main.gval_check_backend, kernel, baseline.int_args, baseline.values, seeds,
                                           gen, pinit, pgen, stack_names, Main.CUDA.TArray, Array, backend.atomic_macro;
                                           mode, tan_seeds = tans, shtan_seeds = shts)
            order == orders[1] && (viol = unique(copy(Main.CUDA.VIOLATIONS)))
        end
        Main.CUDA.ORDER[] = :ascending
        return (parity = res[orders[1]].ok, shuffled_parity = all(o -> res[o].ok || (!res[o].vacuous && res[o].max_rel_err <= order_rtol), orders),
                errs = Tuple(res[o].max_rel_err for o in orders), rtol = res[orders[1]].rtol,
                vacuous = res[orders[1]].vacuous, violations = viol, error = "")
    end

    rows = NamedTuple[]
    for name in names, mode in modes
        path = joinpath(work, name * ".jl")
        # The counter-example is only caught on a draw where two iterations really do collide, so it gets
        # fresh draws until the check flags it. Every other kernel runs on its one draw.
        ndraw = name in expected_flagged ? redraws : 1
        row = nothing
        for d in 1:ndraw
            d > 1 && rm(STADE.io_gpu_yaml_path(path); force = true)
            row = try
                one_draw(name, mode, path)
            catch e
                (parity = false, shuffled_parity = false, errs = (NaN,), rtol = NaN, vacuous = false, violations = NamedTuple[],
                 error = first(split(sprint(showerror, e), "\n")))
            end
            (!isempty(row.violations) || !isempty(row.error)) && break
        end
        push!(rows, merge((kernel = name, mode = mode), row))
    end

    # Three kinds of kernel. expected: the check MUST flag it. known: a flag is recorded and allowed, with the
    # reason in `known`. every other kernel: no flag, no error, same result under every thread order.
    bad = 0
    for r in rows
        flagged = !isempty(r.violations)
        expect = r.kernel in expected_flagged
        isknown = haskey(known, r.kernel)
        ok = isempty(r.error) && (expect ? flagged : (isknown || !flagged) && (isknown || (r.parity && r.shuffled_parity)))
        ok || (bad += 1)
        if verbose && (!ok || flagged || !r.shuffled_parity)
            tag = !ok ? " *** NOT OK ***" : expect ? " ok (expected flag)" : isknown ? " known: " * known[r.kernel] : " ok"
            println(rpad("$(r.kernel) [$(r.mode)]", 36), tag,
                    isempty(r.error) ? "" : "  error: " * r.error,
                    "  parity ascending=", r.parity, " all orders=", r.shuffled_parity,
                    "  max_rel_err=", r.errs, " rtol=", r.rtol)
            for v in r.violations
                println("    ", v.kernel, " array=", v.array, " ", v.kind, " x", v.count,
                        " e.g. element ", v.example_index, " threads ", v.threads)
            end
        end
    end
    println(length(rows) - bad, "/", length(rows), " ownership checks passed",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return (rows = rows, bad = bad)
end

if abspath(PROGRAM_FILE) == @__FILE__
    only = length(ARGS) > 0 ? ARGS : nothing
    exit(validate_ownership(; only = only).bad == 0 ? 0 : 1)
end
