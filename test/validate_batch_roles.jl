include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_batch_roles()

Phase 1 gate for the `bgen_` stage. Checks the derived role of every
buffer against a table written by hand, and checks that every negative
subject refuses.

The expected tables below are written from the primal signature and the
two axes of section 1.5 of the plan, not from a run. A table that agrees
with the code because it was copied from the code proves nothing.
"""
function validate_batch_roles(dir::String = joinpath(@__DIR__, "val-corpus"))
    # subject => (per_sample, adjoint-mode role of every float buffer)
    expected = Dict(
        "ddp_affine" => ([:u], Dict(
            :loss => :reduced, :lossb => :seed,
            :u => :per_sample, :ub => :local,
            :a => :shared, :ab => :accum,
            :b => :shared, :bb => :accum,
            :v => :scratch, :vb => :local)),
        "ddp_twoparam" => ([:u], Dict(
            :loss => :reduced, :lossb => :seed,
            :u => :per_sample, :ub => :local,
            :w => :shared, :wb => :accum,
            :b => :shared, :bb => :accum,
            :h => :scratch, :hb => :local,
            :o => :scratch, :ob => :local)),
        "ddp_ragged" => ([:u], Dict(
            :loss => :reduced, :lossb => :seed,
            :u => :per_sample, :ub => :local,
            :a => :shared, :ab => :accum,
            :v => :scratch, :vb => :local)),
        "ddp_branch" => ([:u], Dict(
            :loss => :reduced, :lossb => :seed,
            :u => :per_sample, :ub => :local,
            :a => :shared, :ab => :accum,
            :b => :shared, :bb => :accum,
            :v => :scratch, :vb => :local)),
        "ddp_tangent_input" => ([:u], Dict(
            :loss => :reduced, :lossb => :seed,
            :u => :per_sample, :ub => :local,
            :a => :shared, :ab => :accum,
            :v => :scratch, :vb => :local)),
        "ddp_batch_hist" => ([:u], Dict(
            :loss => :reduced, :lossb => :seed,
            :u => :per_sample, :ub => :local,
            :a => :shared, :ab => :accum,
            :hist => :reduced, :histb => :seed,
            :v => :scratch, :vb => :local)),
        "affine_loss" => ([:u], Dict(
            :loss => :reduced, :lossb => :seed,
            :u => :per_sample, :ub => :local,
            :a => :shared, :ab => :accum,
            :b => :shared, :bb => :accum,
            :v => :scratch, :vb => :local)),
    )

    # subject => (per_sample, a fragment the error message must contain)
    refusals = [
        ("ddp_written_per_sample", [:v],      Symbol[], "plain overwrite"),
        ("ddp_ambiguous_reduce",   [:u],      Symbol[], "does not say which"),
        ("matvec_loss",            [:u],      Symbol[], "does not say which"),
        ("ddp_batch_hist",         [:u],      Symbol[], "does not say which"),
        ("ddp_affine",             [:nosuch], Symbol[], "not an argument"),
        ("ddp_affine",             [:i_n],    Symbol[], "not a float array argument"),
        ("ddp_affine",             [:u],      [:loss],  "already derived"),
        ("ddp_batch_hist",         [:hist],   [:hist],  "per_sample and in reduced"),
    ]


    checks = 0; bad = 0

    # both resolutions of an ambiguous array must be accepted, and must
    # give the two different classes they name
    resolutions = [
        ("matvec_loss",    [:u, :v], Symbol[], :v, :scratch),
        ("ddp_batch_hist", [:u],     [:hist],  :hist, :reduced),
    ]
    for (name, ps, rd, arr, want) in resolutions
        k = STADE.parse_kernel(STADE.io_read_corpus_entry(joinpath(dir, name * ".jl")))
        checks += 1
        got = try STADE.bgen_roles(k, ps, :adjoint; reduced = rd)[arr] catch e; Symbol(sprint(showerror, e)) end
        if got === want
            println(rpad(name * " [resolve " * string(arr) * "]", 34), " ok  ", want)
        else
            bad += 1
            println(rpad(name * " [resolve " * string(arr) * "]", 34), " FAIL  got ", got)
        end
    end

    for name in sort(collect(keys(expected)))
        per_sample, want = expected[name]
        reduced = name == "ddp_batch_hist" ? [:hist] : Symbol[]
        k = STADE.parse_kernel(STADE.io_read_corpus_entry(joinpath(dir, name * ".jl")))
        got = STADE.bgen_roles(k, per_sample, :adjoint; reduced = reduced)
        checks += 1
        # every expected buffer present with the expected role, and no extras
        missing_k = [b for b in sort(collect(keys(want))) if !haskey(got, b)]
        extra_k   = [b for b in sort(collect(keys(got))) if !haskey(want, b)]
        wrong     = [(b, want[b], got[b]) for b in sort(collect(keys(want)))
                     if haskey(got, b) && got[b] != want[b]]
        if isempty(missing_k) && isempty(extra_k) && isempty(wrong)
            println(rpad(name * " [roles]", 34), " ok  ", length(want), " buffers")
        else
            bad += 1
            println(rpad(name * " [roles]", 34), " FAIL")
            isempty(missing_k) || println("    missing: ", missing_k)
            isempty(extra_k)   || println("    extra:   ", extra_k)
            for (b, w, g) in wrong
                println("    ", b, ": expected ", w, ", got ", g)
            end
        end
    end

    # Every mode must give the same class to a primal argument. Only the
    # companion side changes, so a mode-dependent class is a defect.
    for name in sort(collect(keys(expected)))
        per_sample, _ = expected[name]
        reduced = name == "ddp_batch_hist" ? [:hist] : Symbol[]
        k = STADE.parse_kernel(STADE.io_read_corpus_entry(joinpath(dir, name * ".jl")))
        rs = Dict(m => STADE.bgen_roles(k, per_sample, m; reduced = reduced) for m in (:tangent, :adjoint, :hvp))
        checks += 1
        prim = [a for a in k.sig.args if k.sig.kinds[a] in (:scalar_float, :array_float)]
        diff = [a for a in prim if !(rs[:tangent][a] == rs[:adjoint][a] == rs[:hvp][a])]
        if isempty(diff)
            println(rpad(name * " [mode agreement]", 34), " ok")
        else
            bad += 1
            println(rpad(name * " [mode agreement]", 34), " FAIL  ", diff)
        end
    end

    # In tangent mode a tangent companion of a per_sample argument must
    # NOT be :local. It is an input the caller loads, not a buffer the
    # epilogue clears.
    for name in sort(collect(keys(expected)))
        per_sample, _ = expected[name]
        reduced = name == "ddp_batch_hist" ? [:hist] : Symbol[]
        k = STADE.parse_kernel(STADE.io_read_corpus_entry(joinpath(dir, name * ".jl")))
        roles = STADE.bgen_roles(k, per_sample, :hvp; reduced = reduced)
        checks += 1
        wrong = [STADE.tgen_shadow(p) for p in per_sample
                 if roles[STADE.tgen_shadow(p)] === :local]
        if isempty(wrong)
            println(rpad(name * " [tangent not local]", 34), " ok")
        else
            bad += 1
            println(rpad(name * " [tangent not local]", 34), " FAIL  ", wrong, " marked :local")
        end
    end

    for (name, per_sample, rd, fragment) in refusals
        k = STADE.parse_kernel(STADE.io_read_corpus_entry(joinpath(dir, name * ".jl")))
        checks += 1
        msg = try
            STADE.bgen_roles(k, per_sample, :adjoint; reduced = rd)
            ""
        catch e
            sprint(showerror, e)
        end
        if isempty(msg)
            bad += 1
            println(rpad(name * " [refusal]", 34), " FAIL  accepted ", per_sample)
        elseif !occursin(fragment, msg)
            bad += 1
            println(rpad(name * " [refusal]", 34), " FAIL  wrong reason: ", first(msg, 90))
        else
            println(rpad(name * " [refusal]", 34), " ok  refused ", per_sample)
        end
    end

    println("\n", checks - bad, "/", checks, " checks passed",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end

validate_batch_roles()
