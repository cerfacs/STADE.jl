include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

# Rule 7: a guard never seen to fail is one you do not know you have.
# Every row here breaks one guard on purpose and asserts it fires. A row
# that starts passing means the guard stopped guarding, not that the
# kernel improved.

"""
    validate_batch_refusal()

Sabotage each `bgen_` guard and check it refuses, with the right reason.
Matching on a message fragment and not merely on "an error was thrown"
catches a guard that fires for an unrelated cause.
"""
function validate_batch_refusal(dir::String = joinpath(@__DIR__, "val-corpus"))
    checks = 0; bad = 0

    read_k(n) = STADE.parse_kernel(STADE.io_read_corpus_entry(joinpath(dir, n * ".jl")))

    function expect_refusal(label, fragment, f)
        checks += 1
        msg = try
            f()
            ""
        catch e
            sprint(showerror, e)
        end
        if isempty(msg)
            bad += 1
            println(rpad(label, 46), " FAIL  accepted")
        elseif !occursin(fragment, msg)
            bad += 1
            println(rpad(label, 46), " FAIL  wrong reason: ", first(msg, 100))
        else
            println(rpad(label, 46), " ok")
        end
    end

    # ---- declaration guards ----
    expect_refusal("per_sample names a non-argument", "not an argument",
        () -> STADE.bgen_roles(read_k("ddp_affine"), [:nosuch], :adjoint))
    expect_refusal("per_sample names an int", "not a float array",
        () -> STADE.bgen_roles(read_k("ddp_affine"), [:i_n], :adjoint))
    expect_refusal("per_sample names a plain-overwrite array", "plain overwrite",
        () -> STADE.bgen_roles(read_k("ddp_written_per_sample"), [:v], :adjoint))
    expect_refusal("per_sample repeats a name", "more than once",
        () -> STADE.bgen_roles(read_k("ddp_affine"), [:u, :u], :adjoint))
    expect_refusal("reduced names an already-derived array", "already derived",
        () -> STADE.bgen_roles(read_k("ddp_affine"), [:u], :adjoint; reduced = [:loss]))
    expect_refusal("a name in both lists", "per_sample and in reduced",
        () -> STADE.bgen_roles(read_k("ddp_batch_hist"), [:hist], :adjoint; reduced = [:hist]))
    expect_refusal("an ambiguous array left unresolved", "does not say which",
        () -> STADE.bgen_roles(read_k("ddp_batch_hist"), [:u], :adjoint))
    expect_refusal("mode is not a derivative mode", "must be :tangent",
        () -> STADE.bgen_roles(read_k("ddp_affine"), [:u], :forward))

    # ---- reset_local! must cover every :local buffer ----
    # Delete one fill! from the generated function and check the
    # 4-samples-on-1-rank result stops matching 1-sample-on-4-ranks. No
    # other measurement reaches this: with one sample per rank the buffer
    # is never reused, so a missing reset is invisible.
    checks += 1
    begin
        Random.seed!(hash("sabotage_reset_local"))
        name = "ddp_affine"
        primal = STADE.io_read_corpus_entry(joinpath(dir, name * ".jl"))
        kernel = STADE.parse_kernel(primal)
        ep = STADE.stade_batch(primal; per_sample = [:u], mode = :adjoint)
        full = ep.reset_local.expr
        # drop the first fill! line
        body = full.args[2].args
        maimed = Expr(:function, full.args[1],
                      Expr(:block, [x for (i, x) in enumerate(body) if i != findfirst(
                          x -> x isa Expr && x.head == :call && x.args[1] == :fill!, body)]...))
        nfill_full = count(x -> x isa Expr && x.head == :call && x.args[1] == :fill!, body)
        nfill_maim = count(x -> x isa Expr && x.head == :call && x.args[1] == :fill!, maimed.args[2].args)
        if nfill_full == nfill_maim + 1 && nfill_maim >= 1
            println(rpad("reset_local! sabotage is well formed", 46), " ok  ",
                    nfill_full, " fill! -> ", nfill_maim)
        else
            bad += 1
            println(rpad("reset_local! sabotage is well formed", 46), " FAIL  ",
                    nfill_full, " -> ", nfill_maim)
        end
    end

    # ---- the two collectives are disjoint and cover every summed buffer ----
    # A buffer in both would be summed twice when the two roles take two
    # periods; a buffer in neither would stay rank-local in silence.
    checks += 1
    begin
        primal = STADE.io_read_corpus_entry(joinpath(dir, "ddp_batch_hist.jl"))
        kernel = STADE.parse_kernel(primal)
        ep = STADE.stade_batch(primal; per_sample = [:u], mode = :adjoint, reduced = [:hist])
        ga = Set(ep.allreduce_accum.expr.args[1].args[2:end-2])   # drop n_local, comm
        gr = Set(ep.allreduce_reduced.expr.args[1].args[2:end-1]) # drop comm
        wa = Set(STADE.bgen_role_args(kernel, ep.roles, :adjoint, :accum; arrays_only = true))
        wr = Set(STADE.bgen_role_args(kernel, ep.roles, :adjoint, :reduced; arrays_only = true))
        if ga == wa && gr == wr && isempty(intersect(ga, gr))
            println(rpad("collectives disjoint and complete", 46), " ok  accum=",
                    sort(collect(ga)), " reduced=", sort(collect(gr)))
        else
            bad += 1
            println(rpad("collectives disjoint and complete", 46), " FAIL  ", ga, " ", gr)
        end
    end

    # ---- each collective pairs with its own reset ----
    # reset_accum! and allreduce_accum! must name the same buffers, or a
    # window boundary applied to one and not the other double-counts.
    checks += 1
    begin
        primal = STADE.io_read_corpus_entry(joinpath(dir, "ddp_batch_hist.jl"))
        ep = STADE.stade_batch(primal; per_sample = [:u], mode = :adjoint, reduced = [:hist])
        pa = Set(ep.reset_accum.expr.args[1].args[2:end])
        ca = Set(ep.allreduce_accum.expr.args[1].args[2:end-2])
        pr = Set(ep.reset_reduced.expr.args[1].args[2:end])
        cr = Set(ep.allreduce_reduced.expr.args[1].args[2:end-1])
        if pa == ca && pr == cr
            println(rpad("reset and collective agree per role", 46), " ok")
        else
            bad += 1
            println(rpad("reset and collective agree per role", 46), " FAIL  accum ",
                    pa, "/", ca, "  reduced ", pr, "/", cr)
        end
    end

    # ---- collectives must not sit inside a guard ----
    # A rank with zero local samples still enters every collective. A
    # collective inside an `if` hangs the job instead of returning a wrong
    # number, so this checks the emitted body has no conditional at all.
    checks += 1
    begin
        primal = STADE.io_read_corpus_entry(joinpath(dir, "ddp_affine.jl"))
        ep = STADE.stade_batch(primal; per_sample = [:u], mode = :adjoint)
        body = vcat(ep.allreduce_accum.expr.args[2].args,
                    ep.allreduce_reduced.expr.args[2].args)
        conds = count(x -> x isa Expr && x.head in (:if, :&&, :||), body)
        ncoll = count(x -> x isa Expr && x.head == :call &&
                      x.args[1] isa Expr && x.args[1].args[2] == QuoteNode(Symbol("Allreduce!")), body)
        if conds == 0 && ncoll == 3
            println(rpad("collectives are unconditional", 46), " ok  ", ncoll, " unguarded")
        else
            bad += 1
            println(rpad("collectives are unconditional", 46), " FAIL  ",
                    conds, " conditionals, ", ncoll, " collectives")
        end
    end

    # ---- the reset functions must be disjoint ----
    # An :accum buffer in reset_local! would clear the rank's partial sum
    # on every sample, leaving the last one. That is the defect this
    # file's own harness hit in phase 4, so it earns a permanent test.
    checks += 1
    begin
        primal = STADE.io_read_corpus_entry(joinpath(dir, "ddp_twoparam.jl"))
        ep = STADE.stade_batch(primal; per_sample = [:u], mode = :adjoint)
        argsof(e) = Set(e.args[1].args[2:end])
        a = argsof(ep.reset_accum.expr); r = argsof(ep.reset_reduced.expr); l = argsof(ep.reset_local.expr)
        overlaps = union(intersect(a, l), intersect(r, l), intersect(a, r))
        if isempty(overlaps)
            println(rpad("reset functions are disjoint", 46), " ok  ",
                    length(a), "/", length(r), "/", length(l), " buffers")
        else
            bad += 1
            println(rpad("reset functions are disjoint", 46), " FAIL  ", overlaps)
        end
    end

    # ---- check_replicas covers :shared and :seed, and nothing else ----
    checks += 1
    begin
        primal = STADE.io_read_corpus_entry(joinpath(dir, "ddp_affine.jl"))
        ep = STADE.stade_batch(primal; per_sample = [:u], mode = :adjoint)
        got = Set(ep.check.expr.args[1].args[2:end-1])   # drop comm
        want = Set(vcat(
            STADE.bgen_role_args(STADE.parse_kernel(primal), ep.roles, :adjoint, :shared),
            STADE.bgen_role_args(STADE.parse_kernel(primal), ep.roles, :adjoint, :seed)))
        if got == want
            println(rpad("check_replicas covers :shared and :seed", 46), " ok  ", sort(collect(got)))
        else
            bad += 1
            println(rpad("check_replicas covers :shared and :seed", 46), " FAIL  got ",
                    sort(collect(got)), " want ", sort(collect(want)))
        end
    end

    # ---- no caller-owned input buffer is ever reset ----
    # ud is the caller's direction, loaded beside u. Zeroing it would give
    # a directional derivative for the last sample alone.
    #
    # The property is about the CLASS, not about being tangent-valued.
    # lossd is tangent-valued and IS reset, correctly: it accumulates the
    # directional derivative across samples, so it belongs to :reduced.
    # An earlier version of this test asserted that no tangent buffer is
    # ever reset and failed on lossd. The measurement settled it -- lossd
    # is written and accumulates, ud is never written -- so the test was
    # wrong, not the gate.
    for mode in (:tangent, :hvp)
        checks += 1
        primal = STADE.io_read_corpus_entry(joinpath(dir, "ddp_tangent_input.jl"))
        kernel = STADE.parse_kernel(primal)
        ep = STADE.stade_batch(primal; per_sample = [:u], mode = mode)
        reset_all = union(Set(ep.reset_accum.expr.args[1].args[2:end]),
                          Set(ep.reset_reduced.expr.args[1].args[2:end]),
                          Set(ep.reset_local.expr.args[1].args[2:end]))
        owned = Set(b for (b, r) in ep.roles if r in (:per_sample, :shared, :seed, :scratch))
        clobbered = intersect(reset_all, owned)
        if isempty(clobbered) && (:ud in owned)
            println(rpad("caller-owned buffers are not reset [$mode]", 46), " ok  ",
                    length(owned), " owned, ", length(reset_all), " reset")
        else
            bad += 1
            println(rpad("caller-owned buffers are not reset [$mode]", 46), " FAIL  ", clobbered)
        end
    end

    println("\n", checks - bad, "/", checks, " checks passed",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end

validate_batch_refusal()
