include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

# `:while` adds a statement kind, and about 60 walkers across 8 stages
# recurse through statement lists. A walker that never learned about it
# returns WITHOUT visiting the body, so every name inside the loop goes
# uncollected. Nothing throws. The kernel simply differentiates as though
# the loop were empty.
#
# Reading 60 diffs would not catch that reliably. This does: one kernel
# whose `while` body writes names that appear nowhere else, and one
# assertion per collector that it reports them.

const WHILE_PROBE = quote
    function while_probe(loss, u, a, v, i_n)
        i_k = 1
        while i_k <= i_n
            wonly_s = a[i_k] * u[i_k]
            v[i_k] = wonly_s * wonly_s
            i_k = i_k + 1
        end
        for i_x = 1:i_n
            loss[1] = loss[1] + v[i_x] ^ 2
        end
        return nothing
    end
end

"""
    validate_while_walkers()

Check that every statement-list walker descends into a `:while` body.
Each entry names a collector and the name it must find; `wonly_s` and `v`
are written only inside the loop.
"""
function validate_while_walkers()
    expr = WHILE_PROBE.args[findfirst(a -> a isa Expr && a.head == :function, WHILE_PROBE.args)]
    kernel = STADE.parse_kernel(expr)
    checks = 0; bad = 0

    function expect(label, want::Symbol, got)
        checks += 1
        names = got isa Set ? got : Set(got)
        if want in names
            println(rpad(label, 46), " ok")
        else
            bad += 1
            println(rpad(label, 46), " FAIL  `", want, "` not reported")
        end
    end

    # --- parse_: the shape itself carries the body ---
    checks += 1
    w = kernel.body[2]
    if w.kind === :while && length(w.body) == 3
        println(rpad("parse_ builds a :while with its body", 46), " ok")
    else
        bad += 1
        println(rpad("parse_ builds a :while with its body", 46), " FAIL  ", w.kind)
    end

    # --- shape_: a name written only in the loop must get a kind ---
    expect("shape_ infers a kind for a while-local", :wonly_s, keys(kernel.sig.kinds))

    # --- act_: activity must propagate through the body ---
    expect("act_ sees a while-body write", :wonly_s, keys(STADE.act_analyze(kernel)))

    # --- agen_: reassignment collection ---
    expect("agen_collect_reassigned descends", :i_k,
           STADE.agen_collect_reassigned(kernel.body))

    # --- snap_: value-needed analysis ---
    # snap_: a while body's own local must reach the value-needed analysis
    # through the ordinary walk, exactly as a `for` body's does. The probe
    # uses wonly_s NONLINEARLY, because the analysis reports values a
    # partial derivative needs, not every variable: `wonly_s + wonly_s` is
    # linear and its value is correctly not needed.
    expect("snap_value_needed_vars descends", :wonly_s,
           STADE.agen_value_needed_vars(kernel))

    # snap_: the carried integer gets its own site. This replaces a check
    # that asserted the whole body was marked value_needed, which was the
    # design of a failed first attempt: snap_check_assign! gates a site on
    # activity, and a loop counter is an inactive integer, so marking it
    # produced no site at all. A :whilecarry site with its own Int64 stack
    # is what actually carries the value across the two sweeps.
    checks += 1
    let sites = STADE.snap_plan(kernel, STADE.act_analyze(kernel)),
        carry = [s.array for s in sites if s.kind === :whilecarry]
        if :i_k in carry
            println(rpad("snap_ emits a :whilecarry site for i_k", 46), " ok")
        else
            bad += 1
            println(rpad("snap_ emits a :whilecarry site for i_k", 46), " FAIL  ", carry)
        end
    end

    # --- lin_: the linearized body must exist ---
    checks += 1
    lin = STADE.lin_build(kernel, STADE.act_analyze(kernel))   # returns a body
    lw = lin[2]
    if lw.kind === :while && length(lw.body) == 3
        println(rpad("lin_ carries the while body", 46), " ok")
    else
        bad += 1
        println(rpad("lin_ carries the while body", 46), " FAIL  ", lw.kind)
    end

    # --- cgen_ / bgen_ / val_ collectors, run on the primal body ---
    let out = Set{Symbol}()
        STADE.bgen_collect_written!(kernel.body, out)
        expect("bgen_collect_written! descends", :wonly_s, out)
    end
    let out = Set{Symbol}()
        STADE.cgen_collect_all_assigned!(kernel.body, out)
        expect("cgen_collect_all_assigned! descends", :wonly_s, out)
    end

    println("\n", checks - bad, "/", checks, " checks passed",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end

validate_while_walkers()
