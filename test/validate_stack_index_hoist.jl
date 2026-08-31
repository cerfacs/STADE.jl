include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_stack_index_hoist(dir="val-corpus")

Check that every `keep_push_pop = false` stack access subscripts the stack with a
name, never with a computed expression.

`agen_site_index` reads each site's index into its own `__idx_*` scalar first, so
a generated adjoint or HVP contains `du_stack[__idx_du_stack_0]`, never
`du_stack[((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1]`. Two separate
things depend on that.

A Tier B index embeds a `prefix_*`/`val_*` table lookup. Inline, that is the
`a[b[i]]` indirect indexing `parse_check_no_indirect_indexing` rejects at parse
time -- STADE's own front end refusing its own output, which matters because an
adjoint is a legal input to `stade_hvp` and `stade_gpu`. This is the rule-8
compliance the hoist was introduced for, and it is a hard error, not a
preference.

A Tier A index carries no lookup and so breaks nothing when left inline, but it
is still a multi-term position formula, and a generated file is meant to be read
and audited. Naming it once is what a hand-written skill-stade kernel would do,
and it keeps one form for both tiers instead of a reader having to know which
tier a site landed in to know what to expect.

The oracles in `validate_corpus.jl` cannot see any of this: an inline index and
a hoisted one compute the same number, so gradients agree either way. The
property has to be measured on the generated source directly, which is what this
does.

Both storage modes' outputs are walked, but only `keep_push_pop = false`
generates indexed accesses at all -- `true` emits `push!`/`pop!`, which have no
subscript to check and are skipped by construction.

    cd test && julia validate_stack_index_hoist.jl
"""
function validate_stack_index_hoist(dir::String = joinpath(@__DIR__, "val-corpus"))
    kernels = sort(filter(f -> endswith(f, ".jl") &&
                               !(endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl")),
                          readdir(dir)))
    bad = 0
    checks = 0
    for f in kernels
        name = splitext(f)[1]
        primal = STADE.io_read_corpus_entry(joinpath(dir, f))
        for mode in (:adjoint, :hvp)
            gen = mode == :hvp ? STADE.stade_hvp(primal; keep_push_pop = false, fuse_ii_loops = true) :
                                 STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true)
            expr = mode == :hvp ? gen.hvp : gen.adjoint
            offenders = Any[]
            sites = Int[0]
            collect_offenders!(expr, offenders, sites)
            checks += 1
            if isempty(offenders)
                println(rpad("$(name) [$(mode)]", 34), " ok  ", lpad(sites[1], 4), " indexed stack accesses")
            else
                bad += 1
                println(rpad("$(name) [$(mode)]", 34), " FAIL  ", length(offenders),
                        " computed subscript(s), first: ", first(offenders))
            end
        end
    end
    println("\n", checks - bad, "/", checks, " (kernel, mode) pairs subscript every stack with a name",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end

# A stack argument is named `<var>_stack`, or `<var>_stack_d` for an HVP's shadow. The Tier B
# `prefix_<var>_stack_<block>` and `val_<var>_<block>` tables end in a block number instead, so
# they are not matched here -- they are read INSIDE a hoisted index expression, which is exactly
# the nesting the hoist exists to pull out of the subscript.
is_stack_name(s) = s isa Symbol && (endswith(string(s), "_stack") || endswith(string(s), "_stack_d"))

# Counts every indexed stack access and records those whose subscript is neither a bare name nor a
# literal. A literal is allowed: a non-loop site's index folds to the constant 1, and naming that
# would add a line saying nothing.
function collect_offenders!(e, offenders, sites)
    if e isa Expr
        if e.head == :ref && !isempty(e.args) && is_stack_name(e.args[1])
            sites[1] += 1
            for idx in e.args[2:end]
                (idx isa Symbol || idx isa Number) || push!(offenders, e)
            end
        end
        for a in e.args
            collect_offenders!(a, offenders, sites)
        end
    end
    return nothing
end

validate_stack_index_hoist()
