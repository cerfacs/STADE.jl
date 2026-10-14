include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

# Child statement lists of one IR statement, in program order.
function orc_children(st)
    st.kind == :for   && return Vector{NamedTuple}[st.body]
    st.kind == :while && return Vector{NamedTuple}[st.body]
    st.kind == :if    && return Vector{NamedTuple}[st.then, st.els]
    return Vector{NamedTuple}[]
end

# Outermost loops of `body` for which `accept(loop)` holds. A loop that is accepted is not searched
# further (the GPU gate also takes the outermost independent loop as the kernel); a refused loop is
# searched for accepted loops inside it.
function orc_outermost(body::Vector{NamedTuple}, accept)
    found = NamedTuple[]
    for st in body
        if st.kind == :for && accept(st)
            push!(found, st)
        else
            for ch in orc_children(st)
                append!(found, orc_outermost(ch, accept))
            end
        end
    end
    return found
end

# Does the STRICT array rule accept every written array of this primal loop? The scalar half of the gate
# is not part of this question: the oracle is about arrays.
function orc_primal_independent(loop)
    s = STADE.cgen_loop_array_summary(loop.body, Dict{Symbol,Any}())
    for (arr, widxs) in s.writes
        ridxs = get(s.reads, arr, Any[])
        STADE.cgen_array_ok_strict(loop.body, arr, widxs, ridxs, loop.var, s.defs) || return false
    end
    return !isempty(s.writes)
end

# primal array => :read_only | :write_only | :read_write, restricted to the float arrays that have a shadow
function orc_roles(loop, shadowed::Set{Symbol})
    s = STADE.cgen_loop_array_summary(loop.body, Dict{Symbol,Any}())
    roles = Dict{Symbol,Symbol}()
    for a in union(keys(s.writes), keys(s.reads))
        (a isa Symbol && a in shadowed) || continue
        w = haskey(s.writes, a)
        r = haskey(s.reads, a)
        roles[a] = w && r ? :read_write : (w ? :write_only : :read_only)
    end
    return roles
end

orc_shadow(a::Symbol) = Symbol(string(a), "b")

# The class the shadow of a primal array of each role must have in the reverse loop. A read-write primal
# array has no class asserted: its shadow is consumed and accumulated at the same index by one iteration.
# A write-only primal array may also have a shadow that the reverse loop only READS: when nothing later reads the
# shadow, the generator drops the clearing write. A read-only shadow cannot race, so it is accepted. It may also
# be only OVERWRITTEN (a clearing write with no read, as for a padded scratch array that a loop zeroes first); the
# strict array rule accepts such an array with no idiom, since there is no read index to compare.
const_expected = Dict(:read_only => (:accumulate_only,), :write_only => (:consume_only, :read_only_shadow, :overwrite_only, :clear_mixed))

# Every write to `shadow` in `body` stores a numeric literal (a clearing write).
function orc_clear_writes_only(body::Vector{NamedTuple}, shadow::Symbol)
    for st in body
        if st.kind == :assign
            if st.lhs isa Expr && st.lhs.head == :ref && st.lhs.args[1] == shadow && !(st.rhs isa Number)
                return false
            end
        else
            for ch in orc_children(st)
                orc_clear_writes_only(ch, shadow) || return false
            end
        end
    end
    return true
end

function orc_shadow_class(body, shadow::Symbol)
    sm = STADE.cgen_loop_array_summary(body, Dict{Symbol,Any}())
    haskey(sm.writes, shadow) || return :read_only_shadow
    haskey(sm.reads, shadow) || return :overwrite_only
    STADE.cgen_array_accumulate_only(body, shadow) && return :accumulate_only
    STADE.cgen_array_consume_only(body, shadow) && return :consume_only
    # consumed on one branch and cleared on the other (relu_field): every write is a literal, and the strict rule
    # already accepts it because read and write index are the same
    orc_clear_writes_only(body, shadow) && return :clear_mixed
    return :neither
end

"""
    validate_idiom_oracle(dir="val-corpus"; only=nothing)

Independent check of the idiom classifiers the GPU split gate relies on. The classifiers
(`cgen_array_accumulate_only`, `cgen_array_consume_only`) look at the REVERSE loop. This oracle looks at
the PRIMAL loop instead, which STADE trusts and the classifiers never see, and states what the reverse
loop must then look like:

  * a primal array that the loop only reads has a shadow that the reverse loop only accumulates into
  * a primal array that the loop only writes has a shadow that the reverse loop only consumes (or only reads)

Two implementations that must agree: a primal-side role analysis here, and the reverse-side classifier
in the gate. A reverse loop whose shadow falls in neither class, or in the wrong one, is a mismatch.

Loops are paired by the set of shadow arrays they touch and by loop variable name (the generator keeps it). When that set does not pick exactly one reverse
loop for one primal loop, the loop is SKIPPED and reported with the reason; it never passes silently.
"""
function validate_idiom_oracle(dir::String = joinpath(@__DIR__, "val-corpus");
                               only::Union{Vector{String},Nothing} = nothing, verbose::Bool = true)
    names = sort([splitext(f)[1] for f in readdir(dir) if endswith(f, ".jl") &&
                  !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl", "_cuda.jl", "_jacc.jl"))])
    only !== nothing && (names = filter(n -> n in only, names))
    matched = 0; skipped = 0; mismatches = 0; errors = 0; asserted = 0
    skips = Dict{Symbol,Int}()
    for name in names
        try
            pe = STADE.io_read_corpus_entry(joinpath(dir, name * ".jl"))
            gp = STADE.cgen_ingest(pe)
            adj = STADE.stade_adjoint(pe; keep_push_pop = false).adjoint
            gb = STADE.cgen_ingest(adj)
            shadowed = Set{Symbol}(a for a in gp.args if orc_shadow(a) in gb.args)
            ploops = orc_outermost(gp.body, orc_primal_independent)
            # shadow-array set of a loop: which shadow arrays its body touches
            shset(loop, wrt) = begin
                s = STADE.cgen_loop_array_summary(loop.body, Dict{Symbol,Any}())
                Set{Symbol}(a for a in union(keys(s.writes), keys(s.reads)) if a isa Symbol && a in wrt)
            end
            shadow_set = Set{Symbol}(orc_shadow(a) for a in shadowed)
            for pl in ploops
                roles = orc_roles(pl, shadowed)
                isempty(roles) && continue
                want = Set{Symbol}(orc_shadow(a) for a in keys(roles))
                rloops = orc_outermost(gb.body, l -> l.var == pl.var && shset(l, shadow_set) == want)
                pset = [q for q in ploops if q.var == pl.var && Set{Symbol}(orc_shadow(a) for a in keys(orc_roles(q, shadowed))) == want]
                if length(rloops) != 1 || length(pset) != 1
                    skipped += 1
                    reason = length(rloops) == 0 ? :no_reverse_loop : :ambiguous
                    skips[reason] = get(skips, reason, 0) + 1
                    verbose && println(rpad(name, 24), " SKIP  primal loop ", pl.var, " ", reason,
                                       " (", length(pset), " primal / ", length(rloops), " reverse with the same shadow set)")
                    continue
                end
                rl = rloops[1]
                bad = String[]
                for (a, role) in sort(collect(roles); by = first)
                    haskey(const_expected, role) || continue
                    asserted += 1
                    got = orc_shadow_class(rl.body, orc_shadow(a))
                    got in const_expected[role] || push!(bad, "$(a): $(role) primal, shadow is $(got), expected $(join(const_expected[role], " or "))")
                end
                matched += 1
                if isempty(bad)
                    verbose && println(rpad(name, 24), " ok    ", pl.var, " -> ", rl.var, "  ",
                                       join(["$(a):$(r)" for (a, r) in sort(collect(roles); by = first)], " "))
                else
                    mismatches += 1
                    println(rpad(name, 24), " MISMATCH  ", pl.var, " -> ", rl.var)
                    for b in bad
                        println("    ", b)
                    end
                end
            end
        catch e
            errors += 1
            println(rpad(name, 24), " ERROR ", first(split(sprint(showerror, e), "\n")))
        end
    end
    println("\n", matched, " loop pairs matched (", asserted, " array assertions), ", skipped, " skipped ",
            collect(skips), ", ", mismatches, " mismatches, ", errors, " errors")
    return (matched = matched, skipped = skipped, mismatches = mismatches, errors = errors, asserted = asserted)
end

if abspath(PROGRAM_FILE) == @__FILE__
    r = validate_idiom_oracle(; only = length(ARGS) > 0 ? ARGS : nothing)
    exit(r.mismatches == 0 && r.errors == 0 && r.matched > 0 ? 0 : 1)
end
