include(joinpath(ARGS[1], "src", "STADE.jl"))

# Scoping probe, NOT a validator. Asks how many corpus kernels a rule-1 gather-aliasing check
# would flag, before any such check goes near parse_kernel.
#
# The shape that produced a silently wrong adjoint:
#
#     for i_r = 1:i_n
#         i_k = i_idx[i_r]          # i_k is a GATHER SCALAR: loaded from an array
#         y[i_r] = y[i_k] * x[i_r]  # writes y at one index, reads it at another
#     end
#
# STADE accepts this and emits an adjoint wrong by ~100%. Rule 1 forbids it -- the loop carries a
# value between iterations at a location nothing can order -- but nothing enforces that.
#
# The check must NOT flag two legal shapes that look similar:
#   * a sequential recurrence, `u[i] = c * u[i - 1]`. Also reads and writes one array at differing
#     indices, but the read index is affine in the loop variable, so the dependence direction is
#     known and STADE handles it. geomrecur and prefixscan depend on this.
#   * a scatter-accumulate, `v[j] = v[j] + f(i)` with `j` gathered. The read and write indices are
#     the SAME expression, so there is no cross-iteration ordering question. cellscatter and mpnn
#     depend on this.
#
# So the rule is narrow: same array, read index differs structurally from the write index, and the
# read index mentions a scalar loaded from an array inside the same loop.

# Scalars assigned from an array read anywhere in this loop body.
function gather_scalars_here(body)
    acc = Set{Symbol}()
    walk(e) = begin
        if e isa Expr
            if e.head == :(=) && e.args[1] isa Symbol && contains_ref(e.args[2])
                push!(acc, e.args[1])
            end
            e.head == :for && return
            foreach(walk, e.args)
        end
    end
    for st in body.args
        walk(st)
    end
    return acc
end

contains_ref(e) = e isa Expr && (e.head == :ref || any(contains_ref, e.args))

# Every (array, index-tuple) written, and every one read, AT THIS LOOP'S OWN STATEMENT LEVEL.
# Nested `for`s are not descended into, and that scoping is the whole difficulty. Scanning a loop's
# entire subtree conflates two different things: a read and a write in one loop body, which is the
# hazard, and a write in one child loop read by a sibling child loop, which is just a sequential
# outer loop doing its job. cellscatter's `i_` pass loop and ttgc's are the second kind -- eight
# false positives before this restriction, zero after.
function refs(body)
    writes = Tuple{Symbol,Any}[]
    reads = Tuple{Symbol,Any}[]
    walk(e, lhs) = begin
        e isa Expr || return
        if e.head == :(=)
            l = e.args[1]
            if l isa Expr && l.head == :ref && l.args[1] isa Symbol
                push!(writes, (l.args[1], Tuple(l.args[2:end])))
                for a in l.args[2:end]; walk(a, false); end
            end
            walk(e.args[2], false)
            return
        end
        if e.head == :ref && e.args[1] isa Symbol
            push!(reads, (e.args[1], Tuple(e.args[2:end])))
            for a in e.args[2:end]; walk(a, false); end
            return
        end
        e.head == :for && return          # a nested loop is a separate scope for this purpose
        foreach(a -> walk(a, false), e.args)
    end
    for st in body.args
        walk(st, false)
    end
    return (writes, reads)
end

mentions(e, names) = e isa Symbol ? e in names :
                     e isa Expr ? any(a -> mentions(a, names), e.args) : false

function check_loop(loopexpr, kernel_name, hits)
    body = loopexpr.args[2]
    gs = gather_scalars_here(body)
    isempty(gs) && return
    (writes, reads) = refs(body)
    for (wa, wi) in writes, (ra, ri) in reads
        wa === ra || continue
        wi == ri && continue                      # same index: a legal accumulation
        any(mentions(x, gs) for x in ri) || continue   # read index is not a gather
        push!(hits, (kernel_name, wa, wi, ri))
    end
    return
end

function scan(e, kernel_name, hits)
    if e isa Expr
        e.head == :for && check_loop(e, kernel_name, hits)
        foreach(a -> scan(a, kernel_name, hits), e.args)
    end
end

dir = joinpath(ARGS[1], "test", "val-corpus")
hits = Tuple{String,Symbol,Any,Any}[]
nscanned = Ref(0)
for f in sort(filter(f -> endswith(f, ".jl") &&
                          !(endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl")),
                     readdir(dir)))
    nscanned[] += 1
    scan(STADE.io_read_corpus_entry(joinpath(dir, f)), splitext(f)[1], hits)
end

# and the shape that motivated the check, which must be flagged
probe = Meta.parse("""
function probe(y, x, i_idx, i_n)
    for i_r = 1:i_n
        i_k = i_idx[i_r]
        y[i_r] = y[i_k] * x[i_r]
    end
    return nothing
end
""")
probe_hits = Tuple{String,Symbol,Any,Any}[]
scan(probe, "probe", probe_hits)

println("scanned ", nscanned[], " corpus kernels")
println("flagged: ", length(hits))
for h in hits
    println("  ", rpad(h[1], 22), " ", h[2], " written at ", h[3], ", read at ", h[4])
end
println("\nmotivating shape flagged: ", !isempty(probe_hits), " (", length(probe_hits), " site(s))")
