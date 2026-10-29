include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_layout_bijection()

Check `agen_local_position` on loop nests with numeric bounds, by evaluating the generated
expression at every iteration.

Two properties, each measured directly:

1. The positions of all iterations are exactly `1:multiplicity`, each once. A formula that
   repeats a position makes two iterations share one stack slot, and the gradient is then
   wrong only where the stack is read back. A formula that skips a position leaves the stack
   oversized, which no gradient check sees at all.
2. With the inner indices fixed, consecutive iterations of the OUTERMOST loop get consecutive
   positions. This is the element-major order that gives a GPU kernel (one thread per outermost
   iteration) coalesced stack access. A bijection alone does not prove it: the old row-major
   order is also a bijection.

The nests cover depth 1 to 4, steps 1, 2 and -1, a start other than 1, and trip counts of 1.

    cd test && julia validate_layout_bijection.jl
"""
function validate_layout_bijection()
    # each frame: (lo, step, hi)
    cases = [
        [(1, 1, 7)],
        [(3, 1, 9)],
        [(1, 2, 9)],
        [(5, 1, 1)],
        [(1, 1, 4), (1, 1, 3)],
        [(1, 1, 1), (1, 1, 5)],
        [(1, 1, 5), (1, 1, 1)],
        [(2, 1, 5), (0, 2, 8), (1, 1, 3)],
        [(1, 1, 3), (1, 1, 4), (1, 1, 2), (1, 1, 5)],
        [(1, 1, 3), (7, -1, 4), (1, 2, 6)],
        [(4, 1, 6), (1, 1, 1), (1, 1, 1), (2, 1, 3)],
    ]
    nfail = 0
    for (ci, c) in enumerate(cases)
        vars = [Symbol("v", i) for i in 1:length(c)]
        ctx = Any[(var = vars[i], lo = c[i][1], hi = c[i][3], step = c[i][2]) for i in eachindex(c)]
        pos = Core.eval(Main, Expr(:->, Expr(:tuple, vars...), STADE.agen_local_position(ctx)))
        mult = Core.eval(Main, Expr(:->, Expr(:tuple), STADE.agen_local_multiplicity(ctx)))
        m = Base.invokelatest(mult)
        ranges = [c[i][1]:c[i][2]:c[i][3] for i in eachindex(c)]
        seen = Int[]
        outer_ok = true
        for t in Iterators.product(ranges...)
            push!(seen, Base.invokelatest(pos, t...))
        end
        sorted = sort(seen)
        bij = length(seen) == m && sorted == collect(1:m)
        # property 2: outermost neighbours differ by exactly one slot
        if length(ranges[1]) > 1
            for rest in Iterators.product(ranges[2:end]...)
                ps = [Base.invokelatest(pos, o, rest...) for o in ranges[1]]
                all(diff(ps) .== 1) || (outer_ok = false)
            end
        end
        if !(bij && outer_ok)
            nfail += 1
            println("FAIL case $ci $(c): bijection=$bij outer_consecutive=$outer_ok (multiplicity $m, $(length(seen)) iterations)")
        end
    end
    println(length(cases), " nests checked   ", nfail == 0 ? "all layout checks passed" : "$nfail FAILED")
    return nfail
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_layout_bijection() == 0 ? 0 : 1)
end
