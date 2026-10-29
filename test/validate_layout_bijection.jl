include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_layout_bijection()

Check `agen_local_position` on loop nests with numeric bounds and a parallel mark on each frame, by evaluating the
generated expression at every iteration.

Three properties, each measured directly:

1. The positions of all iterations are exactly `1:multiplicity`, each once. A formula that repeats a position makes two
   iterations share one stack slot, and the gradient is then wrong only where the stack is read back. A formula that
   skips a position leaves the stack oversized, which no gradient check sees at all.
2. With the other indices fixed, consecutive iterations of the FIRST parallel frame (the GPU thread) get consecutive
   positions. A bijection alone does not prove it: row-major order is also a bijection.
3. With no parallel frame, the order is row-major: consecutive iterations of the innermost frame get consecutive positions.

The nests cover depth 1 to 4, steps 1, 2 and -1, a start other than 1, trip counts of 1, a sequential frame outside the
thread (a time or level loop), a thread frame in the middle, and several parallel frames (the first one is the thread).

    cd test && julia validate_layout_bijection.jl
"""
function validate_layout_bijection()
    # each frame: (lo, step, hi, par)
    cases = [
        [(1, 1, 7, true)],
        [(3, 1, 9, true)],
        [(1, 2, 9, true)],
        [(5, -1, 1, true)],
        [(1, 1, 7, false)],
        [(1, 1, 4, true), (1, 1, 3, true)],
        [(1, 1, 4, false), (1, 1, 3, true)],
        [(1, 1, 4, false), (1, 1, 3, false)],
        [(1, 1, 1, true), (1, 1, 5, true)],
        [(1, 1, 5, true), (1, 1, 1, true)],
        [(2, 1, 5, true), (0, 2, 8, true), (1, 1, 3, true)],
        [(2, 1, 5, false), (0, 2, 8, true), (1, 1, 3, false)],
        [(2, 1, 5, false), (0, 2, 8, false), (1, 1, 3, true)],
        [(1, 1, 3, true), (1, 1, 4, true), (1, 1, 2, true), (1, 1, 5, true)],
        [(1, 1, 3, false), (1, 1, 4, false), (1, 1, 2, true), (1, 1, 5, true)],
        [(1, 1, 3, true), (7, -1, 4, true), (1, 2, 6, false)],
        [(4, 1, 6, false), (1, 1, 1, true), (1, 1, 1, false), (2, 1, 3, true)],
    ]
    nfail = 0
    for (ci, c) in enumerate(cases)
        vars = [Symbol("v", i) for i in 1:length(c)]
        ctx = Any[(var = vars[i], lo = c[i][1], hi = c[i][3], step = c[i][2], par = c[i][4]) for i in eachindex(c)]
        pos = Core.eval(Main, Expr(:->, Expr(:tuple, vars...), STADE.agen_local_position(ctx)))
        mult = Core.eval(Main, Expr(:->, Expr(:tuple), STADE.agen_local_multiplicity(ctx)))
        m = Base.invokelatest(mult)
        ranges = [c[i][1]:c[i][2]:c[i][3] for i in eachindex(c)]
        seen = Int[]
        for t in Iterators.product(ranges...)
            push!(seen, Base.invokelatest(pos, t...))
        end
        bij = length(seen) == m && sort(seen) == collect(1:m)
        # property 2 or 3: the frame that must be fastest has consecutive positions
        s = findfirst(f -> f[4], c)
        fast = s === nothing ? length(c) : s
        fast_ok = true
        if length(ranges[fast]) > 1
            others = [ranges[i] for i in eachindex(c) if i != fast]
            for rest in Iterators.product(others...)
                ps = Int[]
                for o in ranges[fast]
                    args = Any[]
                    k = 0
                    for i in eachindex(c)
                        push!(args, i == fast ? o : rest[(k += 1)])
                    end
                    push!(ps, Base.invokelatest(pos, args...))
                end
                all(diff(ps) .== 1) || (fast_ok = false)
            end
        end
        if !(bij && fast_ok)
            nfail += 1
            println("FAIL case $ci $(c): bijection=$bij fastest_frame_consecutive=$fast_ok (multiplicity $m, $(length(seen)) iterations)")
        end
    end
    println(length(cases), " nests checked   ", nfail == 0 ? "all layout checks passed" : "$nfail FAILED")
    return nfail
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_layout_bijection() == 0 ? 0 : 1)
end
