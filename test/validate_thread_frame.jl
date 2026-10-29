include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_thread_frame()

Check `agen_frame_parallel`, the layout stage's guess of which loop the GPU stage turns into the thread, on small kernels
where the right answer follows from the math. The layout puts the first parallel frame in the fastest position. A wrong
"parallel" for a time loop gives its stack a stride between adjacent threads (2.6 to 47 times slower stores on a V100). A
wrong "sequential" for the thread loop loses the coalescing. Neither changes a result, so only this check sees them.

Each case lists the expected verdict of every loop, outermost first, in source order.

    cd test && julia validate_thread_frame.jl
"""
function validate_thread_frame()
    cases = [
        ("time loop with two thread loops (ping-pong)", :(function pp(u, v, n, nsteps)
            for t = 1:nsteps
                for i = 1:n
                    v[i] = 0.5 * u[i]
                end
                for i = 1:n
                    u[i] = v[i] + 1.0
                end
            end
            return nothing
        end), [:t => false, :i => true, :i => true]),
        ("stencil time loop (ping-pong with neighbours)", :(function st(u, v, n, nsteps)
            for t = 1:nsteps
                for i = 2:n - 1
                    v[i] = u[i] + 0.1 * (u[i - 1] - 2.0 * u[i] + u[i + 1])
                end
                for i = 2:n - 1
                    u[i] = v[i]
                end
            end
            return nothing
        end), [:t => false, :i => true, :i => true]),
        # iterations of an accumulate-only loop commute, so the GPU stage makes the OUTER loop the kernel (atomic adds)
        ("accumulate-only outer loop", :(function lv(u, f, n, nl)
            for l = 1:nl
                for i = 1:n
                    u[i] = u[i] + f[i]
                end
            end
            return nothing
        end), [:l => true, :i => false]),
        ("in-place Gauss-Seidel sweep", :(function gs(u, n)
            for i = 2:n
                u[i] = 0.5 * (u[i - 1] + u[i])
            end
            return nothing
        end), [:i => false]),
        ("scalar recurrence", :(function rc(x, y, n)
            s = 0.0
            for i = 1:n
                s = s * 0.9 + x[i]
                y[i] = s
            end
            return nothing
        end), [:i => false]),
        ("element-wise", :(function ew(x, y, a, n)
            for i = 1:n
                y[i] = a * x[i]
            end
            return nothing
        end), [:i => true]),
        ("dot product", :(function dp(x, y, s, n)
            for i = 1:n
                s[1] = s[1] + x[i] * y[i]
            end
            return nothing
        end), [:i => true]),
        ("scatter accumulate", :(function sc(x, r, idx, n)
            for e = 1:n
                j = idx[e]
                r[j] = r[j] + x[e]
            end
            return nothing
        end), [:e => true]),
        ("dense layer, one thread per output", :(function de(x, w, y, n, m)
            for o = 1:m
                s = 0.0
                for k = 1:n
                    s = s + w[(o - 1) * n + k] * x[k]
                end
                y[o] = s
            end
            return nothing
        end), [:o => true, :k => false]),   # k runs inside the thread, so the GPU stage never decides it
    ]
    nfail = 0
    for (name, fexpr, expected) in cases
        k = STADE.parse_kernel(fexpr)
        par = STADE.agen_loop_par_map(k)
        got = Pair{Symbol,Bool}[]
        function walk(body)
            for st in body
                if st.kind == :for
                    push!(got, st.var => par[objectid(st.body)])
                    walk(st.body)
                elseif st.kind == :if
                    walk(st.then); walk(st.els)
                end
            end
        end
        walk(k.body)
        if get(ENV, "SHOW", "") == "1"; println(name, " => ", got); end
        if got != expected
            nfail += 1
            println("FAIL $name: expected $expected, got $got")
        end
    end
    println(length(cases), " kernels checked   ", nfail == 0 ? "all thread-frame checks passed" : "$nfail FAILED")
    return nfail
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_thread_frame() == 0 ? 0 : 1)
end
