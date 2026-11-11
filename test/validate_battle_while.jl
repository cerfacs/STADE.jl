# Battle test, family F12: random per-element `while` bodies (v0.5.8).
# Each subject is one independent loop `for i1 = 2:n` holding 1 to 3 pieces. A safe piece keeps its scalars private to the iteration and
# touches only its own elements, so the loop must be offloaded (K). A hazard piece carries a value between iterations or reads an element
# that another iteration writes, so the loop must stay on the host (H). The oracle is the piece list. Seeds come from STADE.val_stable_seed.
# A false offload (expected H, observed K) is the failure that matters here: it would give wrong numbers.
include(joinpath(@__DIR__, "validate_battle_status.jl"))
using Random

# (text of the piece for index k, text placed before the loop)
const BW_SAFE = [
    k -> ("    r$k = x[i1]\n    while r$k > tol\n        r$k = 0.5 * r$k\n    end\n    y[i1] = y[i1] + r$k\n", ""),
    k -> ("    r$k = x[i1]\n    c$k = 0.0\n    while r$k > tol\n        r$k = 0.5 * r$k\n        c$k = c$k + 1.0\n    end\n    z[i1] = z[i1] + c$k\n", ""),
    k -> ("    r$k = x[i1]\n    while r$k > tol\n        if r$k > 1.0\n            r$k = 0.25 * r$k\n        else\n            r$k = 0.5 * r$k\n        end\n    end\n    y[i1] = y[i1] + r$k\n", ""),
    k -> ("    r$k = 1.0\n    s$k = 0.0\n    while r$k > tol\n        for e$k = 1:m\n            s$k = s$k + x[i1] * r$k\n        end\n        r$k = 0.5 * r$k\n    end\n    z[i1] = z[i1] + s$k\n", ""),
    k -> ("    r$k = 2.0 * x[i1]\n    while r$k > tol + x[i1]\n        r$k = 0.5 * r$k\n    end\n    y[i1] = y[i1] + r$k\n", ""),
    k -> ("    j$k = idx[i1]\n    r$k = x[i1]\n    while r$k > tol\n        w[j$k] = w[j$k] + 1.0\n        r$k = 0.5 * r$k\n    end\n", ""),
]
const BW_HAZARD = [
    ("carried condition scalar", k -> ("    while g$k > tol\n        g$k = 0.5 * g$k\n    end\n    z[i1] = z[i1] + g$k\n", "    g$k = 1.0\n")),
    ("set only inside the while, read after", k -> ("    r$k = x[i1]\n    while r$k > tol\n        r$k = 0.5 * r$k\n        q$k = r$k\n    end\n    y[i1] = y[i1] + q$k\n", "    q$k = 0.0\n")),
    ("condition reads the next element", k -> ("    y[i1] = x[i1]\n    r$k = x[i1]\n    while r$k > y[i1 + 1]\n        r$k = 0.5 * r$k\n    end\n    z[i1] = z[i1] + r$k\n", "")),
    ("previous element read", k -> ("    r$k = x[i1]\n    while r$k > tol\n        r$k = 0.5 * r$k\n    end\n    z[i1] = r$k + z[i1 - 1]\n", "")),
    ("outer counter", k -> ("    r$k = x[i1]\n    while r$k > tol\n        r$k = 0.5 * r$k\n        t$k = t$k + 1.0\n    end\n    y[i1] = y[i1] + r$k\n", "    t$k = 0.0\n")),
    ("condition reads an undefined local", k -> ("    r$k = x[i1]\n    while u$k > tol\n        r$k = 0.5 * r$k\n        u$k = r$k\n    end\n    u$k = x[i1]\n    y[i1] = y[i1] + r$k\n", "    u$k = 1.0\n")),
    ("neighbour read in the body", k -> ("    r$k = x[i1]\n    while r$k > tol\n        r$k = 0.5 * r$k + 0.0 * y[i1 + 1]\n    end\n    y[i1] = r$k\n", "")),
]

function bw_spec(rng)
    np = rand(rng, 1:3)
    pieces = Tuple{Symbol,Int}[]
    haz = rand(rng) < 0.5 ? rand(rng, 1:length(BW_HAZARD)) : 0
    used_gather = false
    for k in 1:np
        i = rand(rng, 1:length(BW_SAFE))
        (i == 6 && used_gather) && (i = 1)
        i == 6 && (used_gather = true)
        push!(pieces, (:safe, i))
    end
    haz > 0 && insert!(pieces, rand(rng, 1:length(pieces)+1), (:hazard, haz))
    return pieces
end

function bw_source(name, pieces)
    pre = ""; body = ""
    for (k, (kind, i)) in enumerate(pieces)
        b, p = kind === :safe ? BW_SAFE[i](k) : BW_HAZARD[i][2](k)
        body *= b; pre *= p
    end
    return "function $name(x, y, z, w, idx, tol, n, m)\n$(pre)    for i1 = 2:n\n" * join(["    " * l for l in split(chomp(body), "\n")], "\n") * "\n    end\n    return nothing\nend\n"
end

function bw_run(count::Int, out::String; modes = (:primal, :tangent), backends = (:cuda, :jacc), seed_name = "battle_while")
    rng = Random.Xoshiro(STADE.val_stable_seed(seed_name))
    specs = [bw_spec(rng) for _ in 1:count]
    open(out, "w") do io
        println(io, join(["subject", "mode", "fuse", "backend", "config", "loop", "expected", "observed", "verdict", "kernels", "allow"], "\t"))
        for (n, sp) in enumerate(specs)
            name = "wr_$(n)"
            fn = bt_parse_fn(bw_source(name, sp))
            exp = any(p -> p[1] === :hazard, sp) ? :H : :K
            tag = join([p[1] === :safe ? "s$(p[2])" : "h$(p[2])" for p in sp], "")
            for mode in modes
                expr = try bt_mode_expr(fn, mode; fuse = true) catch e
                    println(io, join([name * "|" * tag, mode, true, "-", "-", "-", "-", first(sprint(showerror, e), 100), "gen_error", 0, 0], "\t")); continue end
                for be in backends
                    pl = try bt_plan(expr, be, :pinned) catch e
                        println(io, join([name * "|" * tag, mode, true, be, "-", "-", "-", first(sprint(showerror, e), 100), "plan_error", 0, 0], "\t")); continue end
                    obs, allow = bt_observe(pl, [:i1], Symbol[])
                    println(io, join([name * "|" * tag, mode, true, be, "pinned", "i1", exp, join(sort([string(k, "=", c) for (k, c) in obs[:i1]]), ","), bt_verdict(exp, obs[:i1], be), length(pl.kernels), allow], "\t"))
                end
            end
            flush(io)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    out = ARGS[1]; count = parse(Int, ARGS[2])
    bw_run(count, out)
    println("wrote ", out)
end
