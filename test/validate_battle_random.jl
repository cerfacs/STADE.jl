# Battle test, family F11: random nests.  Each subject has a pattern of I and S loops, depth 1 to 6, random options.
# The oracle is the pattern.  Seeds come from STADE.val_stable_seed, so every run draws the same subjects.
include(joinpath(@__DIR__, "validate_battle_status.jl"))
using Random

function bt_random_spec(rng)
    d = rand(rng, 1:6)
    pat = String([rand(rng) < 0.4 ? 'S' : 'I' for _ in 1:d])
    (pattern = pat, tri = d >= 2 && rand(rng) < 0.2, fuse = rand(rng) < 0.3,
     wrapif = rand(rng) < 0.2, tmp = rand(rng) < 0.2, sibling = rand(rng) < 0.2, pre = rand(rng) < 0.1)
end

function bt_random_run(count::Int, out::String; seed_name = "battle_random", modes = (:primal,), offset = 0, stride = 1,
                       backends = (:cuda, :jacc))
    rng = Random.Xoshiro(STADE.val_stable_seed(seed_name))
    specs = [bt_random_spec(rng) for _ in 1:count]
    open(out, "w") do io
        println(io, join(["subject", "mode", "fuse", "backend", "config", "loop", "expected", "observed", "verdict", "kernels", "allow"], "\t"))
        for (n, sp) in enumerate(specs)
            (n - 1) % stride == offset || continue
            name = "rnd_$(n)"
            src, classes = bt_nest_source(name, sp.pattern; tri = sp.tri, fuse = sp.fuse, wrapif = sp.wrapif, tmp = sp.tmp, sibling = sp.sibling, pre = sp.pre)
            fn = bt_parse_fn(src)
            exp = bt_expected(fn, classes)
            prim = fn
            for mode in modes
                expr = try bt_mode_expr(prim, mode; fuse = true) catch e
                    println(io, join([name, mode, true, "-", "-", "-", "-", first(sprint(showerror, e), 100), "gen_error", 0, 0], "\t")); continue end
                for be in backends
                    pl = try bt_plan(expr, be, :pinned) catch e
                        println(io, join([name, mode, true, be, "-", "-", "-", first(sprint(showerror, e), 100), "plan_error", 0, 0], "\t")); continue end
                    obs, allow = bt_observe(pl, collect(keys(classes)), Symbol[])
                    for v in sort(collect(keys(classes)))
                        println(io, join([name * "|" * sp.pattern * (sp.tri ? "t" : "") * (sp.fuse ? "f" : "") * (sp.wrapif ? "w" : "") * (sp.tmp ? "m" : "") * (sp.sibling ? "s" : "") * (sp.pre ? "p" : ""),
                                          mode, true, be, "pinned", v, exp[v],
                                          join(sort([string(k, "=", c) for (k, c) in obs[v]]), ","), bt_verdict(exp[v], obs[v], be), length(pl.kernels), allow], "\t"))
                    end
                end
            end
            flush(io)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    # usage: julia validate_battle_random.jl OUT COUNT [MODES] [OFFSET STRIDE]
    out = ARGS[1]; count = parse(Int, ARGS[2])
    modes = length(ARGS) >= 3 ? Tuple(Symbol.(split(ARGS[3], ","))) : (:primal,)
    off = length(ARGS) >= 5 ? parse(Int, ARGS[4]) : 0; st = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1
    bt_random_run(count, out; modes = modes, offset = off, stride = st)
    println("wrote ", out)
end
