# ==================== validate_jacc_dead_pops.jl ================================
# Checks that the JACC path applies the dead-restore / dead-stack-store pass (`jacc_dead_pops` of stade_jacc, v0.4.73), as the CUDA path always did.
#   julia validate_jacc_dead_pops.jl        units, file driver, tile fill, sabotage
#
# The pass itself (cgen_dead_pops, an Expr to Expr transform) is checked by validate_dead_pops.jl; its runs on a device are checked by validate_corpus_gpu.jl. Here:
#   - units: for each case of validate_dead_pops.jl, stade_jacc with the initstacks function of the adjoint (`init_hosts`) leaves the stack stores that the pass leaves
#     (none for the cases that must lose all), the same plan without `init_hosts` or with `jacc_dead_pops = false` or `dead_scalars = false` is the text of v0.4.72;
#   - a taped forward sweep: the partial-sum stores of a contraction with a relu go (whether the twin then tiles it is checked on the unet2 text and on the V100, see the report);
#   - file driver: stade_jacc_file reads the `initstacks_*` function of the bundle; with two functions that share one it does nothing (as stade_gpu_file);
#   - tile fill (`jacc_tile_fill`, v0.4.73): a forward product whose shorter side is below the fill runs the plain kernel, one at or above it runs the twin, fill = 0 is the
#     v0.4.72 gate (the twin runs on both), and all three agree on the outputs;
#   - SABOTAGE edits of the new code, each of which one check must catch.

isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
isdefined(Main, :DP_UNIT) || include(joinpath(@__DIR__, "validate_dead_pops.jl"))
isdefined(Main, :jcn_run) || include(joinpath(@__DIR__, "validate_jacc_contraction.jl"))

jdp_src(e) = STADE.io_expr_to_source(e)
jdp_stores(plan, stacks) = sum(dp_count(k, stacks)[1] for k in vcat(plan.kernels, [plan.host]); init = 0)
jdp_text(plan) = join(vcat(jdp_src.(plan.helpers), jdp_src.(plan.kernels), [jdp_src(plan.host)]), "\n")

# a taped forward sweep of a contraction: a bias, a double sum over (c, r), then a relu of the sum (the adjoint needs the sign of the sum only, so the partial sums it tapes are never read)
const JDP_CONV = """
function conv(x, w, b, y, m, n, k)
    for idx = 1:m * n
        o = mod(idx - 1, n) + 1
        t = div(idx - 1, n) + 1
        s = b[o]
        for c = 1:k
            for r = 1:3
                s = s + x[(t - 1) * k * 3 + (c - 1) * 3 + r] * w[((c - 1) * 3 + r - 1) * n + o]
            end
        end
        y[idx] = max(s, 0.0)
    end
    return nothing
end"""

function jdp_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for (label, mklen, want, leaves, src) in DP_UNIT
        try
            adj = STADE.stade_adjoint(Meta.parse(src); keep_push_pop = false, fuse_ii_loops = true)
            ih = Dict{Symbol,Expr}(adj.initstacks.args[1].args[1] => adj.initstacks)
            stacks = Set(dp_stack_names(adj.initstacks))
            kw = (; inbounds = false)
            on = STADE.stade_jacc(adj.adjoint; kw..., init_hosts = ih)
            off = STADE.stade_jacc(adj.adjoint; kw..., init_hosts = ih, jacc_dead_pops = false)
            none = STADE.stade_jacc(adj.adjoint; kw...)
            nods = STADE.stade_jacc(adj.adjoint; kw..., init_hosts = ih, dead_scalars = false)
            nods0 = STADE.stade_jacc(adj.adjoint; kw..., dead_scalars = false)
            son, soff = jdp_stores(on, stacks), jdp_stores(off, stacks)
            ok_stores = want === :none ? son == 0 && soff > 0 : want === :some ? (0 < son <= soff) : son <= soff
            check(ok_stores, "$(label): stack stores $(soff) -> $(son) (want $(want))")
            check(jdp_text(off) == jdp_text(none), "$(label): jacc_dead_pops = false, and no init_hosts, give the v0.4.72 text")
            check(jdp_text(nods) == jdp_text(nods0), "$(label): dead_scalars = false keeps the pass off")
            leaves && check(!occursin("allowscalar", jdp_src(on.host)), "$(label): the forward loop is a device kernel")
        catch e
            check(false, "$(label): error $(first(split(sprint(showerror, e), "\n")))")
        end
    end
    # the taped convolution: the partial-sum stores go
    try
        adj = STADE.stade_adjoint(Meta.parse(JDP_CONV); keep_push_pop = false, fuse_ii_loops = true)
        ih = Dict{Symbol,Expr}(adj.initstacks.args[1].args[1] => adj.initstacks)
        stacks = Set(dp_stack_names(adj.initstacks))
        kw = (; inbounds = false, jacc_contraction = 1)
        on = STADE.stade_jacc(adj.adjoint; kw..., init_hosts = ih)
        off = STADE.stade_jacc(adj.adjoint; kw..., init_hosts = ih, jacc_dead_pops = false)
        son, soff = jdp_stores(on, stacks), jdp_stores(off, stacks)
        check(soff > 0 && son < soff, "taped contraction: stack stores $(soff) -> $(son)")
    catch e
        check(false, "taped contraction: error $(first(split(sprint(showerror, e), "\n")))")
    end
    return bad
end

function jdp_file_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    try
        adj = STADE.stade_adjoint(Meta.parse(DP_UNIT[1][5]); keep_push_pop = false, fuse_ii_loops = true)
        stacks = Set(dp_stack_names(adj.initstacks))
        second = Expr(:function, Expr(:call, :second_b, adj.adjoint.args[1].args[2:end]...), adj.adjoint.args[2])
        mktempdir() do tmp
            one, two = joinpath(tmp, "one.jl"), joinpath(tmp, "two.jl")
            write(one, join(jdp_src.([adj.initstacks, adj.adjoint]), "\n\n"))
            write(two, join(jdp_src.([adj.initstacks, adj.adjoint, second]), "\n\n"))
            count_in(path) = count(m -> true, eachmatch(r"_stack(_d)?\[[^\]]*\] = ", read(path, String)))
            STADE.stade_jacc_file(one, joinpath(tmp, "one_on.jl"))
            STADE.stade_jacc_file(one, joinpath(tmp, "one_off.jl"); jacc_dead_pops = false)
            STADE.stade_jacc_file(two, joinpath(tmp, "two_on.jl"))
            STADE.stade_jacc_file(two, joinpath(tmp, "two_off.jl"); jacc_dead_pops = false)
            non, noff = count_in(joinpath(tmp, "one_on.jl")), count_in(joinpath(tmp, "one_off.jl"))
            check(noff > 0 && non < noff, "file driver: stack stores $(noff) -> $(non) with the initstacks function in the bundle")
            check(read(joinpath(tmp, "two_on.jl"), String) == read(joinpath(tmp, "two_off.jl"), String), "file driver: two functions that take the stacks, the text is the one without the pass")
        end
    catch e
        check(false, "file driver: error $(first(split(sprint(showerror, e), "\n")))")
    end
    return bad
end

function jdp_fill_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    name, src, entry, mk, _ = CON_CASES[1]          # "bias product": x (m, k), w (k, n), b (n), y (m, n)
    function args_for(m, n, k, sd)
        r = MersenneTwister(sd)
        Any[randn(r, m * k), randn(r, k * n), randn(r, n), zeros(m * n), m, n, k]
    end
    ran(r) = any(nm -> endswith(nm, "_t!"), r.launched)
    # cmin = 64 so that the tile counts of these small products are enough; the fill is min(cmin, fill) = 64
    for (label, m, n, k, fill, want_twin) in (("shorter side 40 < 64, fill 64", 40, 1500, 24, 64, false),
                                              ("shorter side 40, fill 0 (v0.4.72)", 40, 1500, 24, 0, true),
                                              ("shorter side 200 >= 64, fill 64", 200, 300, 24, 64, true),
                                              ("40 rows of the other side, fill 64", 1500, 40, 24, 64, false))
        try
            args = args_for(m, n, k, 1)
            a = jcn_run(src, entry, args, 64; fill = fill)
            b = jcn_run(src, entry, args, 0)
            err = jcn_relerr(a, b)
            check(ran(a) == want_twin && err <= 1e-12 && jcn_changed(a, args) && jcn_tiles(a.plan) >= 1,
                  "tile fill, $(label): twin ran $(ran(a)) (want $(want_twin)), twin exists $(jcn_tiles(a.plan) >= 1), error $(round(err, sigdigits = 3))")
        catch e
            check(false, "tile fill, $(label): error $(first(split(sprint(showerror, e), "\n")))")
        end
    end
    return bad
end

function validate_jacc_dead_pops(; phases = (:units, :file, :fill, :sabotage), slice = 1:typemax(Int))
    bad = 0
    if :units in phases
        println("-- units")
        bad += jdp_unit_checks()
    end
    if :file in phases
        println("-- file driver")
        bad += jdp_file_checks()
    end
    if :fill in phases
        println("-- tile fill")
        bad += jdp_fill_checks()
    end
    if :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        text = read(joinpath(abspath(joinpath(@__DIR__, "..")), "src", "STADE.jl"), String)
        cases = [
            ("the pass is not applied in stade_jacc", "    dead_scalars && jacc_dead_pops && (expr = cgen_dead_pops(expr, init_hosts))\n", ""),
            ("the switch is ignored", "dead_scalars && jacc_dead_pops && (expr = cgen_dead_pops(expr, init_hosts))", "dead_scalars && (expr = cgen_dead_pops(expr, init_hosts))"),
            ("dead_scalars = false does not stop it", "dead_scalars && jacc_dead_pops && (expr = cgen_dead_pops(expr, init_hosts))", "jacc_dead_pops && (expr = cgen_dead_pops(expr, init_hosts))"),
            ("the file driver passes no initstacks", "jacc_dead_pops, jacc_tile_fill, init_hosts = (nclaim > 1 ? nothing : inits))", "jacc_dead_pops, jacc_tile_fill)"),
            ("the fill gate is dropped", "    fill > 16 && (cond = Expr(:&&, cond, call(:>=, call(:min, d.Mh, d.Nh), min(cmin, fill))))\n", ""),
            ("the fill gate looks at one side only", "call(:>=, call(:min, d.Mh, d.Nh), min(cmin, fill))", "call(:>=, d.Nh, min(cmin, fill))"),
            ("the JACC forward twin ignores the fill", "t = cgen_contraction_emit(d; cmin, fill)", "t = cgen_contraction_emit(d; cmin)"),
            ("the file driver ignores a shared initstacks", "jacc_dead_pops, jacc_tile_fill, init_hosts = (nclaim > 1 ? nothing : inits))", "jacc_dead_pops, jacc_tile_fill, init_hosts = inits)"),
        ]
        mktempdir() do tmp
            for (label, old, new) in cases[intersect(eachindex(cases), slice)]
                n = length(findall(old, text))
                if n != 1
                    check(false, "sabotage \"$label\": anchor found $n times")
                    continue
                end
                get(ENV, "SAB_ANCHORS_ONLY", "") == "1" && (check(true, "anchor \"$label\": found once"); continue)
                path = joinpath(tmp, "STADE_sab.jl")
                write(path, replace(text, old => new; count = 1))
                script = """
                    include($(repr(path)))
                    cd($(repr(@__DIR__)))
                    include($(repr(joinpath(@__DIR__, "validate_jacc_dead_pops.jl"))))
                    n = jdp_unit_checks() + jdp_file_checks() + jdp_fill_checks()
                    println("RESULT ", n)
                """
                out = try read(Base.pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all JACC dead pop checks passed" : "*** $bad JACC dead pop check(s) failed ***")
    return bad
end
abspath(PROGRAM_FILE) == abspath(joinpath(@__DIR__, "validate_jacc_dead_pops.jl")) && exit(validate_jacc_dead_pops() == 0 ? 0 : 1)
