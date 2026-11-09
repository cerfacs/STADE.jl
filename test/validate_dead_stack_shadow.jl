include(joinpath(@__DIR__, "sabotage_run.jl"))
# ==================== validate_dead_stack_shadow.jl =============================
# dead_stack = 2 moves the kept region of a stack to the start of the array. The HVP host allocates the shadow `x_d` of a stack with
# `length(x)`. When the shadow is still read and written (checked mode: inbounds = false, where the dead reads stay), the shadow keeps the
# old indices, so x must keep its size. A shadow of a compacted stack that a kernel still touches is an out-of-range write
# (found on conv_b* and unet2 by the ownership shim, 2026-11-06).
#   julia validate_dead_stack_shadow.jl        units, sabotage
isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

const DSS_SRC = """
function dss_conv(img, w, bs, out, n, cin, cout)
    for idx = 1:cout * n * n
        px = mod(idx - 1, n) + 1
        py = mod(div(idx - 1, n), n) + 1
        oc = div(idx - 1, n * n) + 1
        s = bs[oc]
        for ci = 1:cin
            for ky = 1:3
                for kx = 1:3
                    yy = min(max(py + ky - 2, 1), n)
                    xx = min(max(px + kx - 2, 1), n)
                    s += w[((oc - 1) * cin + ci - 1) * 9 + (ky - 1) * 3 + kx] * img[((ci - 1) * n + yy - 1) * n + xx]
                end
            end
        end
        out[idx] = max(s, 0.0)
    end
    return nothing
end
"""

function dss_plan(inbounds::Bool)
    gen = STADE.stade_hvp(Meta.parse(DSS_SRC); keep_push_pop = false, fuse_ii_loops = true)
    be = STADE.cgen_backend_cuda()
    kw = (inbounds = inbounds, dead_scalars = true)
    pi = STADE.stade_gpu(gen.initstacks, be; kw...)
    pg = STADE.stade_gpu(gen.hvp, be; init_hosts = Dict{Symbol,Expr}(pi.host.args[1].args[1] => pi.host), kw...)
    return (plan = pg, text = join(string.(pg.kernels), "\n"), host = string(pg.host))
end

function dss_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for ib in (true, false)
        r = dss_plan(ib)
        shadow_refs = count(occursin("_stack_d[", l) for l in split(r.text, "\n"))
        compacted = sort!(collect(keys(r.plan.compact_stacks)))
        # no compacted stack has a shadow that a kernel touches
        clash = [x for x in compacted if occursin(string(x, "_d["), r.text)]
        check(isempty(clash), "inbounds = $ib: compacted stacks $(compacted); a compacted stack with a live shadow: $clash")
        if ib
            check(!isempty(compacted) && shadow_refs == 0, "inbounds = true: the shadows are gone ($shadow_refs references) and the stack is compacted")
        else
            check(shadow_refs > 0, "inbounds = false: the shadows stay ($shadow_refs references), so the test measures the case")
            check(isempty(compacted), "inbounds = false: the stack with a live shadow keeps its size")
        end
    end
    return bad
end

function validate_dead_stack_shadow(; phases = (:units, :sabotage))
    bad = 0
    :units in phases && (println("-- units"); bad += dss_unit_checks())
    if :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        text = read(joinpath(abspath(joinpath(@__DIR__, "..")), "src", "STADE.jl"), String)
        cases = [("a stack with a live shadow is compacted",
                  "            (xd in cands && any(((kn, i),) -> cgen_dstack_uses(kdefs[kn].args[2], kdefs[kn].args[1].args[1+i]), get(bind, xd, Tuple{Symbol,Int}[]))) && continue\n", "")]
        mktempdir() do tmp
            for (label, old, new) in cases
                n = length(findall(old, text))
                if n != 1
                    check(false, "sabotage \"$label\": anchor found $n times")
                    continue
                end
                path = joinpath(tmp, "STADE_sab.jl")
                write(path, replace(text, old => new; count = 1))
                script = """
                    include($(repr(path)))
                    cd($(repr(@__DIR__)))
                    include($(repr(joinpath(@__DIR__, "validate_dead_stack_shadow.jl"))))
                    println("RESULT ", dss_unit_checks())
                """
                out = sab_read(script)
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all dead-stack shadow checks passed" : "*** $bad dead-stack shadow check(s) failed ***")
    return bad
end

abspath(PROGRAM_FILE) == abspath(joinpath(@__DIR__, "validate_dead_stack_shadow.jl")) && exit(validate_dead_stack_shadow() == 0 ? 0 : 1)
