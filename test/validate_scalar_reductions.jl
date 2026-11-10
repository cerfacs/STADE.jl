# Regression for the scalar reductions of v0.5.6 (defect B5 of the offload battle test).
#   - a loop that sums into local scalars, through temporaries and with several accumulators, becomes a device reduction above
#     `reduction_threshold` and stays a host loop below it, on CUDA (mapreduce) and on JACC (@parallel_reduce + Array(..)[1] readback)
#   - a loop that does not fit (an accumulator read by a term, a temporary read after the loop, a temporary used before it is set,
#     a temporary that reads an accumulator) is not turned into a reduction
#   - the generated CUDA host function runs on CPU arrays (its reduction branch is plain Julia) and agrees with the source kernel
# Reads generated code. No device is used. The device check is a separate job.
include(joinpath(@__DIR__, "validate_battle_status.jl"))

const SR_SRC = Dict(
"two_acc" => """
function two_acc(x, y, out, n)
    s1 = 0.0
    s2 = 0.0
    for i1 = 1:n
        t1 = x[i1] * y[i1]
        s1 = s1 + t1
        s2 = s2 + t1 * t1
    end
    out[1] = s1
    out[2] = s2
    return nothing
end""",
"minus" => """
function minus(x, y, out, n)
    s1 = 0.0
    for i1 = 1:n
        t1 = x[i1] * y[i1]
        s1 = s1 - t1
    end
    out[1] = s1
    return nothing
end""",
"prefix" => """
function prefix(x, out, n)
    s1 = 0.0
    s2 = 0.0
    for i1 = 1:n
        s1 = s1 + x[i1]
        s2 = s2 + s1 * x[i1]
    end
    out[1] = s2
    return nothing
end""",
"temp_live" => """
function temp_live(x, out, n)
    s1 = 0.0
    t1 = 0.0
    for i1 = 1:n
        t1 = x[i1]
        s1 = s1 + t1
    end
    out[1] = s1
    out[2] = t1
    return nothing
end""",
"temp_reads_acc" => """
function temp_reads_acc(x, out, n)
    s1 = 0.0
    for i1 = 1:n
        t1 = s1 + x[i1]
        s1 = s1 + t1
    end
    out[1] = s1
    return nothing
end""",
"temp_after" => """
function temp_after(x, out, n)
    s1 = 0.0
    t1 = 0.0
    for i1 = 1:n
        s1 = s1 + t1
        t1 = x[i1]
    end
    out[1] = s1
    return nothing
end""")
# the kernels that must become a reduction, with the number of reductions each holds
const SR_REDUCE = Dict("two_acc" => 2, "minus" => 1)
const SR_REFUSE = ["prefix", "temp_live", "temp_reads_acc", "temp_after"]

sr_count(txt, pat) = length(collect(eachmatch(Regex(pat), txt)))

# run the generated CUDA host function on CPU arrays: `CUDA.@allowscalar` is the identity, no kernel is launched
function sr_cpu_cuda(plan, name)
    mod = Module(:SRCPU)
    Core.eval(mod, :(module CUDA; macro allowscalar(e); esc(e); end; end))
    Core.eval(mod, :(using LinearAlgebra))
    Core.eval(mod, plan.host)
    return getfield(mod, Symbol(name * "_cuda"))
end

function validate_scalar_reductions(; verbose = true)
    bad = 0; checks = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("NOT OK ", msg)))
    for (name, src) in sort(collect(SR_SRC); by = first)
        f = Meta.parse(src)
        for (be, pat) in ((:cuda, "mapreduce|sum\\("), (:jacc, "@parallel_reduce"))
            plan = bt_plan(f, be, :pinned)
            txt = string(plan.host)
            n = sr_count(txt, pat)
            if haskey(SR_REDUCE, name)
                say(n == SR_REDUCE[name], "$name $be: $n reductions, expected $(SR_REDUCE[name])")
                if be === :jacc
                    say(sr_count(txt, "Array\\(") >= SR_REDUCE[name], "$name jacc: no Array readback of the reduction")
                    say(occursin("> 0", txt), "$name jacc: no zero-range guard")
                end
            else
                say(n == 0, "$name $be: $n reductions, expected none")
            end
        end
    end
    # CPU run of the generated CUDA host function: above the threshold (mapreduce) and below it (host loop), and with no trip
    for name in sort(collect(keys(SR_REDUCE)))
        f = Meta.parse(SR_SRC[name])
        plan = bt_plan(f, :cuda, :pinned)
        host = sr_cpu_cuda(plan, name)
        orig = Core.eval(Module(:SRORIG), f)
        for n in (0, 7, 40000)
            x = [0.5 + 0.001 * i for i in 1:max(n, 1)]; y = [1.5 - 0.0002 * i for i in 1:max(n, 1)]
            o1 = zeros(2); o2 = zeros(2)
            args = name == "minus" ? (x, y) : (x, y)
            Base.invokelatest(orig, args..., o1, n)
            Base.invokelatest(host, args..., o2, n)
            say(isapprox(o1, o2; rtol = 1e-10, atol = 1e-12), "$name n=$n: generated $(o2) against source $(o1)")
        end
    end
    println(bad == 0 ? "all $checks scalar reduction checks passed" : "$bad of $checks scalar reduction checks NOT OK")
    return bad == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    validate_scalar_reductions() || exit(1)
end
