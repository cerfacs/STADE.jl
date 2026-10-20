include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_selfrace(dir="val-corpus")

Regression gate for the single-site overlap rule of the GPU split gate (`cgen_write_self_overlaps`).

A thread-parallel loop whose body scatters through ONE additive write site, `xb[t + ci - 1] += ...` under an inner serial loop over `ci`, was
emitted as a plain read-modify-write: the pairwise overlap analysis compares different sites with each other and `cgen_expr_injective_ok`
trusts any index that is `t` plus something thread-independent, so thread `t` at tap `ci + 1` and thread `t + 1` at tap `ci` raced on a real
device. The ownership emulation caught it deterministically.

  1. selfrace_conv   the adjoint's `xb` accumulation must be an atomic, and the ownership emulation must find no multi-writer element
  2. selfrace_layout a channel-major layout, `xb[(ci - 1) * i_hw + t]`, is disjoint through a symbolic stride and must STAY a plain write
  3. SABOTAGE        with the single-site check removed, selfrace_conv's `xb` write turns plain again and the ownership emulation flags it
"""
function validate_selfrace(dir::String = joinpath(@__DIR__, "val-corpus"))
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    function cuda_text(name, source = joinpath(@__DIR__, "..", "src", "STADE.jl"))
        work = mktempdir()
        script = """
            include($(repr(source)))
            STADE.stade_adjoint_file($(repr(joinpath(dir, name * ".jl"))), $(repr(joinpath(work, name * "_b.jl"))); keep_push_pop = false, fuse_ii_loops = true)
            STADE.stade_cuda_file($(repr(joinpath(work, name * "_b.jl"))), $(repr(joinpath(work, name * "_b_cuda.jl"))); warp_atomics = false)
        """
        run(pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stdout = devnull, stderr = devnull))
        return read(joinpath(work, name * "_b_cuda.jl"), String)
    end

    conv = cuda_text("selfrace_conv")
    check(occursin("@atomic xb[", conv), "selfrace_conv: the xb accumulation is an atomic")
    layout = cuda_text("selfrace_layout")
    check(!occursin("@atomic xb[", layout), "selfrace_layout: the disjoint strided write stays plain")
    own = success(pipeline(`$(Base.julia_cmd()) --startup-file=no $(joinpath(@__DIR__, "validate_ownership.jl")) selfrace_conv selfrace_layout`; stdout = devnull, stderr = devnull))
    check(own, "ownership emulation: no multi-writer element in either kernel")

    src = read(joinpath(@__DIR__, "..", "src", "STADE.jl"), String)
    old = "arr in racy || any(site -> cgen_write_self_overlaps(site, thread_var, varying), sites) && push!(racy, arr)"
    if count(old, src) == 1
        tmp = tempname() * ".jl"
        write(tmp, replace(src, old => ""))
        sab = cuda_text("selfrace_conv", tmp)
        rm(tmp; force = true)
        check(!occursin("@atomic xb[", sab), "sabotage \"no single-site check\": selfrace_conv's xb write turns plain again")
    else
        check(false, "sabotage line not found exactly once")
    end
    println("\n", bad == 0 ? "all self-overlap checks passed" : "*** $bad self-overlap check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_selfrace() == 0 ? 0 : 1)
end
