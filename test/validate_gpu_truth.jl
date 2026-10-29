include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_gpu_truth(dir="val-corpus"; only=default_kernels, sabotage=true)

Structural gate for the BigFloat truth in the GPU parity script. No GPU needed. `validate_corpus_gpu.jl` runs the script on a device.

The parity script runs the CPU generated code a second time in BigFloat on the same inputs, and passes when the worst GPU error against that truth, over all trials, is at most 20 times the worst CPU error over the same trials, or at most 1e-10 (never below the atomic-count `rtol`). Here the script is generated for each kernel and mode, its device side is replaced by a copy of the CPU side (names without `_cuda`, `deepcopy` for `CuArray`), and the result is run on the host:

  1. TRUTH: the BigFloat run succeeded in every trial (`truth_trials` equals the number of trials), so no kernel silently falls back to the GPU-vs-CPU comparison.
  2. IDENTITY: with a device identical to the host, the script passes and the GPU error equals the CPU error.
  3. SABOTAGE: with the device result changed by 1e-3 in one element, the script must fail. A pass here means the new criterion cannot see a wrong result.
"""
function validate_gpu_truth(dir::String = joinpath(@__DIR__, "val-corpus"); only::Union{Vector{String},Nothing} = nothing, sabotage::Bool = true)
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    names = only !== nothing ? only : ["dotprod", "so_tile", "idm_tile", "stencil_loss", "affine_loss", "cellscatter", "mpnn", "advection", "clamped_sumsq", "cross_nested", "geomrecur", "matvec_loss"]
    trials = 3
    function hostify(txt::String)
        lines = split(txt, "\n")
        init_name = match(r"stacks_cpu = _val_init_stacks_local\((\w+), Any\[", txt).captures[1]
        out = String[]; cut = false
        for l in lines
            if startswith(l, "@inline function __stade_wa_xor") || startswith(l, "function $(init_name)_cuda(")
                cut = true
            end
            cut && startswith(l, "function _val_init_stacks_local") && (cut = false)
            cut && continue
            (startswith(l, "import Pkg") || startswith(l, "haskey(Pkg") || startswith(l, "using CUDA") || startswith(l, "CUDA.allowscalar") || startswith(l, "using JSON3")) && continue
            if startswith(l, "println(JSON3.write(Dict(\"ok\"")
                push!(out, "println(\"RESULT \", __all_ok, \" \", truth_trials, \" \", tr_cpu_worst, \" \", tr_gpu_worst, \" \", elements_changed)")
                continue
            end
            push!(out, l)
        end
        return replace(join(out, "\n"), "_cuda" => "", "CuArray(" => "deepcopy(")
    end
    function run_script(txt::String)
        f = tempname() * ".jl"; write(f, txt)
        out = try read(pipeline(`$(Base.julia_cmd()) --startup-file=no $f`; stderr = devnull), String) catch; "" end
        m = match(r"RESULT (true|false) (\d+) (\S+) (\S+) (\d+)", out)
        m === nothing && return nothing
        return (ok = m.captures[1] == "true", trials = parse(Int, m.captures[2]), tr_cpu = parse(Float64, m.captures[3]), tr_gpu = parse(Float64, m.captures[4]), changed = parse(Int, m.captures[5]))
    end
    # VERDICT: one verdict for the whole run, with two ways to pass (see val_truth_verdict_lines). Each case is a list of per-trial
    # (CPU error, GPU error) pairs. `per_trial` is the rule this replaced, judging every trial against 20 times its own CPU error.
    rtol = 2.5e-14; factor = 20.0; ceiling = 1.0e-10
    function verdict(pairs)
        m = Module(:VerdictCase)
        Core.eval(m, :(__all_ok = true))
        Core.eval(m, :(truth_trials = $(length(pairs))))
        Core.eval(m, :(tr_cpu_worst = $(maximum([p[1] for p in pairs]; init = 0.0))))
        Core.eval(m, :(tr_gpu_worst = $(maximum([p[2] for p in pairs]; init = 0.0))))
        include_string(m, join(STADE.val_truth_verdict_lines(rtol, factor, ceiling), "\n"))
        return Core.eval(m, :__all_ok)
    end
    per_trial(pairs) = all(p -> p[2] <= max(rtol, factor * p[1]), pairs)
    pooled(pairs) = maximum(p[2] for p in pairs) <= max(rtol, factor * maximum(p[1] for p in pairs))
    # the transformer HVP draw that failed on a V100: one accurate CPU trial beside two ordinary ones
    lucky = [(1.0e-13, 3.6e-12), (7.7e-13, 3.0e-12), (6.0e-13, 2.0e-12)]
    check(!per_trial(lucky) && verdict(lucky), "verdict: a trial with a very accurate CPU no longer fails ordinary GPU noise")
    # a transformer adjoint draw measured on a V100: the CPU error stays near 9e-14 while one GPU trial reaches 6.3e-12, 70 times larger
    heavy = [(8.9e-14, 1.07e-12), (8.8e-14, 3.6e-13), (7.3e-14, 6.3e-12)]
    check(!pooled(heavy) && verdict(heavy), "verdict: a GPU error 70 times the CPU's passes below the ceiling")
    check(!verdict([(1.0e-12, 1.0e-8), (1.2e-12, 9.0e-9), (9.0e-13, 1.1e-8)]), "verdict: a GPU error of 1e-8 fails")
    check(!verdict([(1.0e-12, 1.0e-12), (1.0e-12, 1.0e-12), (1.0e-12, 1.0e-3)]), "verdict: one catastrophic trial among good ones fails")
    check(verdict([(1.0e-14, 9.9e-11)]) && !verdict([(1.0e-14, 1.1e-10)]), "verdict: the ceiling is sharp, 9.9e-11 passes and 1.1e-10 fails")
    check(verdict([(1.0e-10, 1.9e-9)]) && !verdict([(1.0e-10, 2.1e-9)]), "verdict: a kernel with a large CPU error keeps the relative bound beyond the ceiling")
    check(verdict(Tuple{Float64,Float64}[]), "verdict: no truth trial leaves the earlier fallback verdict alone")

    for name in names
        tmp = mktempdir(); path = joinpath(tmp, name * ".jl"); cp(joinpath(dir, name * ".jl"), path)
        for mode in (:adjoint, :hvp)
            out = joinpath(tmp, "run_$(mode).jl")
            txt = ""; r = nothing; failed = false
            # a draw in which the kernel writes nothing carries no signal; draw again (the baseline is redrawn per call once its file is removed)
            for attempt in 1:5
                rm(joinpath(tmp, name * ".gpu.yaml"); force = true)
                try
                    STADE.stade_validate_cuda_file(path, out; mode = mode, fuse_ii_loops = true, int_lo = 3, int_hi = 5, trials = trials)
                catch e
                    check(false, "$name [$mode]: script generation failed: $(first(sprint(showerror, e), 100))"); failed = true; break
                end
                txt = read(out, String)
                r = run_script(hostify(txt))
                (r === nothing || r.changed > 0) && break
            end
            failed && continue
            check(occursin("__big = try", txt) && occursin("setprecision(BigFloat", txt), "$name [$mode]: script has the BigFloat truth")
            if r === nothing
                check(false, "$name [$mode]: host run of the script failed"); continue
            end
            check(r.trials == trials, "$name [$mode]: BigFloat run succeeded in $(r.trials)/$trials trials")
            check(r.ok && r.changed > 0 && r.tr_gpu == r.tr_cpu, "$name [$mode]: identical device passes (cpu $(r.tr_cpu), device $(r.tr_gpu))")
            if sabotage
                m = match(r"\n  (\w+)_val_cpu = deepcopy\(__base_\1\); \1_sh_cpu", txt)
                if m === nothing
                    println("info  $name [$mode]: no float array argument, sabotage skipped")
                else
                    a = m.captures[1]
                    bad_txt = replace(hostify(txt), r"(  ret_gpu_t = .*\n)" => SubstitutionString("\\1  length($(a)_val_gpu) > 0 && ($(a)_val_gpu[1] += 1.0e-3)\n"))
                    rs = run_script(bad_txt)
                    check(rs !== nothing && !rs.ok, "$name [$mode]: sabotage \"one element off by 1e-3\" is caught")
                end
            end
        end
    end
    println("\n", bad == 0 ? "all GPU truth checks passed" : "*** $bad GPU truth check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_gpu_truth(; only = isempty(ARGS) ? nothing : ARGS) == 0 ? 0 : 1)
end
