# Regression for v0.5.17: site-level TBR no longer snapshots the partial sums of a boxed reduction inside an outer loop.
#   Defect: a reduction into a length-1 array (`s[1] = 0.0; for i ...; s[1] += x; end; out = rs / s[1]`) inside an outer counted loop got one
#   snapshot per element, because the nonlinear read `rs / s[1]` of the previous iteration reaches the accumulation over the loop back edge. Each
#   snapshot is a stack store inside the element loop, so cgen_ left the loop on the host (one device access per element). The snapshots are
#   dead: the read sees only the final version, which the reset of the next iteration restores through its own snapshot.
#   Fix: snap_fwd_walk! / agen_fwd_walk! drop a boxed array from `seen` after an unconditional reset that does not read it
#   (snap_boxed_reset / agen_boxed_reset), and a loop keeps its entry set when its body may not run.
# Part 1 counts the per-site decisions of `s` (the shapes that must keep their snapshots keep them). Part 2 validates every kernel against finite differences
# and the tangent-adjoint dot-product identity, in both stack modes, with a zero-trip draw (int_lo = 0). Part 3 reads the CUDA output.
# No device is used. The device check is a V100 run of br_cg1d2.
include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))

# kernel => number of sites of `s` that snapshot (v0.5.16 value in the comment)
const BR_EXPECT = Dict(
    "br_a0" => 1,   # 2  reset, accumulate, nonlinear read after: the shape this release fixes
    "br_a1" => 2,   # 2  the reset is under an if: not unconditional, kept
    "br_a2" => 3,   # 3  two indices of one array: not boxed, kept
    "br_a3" => 2,   # 2  nonlinear read of s between the reset and the accumulation, kept
    "br_a4" => 2,   # 2  the reset reads s itself, kept
    "br_a5" => 1,   # 2  reset inside an inner loop of the outer loop
    "br_a6" => 1,   # 2  loop after a straight-line use
    "br_a7" => 2,   # 3  zero-trip shape (nonlinear read before the loop, overwrite after)
    "br_a8" => 2,   # 3  same, the loop body has no nonlinear read after the reset: the loop-entry set must survive a zero-trip loop
    "br_a9" => 2,   # 2  variable index s[j]: the array is not boxed, kept
    "br_pw" => 0, "br_gs" => 0, "br_sd" => 1, "br_n2" => 1, "br_wh" => 1,   # `s` (0: the kernel has no array named s; see BR_EXTRA)
    "br_m4" => 1,   # 2
    "br_cg1d2" => 1 # 2  (and 1 for t)
)

# solver-shaped kernels: expected snapshot sites per variable that is a boxed array (the reset's own site stays, the accumulation's goes)
const BR_EXTRA = Dict(
    "br_pw" => Dict(:s1 => 1, :s2 => 1),   # power iteration, three boxed scalars
    "br_gs" => Dict(:res => 1),            # Gauss-Seidel, residual norm
    "br_sd" => Dict(:s => 1, :t => 1),     # steepest descent, two boxed reductions in one body
    "br_n2" => Dict(:s => 1),              # reset in an inner loop, nonlinear read after it in the outer body
    "br_wh" => Dict(:s => 1),              # while loop
)
# a while is refused with keep_push_pop = false: validated with keep_push_pop = true only
const BR_KPP_TRUE_ONLY = ("br_wh",)

function br_decisions(path)
    ks = STADE.io_read_kernel_corpus(path)
    k = STADE.parse_kernel(STADE.inl_inline_calls(ks)[STADE.io_corpus_entry_name(path, ks)])
    d = STADE.snap_value_needed_sites(k); v = STADE.snap_site_vars(k)
    cnt = Dict{Symbol,Int}()
    for (key, need) in d
        need && (cnt[v[key]] = get(cnt, v[key], 0) + 1)
    end
    return cnt, STADE.agen_value_needed_sites(k) == d
end

function validate_boxed_reset(; verbose = true, only = String[])
    bad = 0; checks = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("NOT OK ", msg)))
    dir = joinpath(@__DIR__, "boxed-reset-corpus")
    tmp = mktempdir()
    names = sort(String[f[1:end-3] for f in readdir(dir) if endswith(f, ".jl")])
    isempty(only) || (names = filter(n -> n in only, names))
    for nm in names
        path = joinpath(dir, nm * ".jl")
        cnt, same = br_decisions(path)
        say(same, "$nm: snap_ and agen_ site decisions differ")
        say(get(cnt, :s, 0) == BR_EXPECT[nm], "$nm: sites of s that snapshot = $(get(cnt, :s, 0)), expected $(BR_EXPECT[nm])")
        nm == "br_cg1d2" && say(get(cnt, :t, 0) == 1, "$nm: sites of t that snapshot = $(get(cnt, :t, 0)), expected 1")
        for (v, e) in get(BR_EXTRA, nm, Dict{Symbol,Int}())
            say(get(cnt, v, 0) == e, "$nm: sites of $v that snapshot = $(get(cnt, v, 0)), expected $e")
        end
        for lo in (3, 0)
            yp = joinpath(tmp, "$nm.lo$lo.yaml")
            STADE.stade_generate_baseline_file(path; yaml_path = yp, int_lo = lo, int_hi = 5)
            for (kp, fz) in (nm in BR_KPP_TRUE_ONLY ? ((true, false),) : ((false, true), (false, false), (true, false)))
                r1 = STADE.stade_validate_adjoint_file(path; yaml_path = yp, keep_push_pop = kp, fuse_ii_loops = fz, int_lo = lo, int_hi = 5)
                say(r1.ok, "$nm adjoint keep_push_pop=$kp fuse=$fz int_lo=$lo")
                r2 = STADE.stade_validate_dotprod_file(path; yaml_path = yp, keep_push_pop = kp, fuse_ii_loops = fz)
                say(r2.ok, "$nm dot-product identity keep_push_pop=$kp fuse=$fz int_lo=$lo")
            end
            if lo == 3 && !(nm in BR_KPP_TRUE_ONLY)
                for fz in (true, false)
                    r3 = STADE.stade_validate_hvp_file(path; yaml_path = yp, keep_push_pop = false, fuse_ii_loops = fz, int_lo = lo, int_hi = 5)
                    say(r3.ok, "$nm hvp keep_push_pop=false fuse=$fz")
                end
            end
        end
    end
    # CUDA output: the fixed shapes have no element loop left on the host in the forward sweep of the adjoint
    # (v0.5.17 left 2, 1 and 1 element loops on the host in br_cg1d2, br_sd and br_pw: loops with a local scalar accumulator. v0.5.18 boxes those locals and offloads them.
    # br_gs keeps its 2: an order-dependent sweep. The counts are what remains, not a claim of minimum)
    for (nm, expect) in (("br_m4", 0), ("br_a0", 0), ("br_n2", 0), ("br_cg1d2", 0), ("br_sd", 0), ("br_pw", 0), ("br_gs", 2))
        nm in names || continue
        b = joinpath(tmp, nm * "_b.jl"); c = joinpath(tmp, nm * "_b_cuda.jl")
        STADE.stade_adjoint_file(joinpath(dir, nm * ".jl"), b; keep_push_pop = false, fuse_ii_loops = true)
        STADE.stade_cuda_file(b, c)
        t = read(c, String)
        t = t[first(findfirst("function $(nm)_b_cuda", t)):end]
        r = findfirst("\nfunction ", t[2:end]); r === nothing || (t = t[1:first(r)])
        n = length(collect(eachmatch(r"\n\s+for i = [^\n]*\n\s+CUDA\.@allowscalar", t)))
        say(n == expect, "$nm: $n host element loops in the CUDA adjoint, expected $expect")
    end
    verbose && println("validate_boxed_reset: $(checks - bad)/$checks ok")
    return bad == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_boxed_reset(; only = ARGS) ? 0 : 1)
end
