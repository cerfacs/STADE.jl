# ==================== validate_contraction_real.jl =======================
# The real transformer_block HVP of the kit (STADE_KIT_KERNELS/nn) through the CPU ownership shim with the contraction twins forced (OWN_CONTRACT=n):
# parity with the plain plan, no ownership conflict, no vacuous row, and the tiled kernels of the three backward loops that v0.4.40 tiles in the HVP
# (hv_12, hv_13, hv_20: four tiled kernels each, one per atomic statement, with one or two product terms) must have run, and the kernels that replace the loops.
#   OWN_CONTRACT=1 STADE_KIT_KERNELS=<kit kernels dir> julia validate_contraction_real.jl
get!(ENV, "OWN_FUSE", "1")   # the benchmark kernels are generated with fuse_ii_loops = true
get!(ENV, "OWN_CONTRACT", "1")
include(joinpath(@__DIR__, "validate_ownership.jl"))
kd = joinpath(get(ENV, "STADE_KIT_KERNELS", "/home/claude/bench/pkg/kernels"), "nn")
res = validate_ownership(kd; only = ["transformer_block"])
tw = sort(unique(n for n in Main.CUDA.LAUNCHED if occursin(r"_t\d+!$", n) || endswith(n, "_a!")))
println("tiled kernels launched: ", tw)
for r in res.rows; println(r.kernel, " ", r.mode, " parity=", r.parity, " shuffled=", r.shuffled_parity, " errs=", r.errs, " vacuous=", r.vacuous, " violations=", length(r.violations)); end
need = String[]
for k in (12, 13, 20)
    push!(need, "cuda_kernel_transformer_block_hv_$(k)_a!")
    for j in 1:4; push!(need, "cuda_kernel_transformer_block_hv_$(k)_t$(j)!"); end
end
miss = [n for n in need if !(n in tw)]
println("tiled HVP kernels missing: ", miss)
exit(res.bad == 0 && isempty(miss) && !any(r.vacuous for r in res.rows) ? 0 : 1)
