# ==================== validate_split_real.jl =============================
# The real transformer_block adjoint and HVP of the kit (STADE_KIT_KERNELS/nn) through the CPU ownership shim with the split twins forced
# (OWN_SPLIT=L): parity with the plain plan, no ownership conflict, and the twins of the two layer-norm backward kernels must have run.
#   OWN_SPLIT=4 STADE_KIT_KERNELS=<kit kernels dir> julia validate_split_real.jl
get!(ENV, "OWN_FUSE", "1")   # the benchmark kernels are generated with fuse_ii_loops = true
include(joinpath(@__DIR__, "validate_ownership.jl"))
kd = joinpath(get(ENV, "STADE_KIT_KERNELS", "/home/claude/bench/pkg/kernels"), "nn")
res = validate_ownership(kd; only = ["transformer_block"])
tw = sort(unique(n for n in Main.CUDA.LAUNCHED if endswith(n, "_s!")))
println("twins launched: ", tw)
for r in res.rows; println(r.kernel, " ", r.mode, " parity=", r.parity, " shuffled=", r.shuffled_parity, " errs=", r.errs, " vacuous=", r.vacuous, " violations=", length(r.violations)); end
need = ["cuda_kernel_transformer_block_b_15_s!", "cuda_kernel_transformer_block_b_22_s!", "cuda_kernel_transformer_block_hv_15_s!", "cuda_kernel_transformer_block_hv_22_s!"]
miss = [n for n in need if !(n in tw)]
println("layer-norm backward twins missing: ", miss)
exit(res.bad == 0 && isempty(miss) && !any(r.vacuous for r in res.rows) ? 0 : 1)
