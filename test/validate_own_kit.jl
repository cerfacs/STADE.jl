# ==================== validate_own_kit.jl =============================
# One kit kernel (adjoint and HVP) through the CPU ownership shim with the kit's own inputs (bench_inputs), not the random baseline.
# The random baseline cannot draw a physical state (euler2d_step: sqrt of a negative, swe2d_step the same), so validate_ownership on
# those kernels errors out before it checks anything; this file is the check that runs them.
#   OWN_KERNEL=solvers/euler2d_step [OWN_SIZE=tiny] [OWN_SPLIT=8] STADE_KIT_KERNELS=<kit kernels dir> [STADE_KIT_TOOLS=<kit tools dir>] julia validate_own_kit.jl
# validate_ownership defines constants in Main, so one kernel per process.
get!(ENV, "OWN_FUSE", "1")
include(joinpath(@__DIR__, "validate_ownership.jl"))
kroot = get(ENV, "STADE_KIT_KERNELS", "/home/claude/bench/pkg/kernels")
sub, name = split(get(ENV, "OWN_KERNEL", "solvers/euler2d_step"), "/")
sz = get(ENV, "OWN_SIZE", "tiny")
include(joinpath(get(ENV, "STADE_KIT_TOOLS", joinpath(dirname(kroot), "tools")), "inputs.jl"))
vals = Dict{Symbol,Any}(bench_inputs(String(name), sz))
ints = Dict{Symbol,Int}(k => v for (k, v) in vals if v isa Integer)
res = validate_ownership(joinpath(kroot, sub); only = [String(name)], baseline_fn = kernel -> (int_args = ints, values = deepcopy(vals)))
for r in res.rows; println(sz, " ", r.kernel, " ", r.mode, " parity=", r.parity, " shuffled=", r.shuffled_parity, " errs=", r.errs, " vacuous=", r.vacuous, " violations=", length(r.violations), " ", r.error); end
ok = res.bad == 0 && length(res.rows) == 2 && all(r -> !r.vacuous && r.parity && r.shuffled_parity && isempty(r.violations) && isempty(r.error), res.rows)
println(ok ? "OWN_KIT ok" : "OWN_KIT FAIL")
exit(ok ? 0 : 1)
