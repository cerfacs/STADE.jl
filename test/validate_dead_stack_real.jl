# ==================== validate_dead_stack_real.jl =============================
# The real unet2 adjoint and HVP of the kit (STADE_KIT_KERNELS/nn), generated with fuse_ii_loops = true, through the CPU ownership shim (v0.4.44, dead_stack = 2):
#   - every one of the ten stacks keeps one store and its reads in the plan, and the kept region moves to the start of the array (plan.compact_stacks names all ten);
#   - the shadow stacks of the HVP are never accessed and the host allocates them with one slot;
#   - the plan with dead_stack = 0 keeps the stores (the check measures the change, not a plan that never had the stores);
#   - parity with the CPU, no ownership conflict, same result under a shuffled thread order (validate_ownership, with the allocation shrunk as the generator does).
#   STADE_KIT_KERNELS=<kit kernels dir> [STADE_KIT_TOOLS=<kit tools dir>] [DS_SIZE=tiny|small] julia validate_dead_stack_real.jl
get!(ENV, "OWN_FUSE", "1")
include(joinpath(@__DIR__, "validate_ownership.jl"))
kd = joinpath(get(ENV, "STADE_KIT_KERNELS", "/home/claude/bench/pkg/kernels"), "nn")
bad = 0
check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (global bad += 1))
src = Meta.parse(read(joinpath(kd, "unet2.jl"), String))
backend = STADE.cgen_backend_cuda()
nstore(plan) = sum(count(l -> occursin(r"^\s*\w+_stack(_d)?\[.*\] = ", l), split(string(k), "\n")) for k in plan.kernels if k.head === :function; init = 0)
for mode in (:adjoint, :hvp)
    gen = mode == :hvp ? STADE.stade_hvp(src; keep_push_pop = false, fuse_ii_loops = true) : STADE.stade_adjoint(src; keep_push_pop = false, fuse_ii_loops = true)
    gexpr = mode == :hvp ? gen.hvp : gen.adjoint
    pinit = STADE.stade_gpu(gen.initstacks, backend)
    ih = Dict{Symbol,Expr}(pinit.host.args[1].args[1] => pinit.host)
    p2 = STADE.stade_gpu(gexpr, backend; init_hosts = ih)
    p0 = STADE.stade_gpu(gexpr, backend; dead_stack = 0)
    check(length(p2.compact_stacks) == 10, "$mode: plan.compact_stacks names $(length(p2.compact_stacks)) of the 10 stacks")
    check(nstore(p0) > nstore(p2) > 0, "$mode: stores to stacks $(nstore(p0)) -> $(nstore(p2))")
    ini = string(STADE.cgen_shrink_stacks(pinit.host, p2.dead_stacks, p2.compact_stacks))
    check(count("undef", ini) == 10 && !occursin(r"undef, \(\(\(", ini), "$mode: the ten allocations lost their prefix sums")
    if mode == :hvp
        check(occursin("s7_stack_d = CuArray{Float64}(undef, (1,))", string(p2.host)), "hvp: the shadow stack of s7 has one slot")
        check(!occursin(r"_stack_d\[", join(string.(p2.kernels), "\n")), "hvp: no kernel accesses a shadow stack")
        check(occursin(r"_stack_d\[", join(string.(p0.kernels), "\n")), "hvp: with dead_stack = 0 the kernels do")
    end
end
include(joinpath(get(ENV, "STADE_KIT_TOOLS", joinpath(dirname(dirname(kd)), "tools")), "inputs.jl"))   # bench_inputs: the sizes the random baseline cannot draw
# validate_ownership defines constants in Main, so one call per process: DS_SIZE = tiny (default) or small
for sz in (get(ENV, "DS_SIZE", "tiny"),)
    vals = Dict{Symbol,Any}(bench_inputs("unet2", sz))
    ints = Dict{Symbol,Int}(k => v for (k, v) in vals if v isa Integer)
    res = validate_ownership(kd; only = ["unet2"], baseline_fn = kernel -> (int_args = ints, values = deepcopy(vals)))
    for r in res.rows; println(sz, " ", r.kernel, " ", r.mode, " parity=", r.parity, " shuffled=", r.shuffled_parity, " errs=", r.errs, " vacuous=", r.vacuous, " violations=", length(r.violations), " ", r.error); end
    check(res.bad == 0 && length(res.rows) == 2 && all(r -> !r.vacuous && r.parity && r.shuffled_parity && isempty(r.violations), res.rows), "unet2 $sz, adjoint and HVP: parity, shuffled parity, no conflict, not vacuous")
end
println(bad == 0 ? "all dead_stack real checks passed" : "*** $bad dead_stack real check(s) failed ***")
exit(bad == 0 ? 0 : 1)
