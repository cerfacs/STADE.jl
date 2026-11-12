# A `while` written as a counted loop with a guard (`for k = 1:maxit`, `if cond`) is a GPU target with keep_push_pop = false.
# Two checks on the CUDA plans of the counted-corpus kernels, for the adjoint and the HVP:
#  (1) every stack the host still indexes, or whose shadow the host still indexes, keeps its full size after the dead-stack shrink.
#      A V100 run found `st_stack` shrunk to one element while the HVP host read `st_stack_d[27]`.
#  (2) sabotage: forcing the old shrink makes check (1) fail, so the check is not vacuous.
# usage: julia validate_counted_while.jl
include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

fails = 0
report(ok, msg) = (global fails; ok || (fails += 1); println(ok ? "ok   " : "FAIL ", msg))

const CW_DIR = joinpath(@__DIR__, "counted-corpus")

# stack names (without a `_d` suffix) that the host indexes
function cw_host_indexed(e, out = Set{Symbol}())
    if e isa Expr
        e.head === :ref && e.args[1] isa Symbol && endswith(string(e.args[1]), "_stack_d") && push!(out, Symbol(string(e.args[1])[1:end-2]))
        e.head === :ref && e.args[1] isa Symbol && endswith(string(e.args[1]), "_stack") && push!(out, e.args[1])
        for a in e.args; cw_host_indexed(a, out); end
    end
    return out
end
# stacks that the shrunk initstacks host allocates with a literal one-element tuple
function cw_shrunk(init)
    out = Set{Symbol}()
    for st in init.args[2].args
        st isa Expr && st.head === :(=) && st.args[1] isa Symbol && st.args[2] isa Expr && st.args[2].head === :call &&
            st.args[2].args[end] isa Expr && st.args[2].args[end].head === :tuple && length(st.args[2].args[end].args) == 1 && push!(out, st.args[1])
    end
    return out
end

function cw_plan(e, mode; force_dead = nothing)
    gen = mode == :hvp ? STADE.stade_hvp(e; keep_push_pop = false) : STADE.stade_adjoint(e; keep_push_pop = false)
    expr = mode == :hvp ? gen.hvp : gen.adjoint
    bk = STADE.cgen_backend_cuda()
    pi = STADE.stade_gpu(gen.initstacks, bk)
    pg = STADE.stade_gpu(expr, bk; init_hosts = Dict{Symbol,Expr}(pi.host.args[1].args[1] => pi.host))
    dead = force_dead === nothing ? pg.dead_stacks : force_dead
    init = STADE.cgen_shrink_stacks(pi.host, dead, pg.compact_stacks)
    return (host = pg.host, init = init)
end

# phc_mg_solve is skipped: its initstacks does not reach the GPU parser yet (open defect, tracked for the next release)
const CW_SKIP = Set(["phc_mg_solve.jl"])
for f in sort(readdir(CW_DIR)); endswith(f, ".jl") || continue
    f in CW_SKIP && (println("skip ", f, " (open defect)"); continue)
    e = STADE.io_read_corpus_entry(joinpath(CW_DIR, f))
    for mode in (:adjoint, :hvp)
        p = cw_plan(e, mode)
        bad = intersect(cw_host_indexed(p.host), cw_shrunk(p.init))
        report(isempty(bad), "$f $mode: no host-indexed stack is shrunk $(sort(collect(bad)))")
    end
end
# sabotage: forcing the shrink of a stack the host still indexes must be caught. The kernels that stay on the host (gs1d and the other carried-value
# solvers) are the ones that index a stack; the offloaded element-wise kernels no longer do, so the target is looked up and not named.
sab = nothing
for f in sort(readdir(CW_DIR)); endswith(f, ".jl") && f != "phc_mg_solve.jl" || continue
    e = STADE.io_read_corpus_entry(joinpath(CW_DIR, f))
    hi = cw_host_indexed(cw_plan(e, :hvp).host)
    if !isempty(hi); global sab = (f, e, sort(collect(hi))); break; end
end
report(sab !== nothing, "sabotage: a counted kernel whose HVP host indexes a stack exists")
if sab !== nothing
    p = cw_plan(sab[2], :hvp; force_dead = sab[3])
    report(!isempty(intersect(cw_host_indexed(p.host), cw_shrunk(p.init))), "sabotage: forced shrink of $(sab[3]) in $(sab[1]) is caught")
end
println(fails == 0 ? "ALL OK" : "$fails FAILED")
exit(fails == 0 ? 0 : 1)
