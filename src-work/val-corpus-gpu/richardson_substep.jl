# richardson_substep(y_init, out, a_coef, dt_stage, num_stages)
#
# Richardson-extrapolation staging for the scalar linear decay
# dy/dt = -a_coef*y: at stage i_seq_stage the number of explicit-Euler
# substeps doubles (nsub = 1, 2, 4, ... across stages), so each stage
# re-integrates the same fixed time interval dt_stage with a finer step
# size dt_stage/nsub than the stage before it. The value reached at the
# end of each stage's integration is written into out[stage], ready for
# a (separate) Richardson combination step. Because nsub is reassigned
# once per outer i_seq_stage iteration and then used as the nested
# substep loop's trip count, this is a Tier B instance from a genuinely
# different domain (time integration, not spatial grid coarsening) than
# mg_vcycle/mg_vcycle_multi/cascadic_mg_prolong. See
# agen_tier_b_offender/agen_tier_b_walk in STADE.jl.
#
# y_init: initial condition y(t0)
# out: output array of length num_stages; out[stage] receives y at
#    t0 + dt_stage from that stage's substep integration
# a_coef: decay coefficient a in dy/dt = -a_coef*y
# dt_stage: length of the fixed time interval integrated at every stage
# num_stages: number of Richardson stages
function richardson_substep(y_init, out, a_coef, dt_stage, num_stages)
    nsub = 1
    for i_seq_stage = 1:num_stages
        h = dt_stage / nsub
        y = y_init
        # advance nsub explicit-Euler substeps across this stage's interval
        for i_seq_sub = 1:nsub
            y = y - h * a_coef * y
        end
        out[i_seq_stage] = y
        nsub = nsub * 2
    end
    return nothing
end
