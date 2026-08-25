# cascadic_mg_prolong(u, rhs, h_coarse, nu, num_levels)
#
# Cascadic multigrid (nested iteration) for a 1-D Poisson-like problem:
# starts on the coarsest grid, then for each successively finer level
# linearly prolongs the previous level's solution onto the new grid
# (twice as many interior points) before relaxing there with nu Jacobi
# sweeps. Unlike a V-cycle there is no restriction phase, so the active
# point count only GROWS across levels -- this isolates the growing-bound
# half of mg_vcycle's ragged nl/n sequence as its own Tier B instance
# (mg_vcycle's single sequential pass mixes a shrinking phase followed by
# a growing one; agen_tier_b_offender only ever reports the first
# offender it meets, so the growing-only case has never appeared in the
# corpus in isolation). See agen_tier_b_offender/agen_tier_b_walk in
# STADE.jl.
#
# u: solution storage, shape (n_fine, num_levels); level lvl's interior
#    points are u[1..nl(lvl)-1, lvl], caller-allocated large enough for
#    the finest level
# rhs: right-hand side storage, same shape as u; rhs[j, lvl] holds the
#    source term at level lvl's j-th interior point, caller-filled
# h_coarse: grid spacing at the coarsest level (level 1)
# nu: number of Jacobi relaxation sweeps applied at every level
# num_levels: number of grid levels, coarsest (1) to finest
function cascadic_mg_prolong(u, rhs, h_coarse, nu, num_levels)
    nl = 2
    hl = h_coarse
    nc = nl - 1
    hl2 = hl * hl
    # coarsest level: relax directly against the caller-supplied rhs
    for i_k = 1:nu
        for i_j = 1:nc
            left = 0.0
            if i_j > 1
                left = u[i_j - 1, 1]
            end
            right = 0.0
            if i_j < nc
                right = u[i_j + 1, 1]
            end
            u[i_j, 1] = 0.5 * (hl2 * rhs[i_j, 1] + left + right)
        end
    end
    for i_level = 2:num_levels
        nl = nl * 2
        hl = hl / 2.0
        nc = nl - 1
        hl2 = hl * hl
        ncoarse = div(nl, 2) - 1
        # prolong: even fine points copy the matching coarse point
        for j = 1:ncoarse
            jf = j * 2
            u[jf, i_level] = u[j, i_level - 1]
        end
        # odd fine points get the average of their two coarse neighbors
        for j = 1:ncoarse + 1
            jf = j * 2 - 1
            cl = 0.0
            if j > 1
                cl = u[j - 1, i_level - 1]
            end
            cr = 0.0
            if j <= ncoarse
                cr = u[j, i_level - 1]
            end
            u[jf, i_level] = 0.5 * (cl + cr)
        end
        # relax nu sweeps on the new, finer grid
        for i_k = 1:nu
            for i_j = 1:nc
                left = 0.0
                if i_j > 1
                    left = u[i_j - 1, i_level]
                end
                right = 0.0
                if i_j < nc
                    right = u[i_j + 1, i_level]
                end
                u[i_j, i_level] = 0.5 * (hl2 * rhs[i_j, i_level] + left + right)
            end
        end
    end
    return nothing
end
