# ph_mg_solve
# Multigrid V-cycles for -u'' = f on nfine-1 interior points, repeated until the squared fine-grid residual falls below tol. Level l is column l of the arrays; nfine = 2^(num_levels-1) * 2.
# LOOP lv1 S H
# LOOP sk1 S H
# LOOP sj1 S H
# LOOP jr I K
# LOOP jc I K
# LOOP jz I K
# LOOP lv2 S H
# LOOP jp I K
# LOOP jq I K
# LOOP sk2 S H
# LOOP sj2 S H
# LOOP jn I K
# WHILE rn H
function ph_mg_solve(u, f, r, nfine, num_levels, h1, nu1, nu2, tol, its)
    rn = 1.0
    cyc = 0
    while rn > tol
        nl = nfine
        hl = h1
        # down: smooth, residual, restrict
        for lv1 = 1:num_levels - 1
            n = nl - 1
            hl2 = hl * hl
            # Gauss-Seidel smoothing: each point reads its left neighbour from the same sweep
            for sk1 = 1:nu1
                for sj1 = 1:n
                    left = 0.0
                    if sj1 > 1
                        left = u[sj1 - 1, lv1]
                    end
                    right = 0.0
                    if sj1 < n
                        right = u[sj1 + 1, lv1]
                    end
                    u[sj1, lv1] = 0.5 * (hl2 * f[sj1, lv1] + left + right)
                end
            end
            # residual: independent per point
            for jr = 1:n
                left = 0.0
                if jr > 1
                    left = u[jr - 1, lv1]
                end
                right = 0.0
                if jr < n
                    right = u[jr + 1, lv1]
                end
                r[jr, lv1] = f[jr, lv1] - ((2.0 * u[jr, lv1] - left) - right) / hl2
            end
            ncg = div(nl, 2)
            nc = ncg - 1
            # full-weighting restriction
            for jc = 1:nc
                jf = jc * 2
                f[jc, lv1 + 1] = 0.25 * r[jf - 1, lv1] + 0.5 * r[jf, lv1] + 0.25 * r[jf + 1, lv1]
            end
            for jz = 1:nc
                u[jz, lv1 + 1] = 0.0
            end
            nl = ncg
            hl = hl * 2.0
        end
        # coarsest level: one point, solved exactly
        hl2 = hl * hl
        u[1, num_levels] = 0.5 * hl2 * f[1, num_levels]
        # up: prolong, correct, smooth
        for lv2 = num_levels - 1:-1:1
            nl = nl * 2
            hl = hl / 2.0
            n = nl - 1
            ncg = div(nl, 2)
            nc = ncg - 1
            hl2 = hl * hl
            for jp = 1:nc
                jf = jp * 2
                u[jf, lv2] = u[jf, lv2] + u[jp, lv2 + 1]
            end
            for jq = 1:nc + 1
                jf = jq * 2 - 1
                cl = 0.0
                if jq > 1
                    cl = u[jq - 1, lv2 + 1]
                end
                cr = 0.0
                if jq <= nc
                    cr = u[jq, lv2 + 1]
                end
                u[jf, lv2] = u[jf, lv2] + 0.5 * (cl + cr)
            end
            for sk2 = 1:nu2
                for sj2 = 1:n
                    left = 0.0
                    if sj2 > 1
                        left = u[sj2 - 1, lv2]
                    end
                    right = 0.0
                    if sj2 < n
                        right = u[sj2 + 1, lv2]
                    end
                    u[sj2, lv2] = 0.5 * (hl2 * f[sj2, lv2] + left + right)
                end
            end
        end
        # squared residual on the fine grid: a scalar reduction
        hf2 = h1 * h1
        nfm = nfine - 1
        rn = 0.0
        for jn = 1:nfm
            left = 0.0
            if jn > 1
                left = u[jn - 1, 1]
            end
            right = 0.0
            if jn < nfm
                right = u[jn + 1, 1]
            end
            rn = rn + (f[jn, 1] - ((2.0 * u[jn, 1] - left) - right) / hf2) * (f[jn, 1] - ((2.0 * u[jn, 1] - left) - right) / hf2)
        end
        cyc = cyc + 1
    end
    its[1] = cyc
    return nothing
end
