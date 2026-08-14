function mg_relax_d(u, ud, f, fd, n, hl2, hl2d, lvl, nu)
    for i_seq_k = 1:nu
        for i_seq_j = 1:n
            leftd = 0.0
            left = 0.0
            if i_seq_j > 1
                leftd = ud[i_seq_j - 1, lvl]
                left = u[i_seq_j - 1, lvl]
            end
            rightd = 0.0
            right = 0.0
            if i_seq_j < n
                rightd = ud[i_seq_j + 1, lvl]
                right = u[i_seq_j + 1, lvl]
            end
            ud[i_seq_j, lvl] = 0.5 * (((f[i_seq_j, lvl] * hl2d + hl2 * fd[i_seq_j, lvl]) + leftd) + rightd)
            u[i_seq_j, lvl] = 0.5 * (hl2 * f[i_seq_j, lvl] + left + right)
        end
    end
    return nothing
end

function mg_relax(u, f, n, hl2, lvl, nu)
    for i_seq_k = 1:nu
        for i_seq_j = 1:n
            left = 0.0
            if i_seq_j > 1
                left = u[i_seq_j - 1, lvl]
            end
            right = 0.0
            if i_seq_j < n
                right = u[i_seq_j + 1, lvl]
            end
            u[i_seq_j, lvl] = 0.5 * (hl2 * f[i_seq_j, lvl] + left + right)
        end
    end
end

function mg_vcycle_multi_d(u, ud, f, fd, r, rd, nfine, num_levels, h1, h1d, nu1, nu2, n)
    nd = 0.0
    n = n * 2
    nld = 0.0
    nl = nfine
    hld = h1d
    hl = h1
    for i_seq_level = 1:num_levels - 1
        nd = 0.0
        n = nl - 1
        hl2d = hl * hld + hl * hld
        hl2 = hl * hl
        for i_seq_k_mg_relax_c1 = 1:nu1
            for i_seq_j_mg_relax_c1 = 1:n
                left_mg_relax_c1d = 0.0
                left_mg_relax_c1 = 0.0
                if i_seq_j_mg_relax_c1 > 1
                    left_mg_relax_c1d = ud[i_seq_j_mg_relax_c1 - 1, i_seq_level]
                    left_mg_relax_c1 = u[i_seq_j_mg_relax_c1 - 1, i_seq_level]
                end
                right_mg_relax_c1d = 0.0
                right_mg_relax_c1 = 0.0
                if i_seq_j_mg_relax_c1 < n
                    right_mg_relax_c1d = ud[i_seq_j_mg_relax_c1 + 1, i_seq_level]
                    right_mg_relax_c1 = u[i_seq_j_mg_relax_c1 + 1, i_seq_level]
                end
                ud[i_seq_j_mg_relax_c1, i_seq_level] = 0.5 * (((f[i_seq_j_mg_relax_c1, i_seq_level] * hl2d + hl2 * fd[i_seq_j_mg_relax_c1, i_seq_level]) + left_mg_relax_c1d) + right_mg_relax_c1d)
                u[i_seq_j_mg_relax_c1, i_seq_level] = 0.5 * (hl2 * f[i_seq_j_mg_relax_c1, i_seq_level] + left_mg_relax_c1 + right_mg_relax_c1)
            end
        end
        for j = 1:n
            leftd = 0.0
            left = 0.0
            if j > 1
                leftd = ud[j - 1, i_seq_level]
                left = u[j - 1, i_seq_level]
            end
            rightd = 0.0
            right = 0.0
            if j < n
                rightd = ud[j + 1, i_seq_level]
                right = u[j + 1, i_seq_level]
            end
            rd[j, i_seq_level] = fd[j, i_seq_level] + -(((1.0 / hl2) * ((2.0 * ud[j, i_seq_level] + -leftd) + -rightd) + -(((2.0 * u[j, i_seq_level] - left) - right) / hl2 ^ 2) * hl2d))
            r[j, i_seq_level] = f[j, i_seq_level] - ((2.0 * u[j, i_seq_level] - left) - right) / hl2
        end
        ncgd = 0.0
        ncg = div(nl, 2)
        ncd = 0.0
        nc = ncg - 1
        for j = 1:nc
            jfd = 0.0
            jf = j * 2
            fd[j, i_seq_level + 1] = (0.25 * rd[jf - 1, i_seq_level] + 0.5 * rd[jf, i_seq_level]) + 0.25 * rd[jf + 1, i_seq_level]
            f[j, i_seq_level + 1] = 0.25 * r[jf - 1, i_seq_level] + 0.5 * r[jf, i_seq_level] + 0.25 * r[jf + 1, i_seq_level]
        end
        for j = 1:nc
            ud[j, i_seq_level + 1] = 0.0
            u[j, i_seq_level + 1] = 0.0
        end
        nld = 0.0
        nl = ncg
        hld = 2.0hld
        hl = hl * 2.0
    end
    hl2d = hl * hld + hl * hld
    hl2 = hl * hl
    ud[1, num_levels] = (0.5 * f[1, num_levels]) * hl2d + (0.5hl2) * fd[1, num_levels]
    u[1, num_levels] = 0.5 * hl2 * f[1, num_levels]
    for i_seq_level = num_levels - 1:-1:1
        nld = 0.0
        nl = nl * 2
        hld = 0.5hld
        hl = hl / 2.0
        nd = 0.0
        n = nl - 1
        ncgd = 0.0
        ncg = div(nl, 2)
        ncd = 0.0
        nc = ncg - 1
        hl2d = hl * hld + hl * hld
        hl2 = hl * hl
        for j = 1:nc
            jfd = 0.0
            jf = j * 2
            ud[jf, i_seq_level] = ud[jf, i_seq_level] + ud[j, i_seq_level + 1]
            u[jf, i_seq_level] = u[jf, i_seq_level] + u[j, i_seq_level + 1]
        end
        for j = 1:nc + 1
            jfd = 0.0
            jf = j * 2 - 1
            cld = 0.0
            cl = 0.0
            if j > 1
                cld = ud[j - 1, i_seq_level + 1]
                cl = u[j - 1, i_seq_level + 1]
            end
            crd = 0.0
            cr = 0.0
            if j <= nc
                crd = ud[j, i_seq_level + 1]
                cr = u[j, i_seq_level + 1]
            end
            ud[jf, i_seq_level] = ud[jf, i_seq_level] + 0.5 * (cld + crd)
            u[jf, i_seq_level] = u[jf, i_seq_level] + 0.5 * (cl + cr)
        end
        for i_seq_k_mg_relax_c2 = 1:nu2
            for i_seq_j_mg_relax_c2 = 1:n
                left_mg_relax_c2d = 0.0
                left_mg_relax_c2 = 0.0
                if i_seq_j_mg_relax_c2 > 1
                    left_mg_relax_c2d = ud[i_seq_j_mg_relax_c2 - 1, i_seq_level]
                    left_mg_relax_c2 = u[i_seq_j_mg_relax_c2 - 1, i_seq_level]
                end
                right_mg_relax_c2d = 0.0
                right_mg_relax_c2 = 0.0
                if i_seq_j_mg_relax_c2 < n
                    right_mg_relax_c2d = ud[i_seq_j_mg_relax_c2 + 1, i_seq_level]
                    right_mg_relax_c2 = u[i_seq_j_mg_relax_c2 + 1, i_seq_level]
                end
                ud[i_seq_j_mg_relax_c2, i_seq_level] = 0.5 * (((f[i_seq_j_mg_relax_c2, i_seq_level] * hl2d + hl2 * fd[i_seq_j_mg_relax_c2, i_seq_level]) + left_mg_relax_c2d) + right_mg_relax_c2d)
                u[i_seq_j_mg_relax_c2, i_seq_level] = 0.5 * (hl2 * f[i_seq_j_mg_relax_c2, i_seq_level] + left_mg_relax_c2 + right_mg_relax_c2)
            end
        end
    end
    return nothing
end

function mg_vcycle_multi(u, f, r, nfine, num_levels, h1, nu1, nu2, n)
    n = n * 2
    nl = nfine
    hl = h1
    for i_seq_level = 1:num_levels - 1
        n = nl - 1
        hl2 = hl * hl
        mg_relax(u, f, n, hl2, i_seq_level, nu1)
        for j = 1:n
            left = 0.0
            if j > 1
                left = u[j - 1, i_seq_level]
            end
            right = 0.0
            if j < n
                right = u[j + 1, i_seq_level]
            end
            r[j, i_seq_level] = f[j, i_seq_level] - ((2.0 * u[j, i_seq_level] - left) - right) / hl2
        end
        ncg = div(nl, 2)
        nc = ncg - 1
        for j = 1:nc
            jf = j * 2
            f[j, i_seq_level + 1] = 0.25 * r[jf - 1, i_seq_level] + 0.5 * r[jf, i_seq_level] + 0.25 * r[jf + 1, i_seq_level]
        end
        for j = 1:nc
            u[j, i_seq_level + 1] = 0.0
        end
        nl = ncg
        hl = hl * 2.0
    end
    hl2 = hl * hl
    u[1, num_levels] = 0.5 * hl2 * f[1, num_levels]
    for i_seq_level = num_levels - 1:-1:1
        nl = nl * 2
        hl = hl / 2.0
        n = nl - 1
        ncg = div(nl, 2)
        nc = ncg - 1
        hl2 = hl * hl
        for j = 1:nc
            jf = j * 2
            u[jf, i_seq_level] = u[jf, i_seq_level] + u[j, i_seq_level + 1]
        end
        for j = 1:nc + 1
            jf = j * 2 - 1
            cl = 0.0
            if j > 1
                cl = u[j - 1, i_seq_level + 1]
            end
            cr = 0.0
            if j <= nc
                cr = u[j, i_seq_level + 1]
            end
            u[jf, i_seq_level] = u[jf, i_seq_level] + 0.5 * (cl + cr)
        end
        mg_relax(u, f, n, hl2, i_seq_level, nu2)
    end
end
