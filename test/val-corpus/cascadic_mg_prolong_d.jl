function cascadic_mg_prolong_d(u, ud, rhs, rhsd, h_coarse, h_coarsed, nu, num_levels)
    nld = 0.0
    nl = 2
    hld = h_coarsed
    hl = h_coarse
    ncd = 0.0
    nc = nl - 1
    hl2d = hl * hld + hl * hld
    hl2 = hl * hl
    for i_seq_k = 1:nu
        for i_seq_j = 1:nc
            leftd = 0.0
            left = 0.0
            if i_seq_j > 1
                leftd = ud[i_seq_j - 1, 1]
                left = u[i_seq_j - 1, 1]
            end
            rightd = 0.0
            right = 0.0
            if i_seq_j < nc
                rightd = ud[i_seq_j + 1, 1]
                right = u[i_seq_j + 1, 1]
            end
            ud[i_seq_j, 1] = 0.5 * (((rhs[i_seq_j, 1] * hl2d + hl2 * rhsd[i_seq_j, 1]) + leftd) + rightd)
            u[i_seq_j, 1] = 0.5 * (hl2 * rhs[i_seq_j, 1] + left + right)
        end
    end
    for i_seq_level = 2:num_levels
        nld = 0.0
        nl = nl * 2
        hld = 0.5hld
        hl = hl / 2.0
        ncd = 0.0
        nc = nl - 1
        hl2d = hl * hld + hl * hld
        hl2 = hl * hl
        ncoarsed = 0.0
        ncoarse = div(nl, 2) - 1
        for j = 1:ncoarse
            jfd = 0.0
            jf = j * 2
            ud[jf, i_seq_level] = ud[j, i_seq_level - 1]
            u[jf, i_seq_level] = u[j, i_seq_level - 1]
        end
        for j = 1:ncoarse + 1
            jfd = 0.0
            jf = j * 2 - 1
            cld = 0.0
            cl = 0.0
            if j > 1
                cld = ud[j - 1, i_seq_level - 1]
                cl = u[j - 1, i_seq_level - 1]
            end
            crd = 0.0
            cr = 0.0
            if j <= ncoarse
                crd = ud[j, i_seq_level - 1]
                cr = u[j, i_seq_level - 1]
            end
            ud[jf, i_seq_level] = 0.5 * (cld + crd)
            u[jf, i_seq_level] = 0.5 * (cl + cr)
        end
        for i_seq_k = 1:nu
            for i_seq_j = 1:nc
                leftd = 0.0
                left = 0.0
                if i_seq_j > 1
                    leftd = ud[i_seq_j - 1, i_seq_level]
                    left = u[i_seq_j - 1, i_seq_level]
                end
                rightd = 0.0
                right = 0.0
                if i_seq_j < nc
                    rightd = ud[i_seq_j + 1, i_seq_level]
                    right = u[i_seq_j + 1, i_seq_level]
                end
                ud[i_seq_j, i_seq_level] = 0.5 * (((rhs[i_seq_j, i_seq_level] * hl2d + hl2 * rhsd[i_seq_j, i_seq_level]) + leftd) + rightd)
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * rhs[i_seq_j, i_seq_level] + left + right)
            end
        end
    end
    return nothing
end

function cascadic_mg_prolong(u, rhs, h_coarse, nu, num_levels)
    nl = 2
    hl = h_coarse
    nc = nl - 1
    hl2 = hl * hl
    for i_seq_k = 1:nu
        for i_seq_j = 1:nc
            left = 0.0
            if i_seq_j > 1
                left = u[i_seq_j - 1, 1]
            end
            right = 0.0
            if i_seq_j < nc
                right = u[i_seq_j + 1, 1]
            end
            u[i_seq_j, 1] = 0.5 * (hl2 * rhs[i_seq_j, 1] + left + right)
        end
    end
    for i_seq_level = 2:num_levels
        nl = nl * 2
        hl = hl / 2.0
        nc = nl - 1
        hl2 = hl * hl
        ncoarse = div(nl, 2) - 1
        for j = 1:ncoarse
            jf = j * 2
            u[jf, i_seq_level] = u[j, i_seq_level - 1]
        end
        for j = 1:ncoarse + 1
            jf = j * 2 - 1
            cl = 0.0
            if j > 1
                cl = u[j - 1, i_seq_level - 1]
            end
            cr = 0.0
            if j <= ncoarse
                cr = u[j, i_seq_level - 1]
            end
            u[jf, i_seq_level] = 0.5 * (cl + cr)
        end
        for i_seq_k = 1:nu
            for i_seq_j = 1:nc
                left = 0.0
                if i_seq_j > 1
                    left = u[i_seq_j - 1, i_seq_level]
                end
                right = 0.0
                if i_seq_j < nc
                    right = u[i_seq_j + 1, i_seq_level]
                end
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * rhs[i_seq_j, i_seq_level] + left + right)
            end
        end
    end
end
