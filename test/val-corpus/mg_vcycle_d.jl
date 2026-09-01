function mg_vcycle_d(u, ud, f, fd, r, rd, nfine, num_levels, h1, h1d, nu1, nu2, n)
    nd = 0.0
    n = n * 2
    nld = 0.0
    nl = nfine
    hld = h1d
    hl = h1
    for i_level = 1:num_levels - 1
        nd = 0.0
        n = nl - 1
        __cse_0 = hl * hld
        hl2d = __cse_0 + __cse_0
        hl2 = hl * hl
        for i_k = 1:nu1
            for i_j = 1:n
                leftd = 0.0
                left = 0.0
                if i_j > 1
                    leftd = ud[i_j - 1, i_level]
                    left = u[i_j - 1, i_level]
                end
                rightd = 0.0
                right = 0.0
                if i_j < n
                    rightd = ud[i_j + 1, i_level]
                    right = u[i_j + 1, i_level]
                end
                __cse_1 = f[i_j, i_level]
                ud[i_j, i_level] = 0.5 * (((__cse_1 * hl2d + hl2 * fd[i_j, i_level]) + leftd) + rightd)
                u[i_j, i_level] = 0.5 * (hl2 * __cse_1 + left + right)
            end
        end
        for j = 1:n
            leftd = 0.0
            left = 0.0
            if j > 1
                leftd = ud[j - 1, i_level]
                left = u[j - 1, i_level]
            end
            rightd = 0.0
            right = 0.0
            if j < n
                rightd = ud[j + 1, i_level]
                right = u[j + 1, i_level]
            end
            __cse_2 = (2.0 * u[j, i_level] - left) - right
            rd[j, i_level] = fd[j, i_level] + -(((1.0 / hl2) * ((2.0 * ud[j, i_level] + -leftd) + -rightd) + -(__cse_2 / hl2 ^ 2) * hl2d))
            r[j, i_level] = f[j, i_level] - __cse_2 / hl2
        end
        ncgd = 0.0
        ncg = div(nl, 2)
        ncd = 0.0
        nc = ncg - 1
        for j = 1:nc
            jfd = 0.0
            jf = j * 2
            fd[j, i_level + 1] = (0.25 * rd[jf - 1, i_level] + 0.5 * rd[jf, i_level]) + 0.25 * rd[jf + 1, i_level]
            f[j, i_level + 1] = 0.25 * r[jf - 1, i_level] + 0.5 * r[jf, i_level] + 0.25 * r[jf + 1, i_level]
        end
        for j = 1:nc
            ud[j, i_level + 1] = 0.0
            u[j, i_level + 1] = 0.0
        end
        nld = 0.0
        nl = ncg
        hld = 2.0hld
        hl = hl * 2.0
    end
    __cse_3 = hl * hld
    hl2d = __cse_3 + __cse_3
    hl2 = hl * hl
    __cse_4 = f[1, num_levels]
    ud[1, num_levels] = (0.5__cse_4) * hl2d + (0.5hl2) * fd[1, num_levels]
    u[1, num_levels] = 0.5 * hl2 * __cse_4
    for i_level = num_levels - 1:-1:1
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
        __cse_5 = hl * hld
        hl2d = __cse_5 + __cse_5
        hl2 = hl * hl
        for j = 1:nc
            jfd = 0.0
            jf = j * 2
            ud[jf, i_level] = ud[jf, i_level] + ud[j, i_level + 1]
            u[jf, i_level] = u[jf, i_level] + u[j, i_level + 1]
        end
        for j = 1:nc + 1
            jfd = 0.0
            jf = j * 2 - 1
            cld = 0.0
            cl = 0.0
            if j > 1
                cld = ud[j - 1, i_level + 1]
                cl = u[j - 1, i_level + 1]
            end
            crd = 0.0
            cr = 0.0
            if j <= nc
                crd = ud[j, i_level + 1]
                cr = u[j, i_level + 1]
            end
            ud[jf, i_level] = ud[jf, i_level] + 0.5 * (cld + crd)
            u[jf, i_level] = u[jf, i_level] + 0.5 * (cl + cr)
        end
        for i_k = 1:nu2
            for i_j = 1:n
                leftd = 0.0
                left = 0.0
                if i_j > 1
                    leftd = ud[i_j - 1, i_level]
                    left = u[i_j - 1, i_level]
                end
                rightd = 0.0
                right = 0.0
                if i_j < n
                    rightd = ud[i_j + 1, i_level]
                    right = u[i_j + 1, i_level]
                end
                __cse_6 = f[i_j, i_level]
                ud[i_j, i_level] = 0.5 * (((__cse_6 * hl2d + hl2 * fd[i_j, i_level]) + leftd) + rightd)
                u[i_j, i_level] = 0.5 * (hl2 * __cse_6 + left + right)
            end
        end
    end
    return nothing
end

function mg_vcycle(u, f, r, nfine, num_levels, h1, nu1, nu2, n)
    n = n * 2
    nl = nfine
    hl = h1
    for i_level = 1:num_levels - 1
        n = nl - 1
        hl2 = hl * hl
        for i_k = 1:nu1
            for i_j = 1:n
                left = 0.0
                if i_j > 1
                    left = u[i_j - 1, i_level]
                end
                right = 0.0
                if i_j < n
                    right = u[i_j + 1, i_level]
                end
                u[i_j, i_level] = 0.5 * (hl2 * f[i_j, i_level] + left + right)
            end
        end
        for j = 1:n
            left = 0.0
            if j > 1
                left = u[j - 1, i_level]
            end
            right = 0.0
            if j < n
                right = u[j + 1, i_level]
            end
            r[j, i_level] = f[j, i_level] - ((2.0 * u[j, i_level] - left) - right) / hl2
        end
        ncg = div(nl, 2)
        nc = ncg - 1
        for j = 1:nc
            jf = j * 2
            f[j, i_level + 1] = 0.25 * r[jf - 1, i_level] + 0.5 * r[jf, i_level] + 0.25 * r[jf + 1, i_level]
        end
        for j = 1:nc
            u[j, i_level + 1] = 0.0
        end
        nl = ncg
        hl = hl * 2.0
    end
    hl2 = hl * hl
    u[1, num_levels] = 0.5 * hl2 * f[1, num_levels]
    for i_level = num_levels - 1:-1:1
        nl = nl * 2
        hl = hl / 2.0
        n = nl - 1
        ncg = div(nl, 2)
        nc = ncg - 1
        hl2 = hl * hl
        for j = 1:nc
            jf = j * 2
            u[jf, i_level] = u[jf, i_level] + u[j, i_level + 1]
        end
        for j = 1:nc + 1
            jf = j * 2 - 1
            cl = 0.0
            if j > 1
                cl = u[j - 1, i_level + 1]
            end
            cr = 0.0
            if j <= nc
                cr = u[j, i_level + 1]
            end
            u[jf, i_level] = u[jf, i_level] + 0.5 * (cl + cr)
        end
        for i_k = 1:nu2
            for i_j = 1:n
                left = 0.0
                if i_j > 1
                    left = u[i_j - 1, i_level]
                end
                right = 0.0
                if i_j < n
                    right = u[i_j + 1, i_level]
                end
                u[i_j, i_level] = 0.5 * (hl2 * f[i_j, i_level] + left + right)
            end
        end
    end
end
