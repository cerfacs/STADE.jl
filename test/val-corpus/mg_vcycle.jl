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
