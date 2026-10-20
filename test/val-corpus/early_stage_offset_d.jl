function early_stage_offset_d(u, ud, rhs, rhsd, nu, num_levels)
    nld = 0.0
    nl = 2
    ncd = 0.0
    nc = nl - 1
    for i_k = 1:nu
        for i_j = 1:nc
            leftd = 0.0
            left = 0.0
            if i_j > 1
                leftd = ud[i_j - 1, 1]
                left = u[i_j - 1, 1]
            end
            __cse_0 = rhs[i_j, 1] + left
            ud[i_j, 1] = (0.5left) * (rhsd[i_j, 1] + leftd) + (0.5__cse_0) * leftd
            u[i_j, 1] = 0.5 * __cse_0 * left
        end
    end
    for i_level = 2:num_levels
        nld = 0.0
        nl = nl * 2
        ncd = 0.0
        nc = nl - 1
        for j = 1:nl
            cld = 0.0
            cl = 0.0
            if j > 1
                cld = ud[j - 1, i_level - 1]
                cl = u[j - 1, i_level - 1]
            end
            __cse_1 = u[j, i_level - 1]
            ud[j, i_level] = cl * ud[j, i_level - 1] + __cse_1 * cld
            u[j, i_level] = __cse_1 * cl
        end
    end
    return nothing
end

function early_stage_offset(u, rhs, nu, num_levels)
    nl = 2
    nc = nl - 1
    for i_k = 1:nu
        for i_j = 1:nc
            left = 0.0
            if i_j > 1
                left = u[i_j - 1, 1]
            end
            u[i_j, 1] = 0.5 * (rhs[i_j, 1] + left) * left
        end
    end
    for i_level = 2:num_levels
        nl = nl * 2
        nc = nl - 1
        for j = 1:nl
            cl = 0.0
            if j > 1
                cl = u[j - 1, i_level - 1]
            end
            u[j, i_level] = u[j, i_level - 1] * cl
        end
    end
end
