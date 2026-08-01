function initstacks_mg_vcycle_b(f, u)
    n_stack = Int[]
    f_stack = Vector{typeof(f)}()
    branch_stack = Int[]
    hl_stack = Float64[]
    hl2_stack = Float64[]
    u_stack = Vector{typeof(u)}()
    integer8_stack = Int[]
    return (n_stack, f_stack, branch_stack, hl_stack, hl2_stack, u_stack, integer8_stack)
end
function mg_vcycle_b(u, ub, f, fb, r, rb, nfine, num_levels, h1, h1b, nu1, nu2, n, n_stack, f_stack, branch_stack, hl_stack, hl2_stack, u_stack, integer8_stack)
    nl = nfine
    hl = h1
    for i_seq_level = 1:num_levels - 1
        n = nl - 1
        hl2 = hl * hl
        for i_seq_k = 1:nu1
            for i_seq_j = 1:n
                left = 0.0
                if i_seq_j > 1
                    left = u[i_seq_j - 1, i_seq_level]
                    push!(branch_stack, 0)
                else
                    push!(branch_stack, 1)
                end
                right = 0.0
                if i_seq_j < n
                    right = u[i_seq_j + 1, i_seq_level]
                    push!(branch_stack, 0)
                else
                    push!(branch_stack, 1)
                end
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
            end
            # push!(integer8_stack, i_seq_j - 1)
            push!(integer8_stack, n)
        end
        push!(n_stack, n)
        push!(u_stack, copy(u))
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
        push!(f_stack, copy(f))
        for j = 1:nc
            jf = j * 2
            f[j, i_seq_level + 1] = 0.25 * r[jf - 1, i_seq_level] + 0.5 * r[jf, i_seq_level] + 0.25 * r[jf + 1, i_seq_level]
        end
        push!(u_stack, copy(u))
        for j = 1:nc
            u[j, i_seq_level + 1] = 0.0
        end
        nl = ncg
        push!(hl_stack, hl)
        hl = hl * 2.0
    end
    # Add initialization of 'nc' 
    # to be able to register its latest value
    # for later use in backward pass
    nc = 1
    for i_seq_level = num_levels - 1:-1:1
        nl = nl * 2
        push!(hl_stack, hl)
        hl = hl / 2.0
        n = nl - 1
        ncg = div(nl, 2)
        nc = ncg - 1
        push!(hl2_stack, hl2)
        hl2 = hl * hl
        for i_seq_k = 1:nu2
            for i_seq_j = 1:n
                if i_seq_j > 1
                    push!(branch_stack, 0)
                else
                    push!(branch_stack, 1)
                end
                if i_seq_j < n
                    push!(branch_stack, 0)
                else
                    push!(branch_stack, 1)
                end
            end
            # push!(integer8_stack, i_seq_j - 1)
            push!(integer8_stack, n)
        end
    end
    hlb = 0.0
    for i_seq_level = 1:1:num_levels - 1
        hl2b = 0.0
        for i_seq_k = nu2:-1:1
            ad_to0 = pop!(integer8_stack)
            for i_seq_j = ad_to0:-1:1
                tempb = 0.5 * ub[i_seq_j, i_seq_level]
                ub[i_seq_j, i_seq_level] = 0.0
                hl2b = hl2b + f[i_seq_j, i_seq_level] * tempb
                fb[i_seq_j, i_seq_level] = fb[i_seq_j, i_seq_level] + hl2 * tempb
                leftb = tempb
                rightb = tempb
                branch = pop!(branch_stack)
                if branch == 0
                    ub[i_seq_j + 1, i_seq_level] = ub[i_seq_j + 1, i_seq_level] + rightb
                end
                branch = pop!(branch_stack)
                if branch == 0
                    ub[i_seq_j - 1, i_seq_level] = ub[i_seq_j - 1, i_seq_level] + leftb
                end
            end
        end
        for j = 1:nc + 1
            jf = j * 2 - 1
            if j > 1
                push!(branch_stack, 0)
            else
                push!(branch_stack, 1)
            end
            if j <= nc
                push!(branch_stack, 0)
            else
                push!(branch_stack, 1)
            end
            clb = 0.5 * ub[jf, i_seq_level]
            crb = 0.5 * ub[jf, i_seq_level]
            branch = pop!(branch_stack)
            if branch == 0
                ub[j, i_seq_level + 1] = ub[j, i_seq_level + 1] + crb
            end
            branch = pop!(branch_stack)
            if branch == 0
                ub[j - 1, i_seq_level + 1] = ub[j - 1, i_seq_level + 1] + clb
            end
        end
        for j = 1:nc
            jf = j * 2
            ub[j, i_seq_level + 1] = ub[j, i_seq_level + 1] + ub[jf, i_seq_level]
        end
        hl2 = pop!(hl2_stack)
        hlb = hlb + 2 * hl * hl2b
        hl = pop!(hl_stack)
        hlb = hlb / 2.0
    end
    hl2 = hl * hl
    hl2b = f[1, num_levels] * 0.5 * ub[1, num_levels]
    fb[1, num_levels] = fb[1, num_levels] + hl2 * 0.5 * ub[1, num_levels]
    ub[1, num_levels] = 0.0
    hlb = hlb + 2 * hl * hl2b
    for i_seq_level = num_levels - 1:-1:1
        hl = pop!(hl_stack)
        hlb = 2.0hlb
        u .= pop!(u_stack)
        for j = 1:nc
            ub[j, i_seq_level + 1] = 0.0
        end
        f .= pop!(f_stack)
        for j = 1:nc
            jf = j * 2
            rb[jf - 1, i_seq_level] = rb[jf - 1, i_seq_level] + 0.25 * fb[j, i_seq_level + 1]
            rb[jf, i_seq_level] = rb[jf, i_seq_level] + 0.5 * fb[j, i_seq_level + 1]
            rb[jf + 1, i_seq_level] = rb[jf + 1, i_seq_level] + 0.25 * fb[j, i_seq_level + 1]
            fb[j, i_seq_level + 1] = 0.0
        end
        hl2 = hl * hl
        hl2b = 0.0
        u .= pop!(u_stack)
        n = pop!(n_stack)
        for j = 1:n
            left = 0.0
            if j > 1
                left = u[j - 1, i_seq_level]
                push!(branch_stack, 0)
            else
                push!(branch_stack, 1)
            end
            right = 0.0
            if j < n
                right = u[j + 1, i_seq_level]
                push!(branch_stack, 0)
            else
                push!(branch_stack, 1)
            end
            fb[j, i_seq_level] = fb[j, i_seq_level] + rb[j, i_seq_level]
            tempb = -(rb[j, i_seq_level] / hl2)
            rb[j, i_seq_level] = 0.0
            ub[j, i_seq_level] = ub[j, i_seq_level] + 2.0tempb
            leftb = -tempb
            rightb = -tempb
            hl2b = hl2b - (((2.0 * u[j, i_seq_level] - left) - right) * tempb) / hl2
            branch = pop!(branch_stack)
            if branch == 0
                ub[j + 1, i_seq_level] = ub[j + 1, i_seq_level] + rightb
            end
            branch = pop!(branch_stack)
            if branch == 0
                ub[j - 1, i_seq_level] = ub[j - 1, i_seq_level] + leftb
            end
        end
        for i_seq_k = nu1:-1:1
            ad_to = pop!(integer8_stack)
            for i_seq_j = ad_to:-1:1
                tempb = 0.5 * ub[i_seq_j, i_seq_level]
                ub[i_seq_j, i_seq_level] = 0.0
                hl2b = hl2b + f[i_seq_j, i_seq_level] * tempb
                fb[i_seq_j, i_seq_level] = fb[i_seq_j, i_seq_level] + hl2 * tempb
                leftb = tempb
                rightb = tempb
                branch = pop!(branch_stack)
                if branch == 0
                    ub[i_seq_j + 1, i_seq_level] = ub[i_seq_j + 1, i_seq_level] + rightb
                end
                branch = pop!(branch_stack)
                if branch == 0
                    ub[i_seq_j - 1, i_seq_level] = ub[i_seq_j - 1, i_seq_level] + leftb
                end
            end
        end
        hlb = hlb + 2 * hl * hl2b
    end
    h1b = h1b + hlb
    return h1b
end
function mg_vcycle(u, f, r, nfine, num_levels, h1, nu1, nu2, n)
    n = n * 2
    nl = nfine
    hl = h1
    for i_seq_level = 1:num_levels - 1
        n = nl - 1
        hl2 = hl * hl
        for i_seq_k = 1:nu1
            for i_seq_j = 1:n
                left = 0.0
                if i_seq_j > 1
                    left = u[i_seq_j - 1, i_seq_level]
                end
                right = 0.0
                if i_seq_j < n
                    right = u[i_seq_j + 1, i_seq_level]
                end
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
            end
        end
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
        for i_seq_k = 1:nu2
            for i_seq_j = 1:n
                left = 0.0
                if i_seq_j > 1
                    left = u[i_seq_j - 1, i_seq_level]
                end
                right = 0.0
                if i_seq_j < n
                    right = u[i_seq_j + 1, i_seq_level]
                end
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
            end
        end
    end
end