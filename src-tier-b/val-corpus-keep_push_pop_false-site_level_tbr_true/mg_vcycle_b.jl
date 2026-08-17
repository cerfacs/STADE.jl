function initstacks_mg_vcycle_b(nu1, nu2, num_levels)
    hl2_stack = Vector{Float64}(undef, (((div((num_levels - 1) - 1, 1) + 1) + 1) + (div(1 - (num_levels - 1), -1) + 1)) + 1)
    tripcount_stack = Vector{Int64}(undef, ((((((div((num_levels - 1) - 1, 1) + 1) * (div(nu1 - 1, 1) + 1) + (div((num_levels - 1) - 1, 1) + 1)) + (div((num_levels - 1) - 1, 1) + 1)) + (div((num_levels - 1) - 1, 1) + 1)) + (div(1 - (num_levels - 1), -1) + 1)) + (div(1 - (num_levels - 1), -1) + 1)) + (div(1 - (num_levels - 1), -1) + 1) * (div(nu2 - 1, 1) + 1))
    left_stack = Vector{Float64}()
    branch_stack = Vector{Int64}()
    right_stack = Vector{Float64}()
    u_stack = Vector{Float64}()
    r_stack = Vector{Float64}()
    f_stack = Vector{Float64}()
    hl_stack = Vector{Float64}(undef, ((div((num_levels - 1) - 1, 1) + 1) + (div(1 - (num_levels - 1), -1) + 1)) + 1)
    cl_stack = Vector{Float64}()
    cr_stack = Vector{Float64}()
    return (hl2_stack, tripcount_stack, left_stack, branch_stack, right_stack, u_stack, r_stack, f_stack, hl_stack, cl_stack, cr_stack)
end

function mg_vcycle_b(u, ub, f, fb, r, rb, nfine, num_levels, h1, h1b, nu1, nu2, n, hl2_stack, tripcount_stack, left_stack, branch_stack, right_stack, u_stack, r_stack, f_stack, hl_stack, cl_stack, cr_stack)
    cl = 0.0
    cr = 0.0
    hl = 0.0
    hl2 = 0.0
    left = 0.0
    right = 0.0
    clb = 0.0
    crb = 0.0
    hlb = 0.0
    hl2b = 0.0
    leftb = 0.0
    rightb = 0.0
    n = n * 2
    nl = nfine
    hl = h1
    for i_seq_level = 1:num_levels - 1
        n = nl - 1
        hl2_stack[(i_seq_level - 1) + 1] = hl2
        hl2 = hl * hl
        for i_seq_k = 1:nu1
            tripcount_stack[((i_seq_level - 1) * (div(nu1 - 1, 1) + 1) + (i_seq_k - 1)) + 1] = n
            for i_seq_j = 1:n
                push!(left_stack, left)
                left = 0.0
                if i_seq_j > 1
                    push!(branch_stack, 1)
                    push!(left_stack, left)
                    left = u[i_seq_j - 1, i_seq_level]
                else
                    push!(branch_stack, 0)
                end
                push!(right_stack, right)
                right = 0.0
                if i_seq_j < n
                    push!(branch_stack, 1)
                    push!(right_stack, right)
                    right = u[i_seq_j + 1, i_seq_level]
                else
                    push!(branch_stack, 0)
                end
                push!(u_stack, u[i_seq_j, i_seq_level])
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
                push!(left_stack, left)
                push!(right_stack, right)
            end
            push!(left_stack, left)
            push!(right_stack, right)
        end
        tripcount_stack[(div((num_levels - 1) - 1, 1) + 1) * (div(nu1 - 1, 1) + 1) + ((i_seq_level - 1) + 1)] = n
        for j = 1:n
            push!(left_stack, left)
            left = 0.0
            if j > 1
                push!(branch_stack, 1)
                push!(left_stack, left)
                left = u[j - 1, i_seq_level]
            else
                push!(branch_stack, 0)
            end
            push!(right_stack, right)
            right = 0.0
            if j < n
                push!(branch_stack, 1)
                push!(right_stack, right)
                right = u[j + 1, i_seq_level]
            else
                push!(branch_stack, 0)
            end
            push!(r_stack, r[j, i_seq_level])
            r[j, i_seq_level] = f[j, i_seq_level] - ((2.0 * u[j, i_seq_level] - left) - right) / hl2
            push!(left_stack, left)
            push!(right_stack, right)
        end
        ncg = div(nl, 2)
        nc = ncg - 1
        tripcount_stack[((div((num_levels - 1) - 1, 1) + 1) * (div(nu1 - 1, 1) + 1) + (div((num_levels - 1) - 1, 1) + 1)) + ((i_seq_level - 1) + 1)] = nc
        for j = 1:nc
            jf = j * 2
            push!(f_stack, f[j, i_seq_level + 1])
            f[j, i_seq_level + 1] = 0.25 * r[jf - 1, i_seq_level] + 0.5 * r[jf, i_seq_level] + 0.25 * r[jf + 1, i_seq_level]
        end
        tripcount_stack[(((div((num_levels - 1) - 1, 1) + 1) * (div(nu1 - 1, 1) + 1) + (div((num_levels - 1) - 1, 1) + 1)) + (div((num_levels - 1) - 1, 1) + 1)) + ((i_seq_level - 1) + 1)] = nc
        for j = 1:nc
            push!(u_stack, u[j, i_seq_level + 1])
            u[j, i_seq_level + 1] = 0.0
        end
        nl = ncg
        hl_stack[(i_seq_level - 1) + 1] = hl
        hl = hl * 2.0
        push!(left_stack, left)
        push!(right_stack, right)
    end
    hl2_stack[(div((num_levels - 1) - 1, 1) + 1) + 1] = hl2
    hl2 = hl * hl
    push!(u_stack, u[1, num_levels])
    u[1, num_levels] = 0.5 * hl2 * f[1, num_levels]
    for i_seq_level = num_levels - 1:-1:1
        nl = nl * 2
        hl_stack[(div((num_levels - 1) - 1, 1) + 1) + (div(i_seq_level - (num_levels - 1), -1) + 1)] = hl
        hl = hl / 2.0
        n = nl - 1
        ncg = div(nl, 2)
        nc = ncg - 1
        hl2_stack[((div((num_levels - 1) - 1, 1) + 1) + 1) + (div(i_seq_level - (num_levels - 1), -1) + 1)] = hl2
        hl2 = hl * hl
        tripcount_stack[((((div((num_levels - 1) - 1, 1) + 1) * (div(nu1 - 1, 1) + 1) + (div((num_levels - 1) - 1, 1) + 1)) + (div((num_levels - 1) - 1, 1) + 1)) + (div((num_levels - 1) - 1, 1) + 1)) + (div(i_seq_level - (num_levels - 1), -1) + 1)] = nc
        for j = 1:nc
            jf = j * 2
            push!(u_stack, u[jf, i_seq_level])
            u[jf, i_seq_level] = u[jf, i_seq_level] + u[j, i_seq_level + 1]
        end
        tripcount_stack[(((((div((num_levels - 1) - 1, 1) + 1) * (div(nu1 - 1, 1) + 1) + (div((num_levels - 1) - 1, 1) + 1)) + (div((num_levels - 1) - 1, 1) + 1)) + (div((num_levels - 1) - 1, 1) + 1)) + (div(1 - (num_levels - 1), -1) + 1)) + (div(i_seq_level - (num_levels - 1), -1) + 1)] = nc
        for j = 1:nc + 1
            jf = j * 2 - 1
            push!(cl_stack, cl)
            cl = 0.0
            if j > 1
                push!(branch_stack, 1)
                push!(cl_stack, cl)
                cl = u[j - 1, i_seq_level + 1]
            else
                push!(branch_stack, 0)
            end
            push!(cr_stack, cr)
            cr = 0.0
            if j <= nc
                push!(branch_stack, 1)
                push!(cr_stack, cr)
                cr = u[j, i_seq_level + 1]
            else
                push!(branch_stack, 0)
            end
            push!(u_stack, u[jf, i_seq_level])
            u[jf, i_seq_level] = u[jf, i_seq_level] + 0.5 * (cl + cr)
            push!(cl_stack, cl)
            push!(cr_stack, cr)
        end
        for i_seq_k = 1:nu2
            tripcount_stack[((((((div((num_levels - 1) - 1, 1) + 1) * (div(nu1 - 1, 1) + 1) + (div((num_levels - 1) - 1, 1) + 1)) + (div((num_levels - 1) - 1, 1) + 1)) + (div((num_levels - 1) - 1, 1) + 1)) + (div(1 - (num_levels - 1), -1) + 1)) + (div(1 - (num_levels - 1), -1) + 1)) + ((div(i_seq_level - (num_levels - 1), -1) * (div(nu2 - 1, 1) + 1) + (i_seq_k - 1)) + 1)] = n
            for i_seq_j = 1:n
                push!(left_stack, left)
                left = 0.0
                if i_seq_j > 1
                    push!(branch_stack, 1)
                    push!(left_stack, left)
                    left = u[i_seq_j - 1, i_seq_level]
                else
                    push!(branch_stack, 0)
                end
                push!(right_stack, right)
                right = 0.0
                if i_seq_j < n
                    push!(branch_stack, 1)
                    push!(right_stack, right)
                    right = u[i_seq_j + 1, i_seq_level]
                else
                    push!(branch_stack, 0)
                end
                push!(u_stack, u[i_seq_j, i_seq_level])
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
                push!(left_stack, left)
                push!(right_stack, right)
            end
            push!(left_stack, left)
            push!(right_stack, right)
        end
        push!(cl_stack, cl)
        push!(cr_stack, cr)
        push!(left_stack, left)
        push!(right_stack, right)
    end
    push!(cl_stack, cl)
    push!(cr_stack, cr)
    hl_stack[((div((num_levels - 1) - 1, 1) + 1) + (div(1 - (num_levels - 1), -1) + 1)) + 1] = hl
    hl2_stack[(((div((num_levels - 1) - 1, 1) + 1) + 1) + (div(1 - (num_levels - 1), -1) + 1)) + 1] = hl2
    push!(left_stack, left)
    push!(right_stack, right)
    cl = pop!(cl_stack)
    cr = pop!(cr_stack)
    hl = hl_stack[((div((num_levels - 1) - 1, 1) + 1) + (div(1 - (num_levels - 1), -1) + 1)) + 1]
    hl2 = hl2_stack[(((div((num_levels - 1) - 1, 1) + 1) + 1) + (div(1 - (num_levels - 1), -1) + 1)) + 1]
    left = pop!(left_stack)
    right = pop!(right_stack)
    for i_seq_level = 1:num_levels - 1
        cl = pop!(cl_stack)
        cr = pop!(cr_stack)
        left = pop!(left_stack)
        right = pop!(right_stack)
        for i_seq_k = nu2:-1:1
            left = pop!(left_stack)
            right = pop!(right_stack)
            n = tripcount_stack[((((((div((num_levels - 1) - 1, 1) + 1) * (div(nu1 - 1, 1) + 1) + (div((num_levels - 1) - 1, 1) + 1)) + (div((num_levels - 1) - 1, 1) + 1)) + (div((num_levels - 1) - 1, 1) + 1)) + (div(1 - (num_levels - 1), -1) + 1)) + (div(1 - (num_levels - 1), -1) + 1)) + ((div(i_seq_level - (num_levels - 1), -1) * (div(nu2 - 1, 1) + 1) + (i_seq_k - 1)) + 1)]
            for i_seq_j = n:-1:1
                left = pop!(left_stack)
                right = pop!(right_stack)
                __branch_pre_4 = pop!(branch_stack)
                right = 0.0
                if __branch_pre_4 == 1
                    right = u[i_seq_j + 1, i_seq_level]
                else
                    right = 0.0
                end
                __branch_pre_2 = pop!(branch_stack)
                left = 0.0
                if __branch_pre_2 == 1
                    left = u[i_seq_j - 1, i_seq_level]
                else
                    left = 0.0
                end
                u[i_seq_j, i_seq_level] = pop!(u_stack)
                hl2b = hl2b + f[i_seq_j, i_seq_level] * (0.5 * ub[i_seq_j, i_seq_level])
                fb[i_seq_j, i_seq_level] = fb[i_seq_j, i_seq_level] + hl2 * (0.5 * ub[i_seq_j, i_seq_level])
                leftb = leftb + 0.5 * ub[i_seq_j, i_seq_level]
                rightb = rightb + 0.5 * ub[i_seq_j, i_seq_level]
                ub[i_seq_j, i_seq_level] = 0.0
                if __branch_pre_4 == 1
                    ub[i_seq_j + 1, i_seq_level] = ub[i_seq_j + 1, i_seq_level] + rightb
                    rightb = 0.0
                end
                right = pop!(right_stack)
                rightb = 0.0
                if __branch_pre_2 == 1
                    ub[i_seq_j - 1, i_seq_level] = ub[i_seq_j - 1, i_seq_level] + leftb
                    leftb = 0.0
                end
                left = pop!(left_stack)
                leftb = 0.0
            end
        end
        nc = tripcount_stack[(((((div((num_levels - 1) - 1, 1) + 1) * (div(nu1 - 1, 1) + 1) + (div((num_levels - 1) - 1, 1) + 1)) + (div((num_levels - 1) - 1, 1) + 1)) + (div((num_levels - 1) - 1, 1) + 1)) + (div(1 - (num_levels - 1), -1) + 1)) + (div(i_seq_level - (num_levels - 1), -1) + 1)]
        for j = nc + 1:-1:1
            cl = pop!(cl_stack)
            cr = pop!(cr_stack)
            jf = j * 2 - 1
            __branch_pre_5 = pop!(branch_stack)
            cr = 0.0
            if __branch_pre_5 == 1
                cr = u[j, i_seq_level + 1]
            else
                cr = 0.0
            end
            __branch_pre_3 = pop!(branch_stack)
            cl = 0.0
            if __branch_pre_3 == 1
                cl = u[j - 1, i_seq_level + 1]
            else
                cl = 0.0
            end
            u[jf, i_seq_level] = pop!(u_stack)
            clb = clb + 0.5 * ub[jf, i_seq_level]
            crb = crb + 0.5 * ub[jf, i_seq_level]
            if __branch_pre_5 == 1
                ub[j, i_seq_level + 1] = ub[j, i_seq_level + 1] + crb
                crb = 0.0
            end
            cr = pop!(cr_stack)
            crb = 0.0
            if __branch_pre_3 == 1
                ub[j - 1, i_seq_level + 1] = ub[j - 1, i_seq_level + 1] + clb
                clb = 0.0
            end
            cl = pop!(cl_stack)
            clb = 0.0
        end
        nc = tripcount_stack[((((div((num_levels - 1) - 1, 1) + 1) * (div(nu1 - 1, 1) + 1) + (div((num_levels - 1) - 1, 1) + 1)) + (div((num_levels - 1) - 1, 1) + 1)) + (div((num_levels - 1) - 1, 1) + 1)) + (div(i_seq_level - (num_levels - 1), -1) + 1)]
        for j = nc:-1:1
            jf = j * 2
            u[jf, i_seq_level] = pop!(u_stack)
            ub[j, i_seq_level + 1] = ub[j, i_seq_level + 1] + ub[jf, i_seq_level]
        end
        hl2 = hl2_stack[((div((num_levels - 1) - 1, 1) + 1) + 1) + (div(i_seq_level - (num_levels - 1), -1) + 1)]
        hlb = hlb + hl * hl2b
        hlb = hlb + hl * hl2b
        hl2b = 0.0
        hl = hl_stack[(div((num_levels - 1) - 1, 1) + 1) + (div(i_seq_level - (num_levels - 1), -1) + 1)]
        hlb = 0.5hlb
    end
    u[1, num_levels] = pop!(u_stack)
    hl2b = hl2b + (0.5 * f[1, num_levels]) * ub[1, num_levels]
    fb[1, num_levels] = fb[1, num_levels] + (0.5hl2) * ub[1, num_levels]
    ub[1, num_levels] = 0.0
    hl2 = hl2_stack[(div((num_levels - 1) - 1, 1) + 1) + 1]
    hlb = hlb + hl * hl2b
    hlb = hlb + hl * hl2b
    hl2b = 0.0
    for i_seq_level = num_levels - 1:-1:1
        left = pop!(left_stack)
        right = pop!(right_stack)
        hl = hl_stack[(i_seq_level - 1) + 1]
        hlb = 2.0hlb
        nc = tripcount_stack[(((div((num_levels - 1) - 1, 1) + 1) * (div(nu1 - 1, 1) + 1) + (div((num_levels - 1) - 1, 1) + 1)) + (div((num_levels - 1) - 1, 1) + 1)) + ((i_seq_level - 1) + 1)]
        for j = nc:-1:1
            u[j, i_seq_level + 1] = pop!(u_stack)
            ub[j, i_seq_level + 1] = 0.0
        end
        nc = tripcount_stack[((div((num_levels - 1) - 1, 1) + 1) * (div(nu1 - 1, 1) + 1) + (div((num_levels - 1) - 1, 1) + 1)) + ((i_seq_level - 1) + 1)]
        for j = nc:-1:1
            jf = j * 2
            f[j, i_seq_level + 1] = pop!(f_stack)
            rb[jf - 1, i_seq_level] = rb[jf - 1, i_seq_level] + 0.25 * fb[j, i_seq_level + 1]
            rb[jf, i_seq_level] = rb[jf, i_seq_level] + 0.5 * fb[j, i_seq_level + 1]
            rb[jf + 1, i_seq_level] = rb[jf + 1, i_seq_level] + 0.25 * fb[j, i_seq_level + 1]
            fb[j, i_seq_level + 1] = 0.0
        end
        n = tripcount_stack[(div((num_levels - 1) - 1, 1) + 1) * (div(nu1 - 1, 1) + 1) + ((i_seq_level - 1) + 1)]
        for j = n:-1:1
            left = pop!(left_stack)
            right = pop!(right_stack)
            __branch_pre_4 = pop!(branch_stack)
            right = 0.0
            if __branch_pre_4 == 1
                right = u[j + 1, i_seq_level]
            else
                right = 0.0
            end
            __branch_pre_2 = pop!(branch_stack)
            left = 0.0
            if __branch_pre_2 == 1
                left = u[j - 1, i_seq_level]
            else
                left = 0.0
            end
            r[j, i_seq_level] = pop!(r_stack)
            fb[j, i_seq_level] = fb[j, i_seq_level] + rb[j, i_seq_level]
            ub[j, i_seq_level] = ub[j, i_seq_level] + 2.0 * ((1.0 / hl2) * -(rb[j, i_seq_level]))
            leftb = leftb + -((1.0 / hl2) * -(rb[j, i_seq_level]))
            rightb = rightb + -((1.0 / hl2) * -(rb[j, i_seq_level]))
            hl2b = hl2b + -(((2.0 * u[j, i_seq_level] - left) - right) / hl2 ^ 2) * -(rb[j, i_seq_level])
            rb[j, i_seq_level] = 0.0
            if __branch_pre_4 == 1
                ub[j + 1, i_seq_level] = ub[j + 1, i_seq_level] + rightb
                rightb = 0.0
            end
            right = pop!(right_stack)
            rightb = 0.0
            if __branch_pre_2 == 1
                ub[j - 1, i_seq_level] = ub[j - 1, i_seq_level] + leftb
                leftb = 0.0
            end
            left = pop!(left_stack)
            leftb = 0.0
        end
        for i_seq_k = nu1:-1:1
            left = pop!(left_stack)
            right = pop!(right_stack)
            n = tripcount_stack[((i_seq_level - 1) * (div(nu1 - 1, 1) + 1) + (i_seq_k - 1)) + 1]
            for i_seq_j = n:-1:1
                left = pop!(left_stack)
                right = pop!(right_stack)
                __branch_pre_4 = pop!(branch_stack)
                right = 0.0
                if __branch_pre_4 == 1
                    right = u[i_seq_j + 1, i_seq_level]
                else
                    right = 0.0
                end
                __branch_pre_2 = pop!(branch_stack)
                left = 0.0
                if __branch_pre_2 == 1
                    left = u[i_seq_j - 1, i_seq_level]
                else
                    left = 0.0
                end
                u[i_seq_j, i_seq_level] = pop!(u_stack)
                hl2b = hl2b + f[i_seq_j, i_seq_level] * (0.5 * ub[i_seq_j, i_seq_level])
                fb[i_seq_j, i_seq_level] = fb[i_seq_j, i_seq_level] + hl2 * (0.5 * ub[i_seq_j, i_seq_level])
                leftb = leftb + 0.5 * ub[i_seq_j, i_seq_level]
                rightb = rightb + 0.5 * ub[i_seq_j, i_seq_level]
                ub[i_seq_j, i_seq_level] = 0.0
                if __branch_pre_4 == 1
                    ub[i_seq_j + 1, i_seq_level] = ub[i_seq_j + 1, i_seq_level] + rightb
                    rightb = 0.0
                end
                right = pop!(right_stack)
                rightb = 0.0
                if __branch_pre_2 == 1
                    ub[i_seq_j - 1, i_seq_level] = ub[i_seq_j - 1, i_seq_level] + leftb
                    leftb = 0.0
                end
                left = pop!(left_stack)
                leftb = 0.0
            end
        end
        hl2 = hl2_stack[(i_seq_level - 1) + 1]
        hlb = hlb + hl * hl2b
        hlb = hlb + hl * hl2b
        hl2b = 0.0
    end
    h1b = h1b + hlb
    hlb = 0.0
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
