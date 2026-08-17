function initstacks_cascadic_mg_prolong_b(nu, num_levels)
    tripcount_stack = Vector{Int64}(undef, (((div(nu - 1, 1) + 1) + (div(num_levels - 2, 1) + 1)) + (div(num_levels - 2, 1) + 1)) + (div(num_levels - 2, 1) + 1) * (div(nu - 1, 1) + 1))
    left_stack = Vector{Float64}()
    branch_stack = Vector{Int64}()
    right_stack = Vector{Float64}()
    hl_stack = Vector{Float64}(undef, (div(num_levels - 2, 1) + 1) + 1)
    hl2_stack = Vector{Float64}(undef, (div(num_levels - 2, 1) + 1) + 1)
    cl_stack = Vector{Float64}()
    cr_stack = Vector{Float64}()
    return (tripcount_stack, left_stack, branch_stack, right_stack, hl_stack, hl2_stack, cl_stack, cr_stack)
end

function cascadic_mg_prolong_b(u, ub, rhs, rhsb, h_coarse, h_coarseb, nu, num_levels, tripcount_stack, left_stack, branch_stack, right_stack, hl_stack, hl2_stack, cl_stack, cr_stack)
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
    nl = 2
    hl = h_coarse
    nc = nl - 1
    hl2 = hl * hl
    for i_seq_k = 1:nu
        tripcount_stack[(i_seq_k - 1) + 1] = nc
        for i_seq_j = 1:nc
            push!(left_stack, left)
            left = 0.0
            if i_seq_j > 1
                push!(branch_stack, 1)
                push!(left_stack, left)
                left = u[i_seq_j - 1, 1]
            else
                push!(branch_stack, 0)
            end
            push!(right_stack, right)
            right = 0.0
            if i_seq_j < nc
                push!(branch_stack, 1)
                push!(right_stack, right)
                right = u[i_seq_j + 1, 1]
            else
                push!(branch_stack, 0)
            end
            u[i_seq_j, 1] = 0.5 * (hl2 * rhs[i_seq_j, 1] + left + right)
            push!(left_stack, left)
            push!(right_stack, right)
        end
        push!(left_stack, left)
        push!(right_stack, right)
    end
    for i_seq_level = 2:num_levels
        nl = nl * 2
        hl_stack[(i_seq_level - 2) + 1] = hl
        hl = hl / 2.0
        nc = nl - 1
        hl2_stack[(i_seq_level - 2) + 1] = hl2
        hl2 = hl * hl
        ncoarse = div(nl, 2) - 1
        tripcount_stack[(div(nu - 1, 1) + 1) + ((i_seq_level - 2) + 1)] = ncoarse
        for j = 1:ncoarse
            jf = j * 2
            u[jf, i_seq_level] = u[j, i_seq_level - 1]
        end
        tripcount_stack[((div(nu - 1, 1) + 1) + (div(num_levels - 2, 1) + 1)) + ((i_seq_level - 2) + 1)] = ncoarse
        for j = 1:ncoarse + 1
            jf = j * 2 - 1
            push!(cl_stack, cl)
            cl = 0.0
            if j > 1
                push!(branch_stack, 1)
                push!(cl_stack, cl)
                cl = u[j - 1, i_seq_level - 1]
            else
                push!(branch_stack, 0)
            end
            push!(cr_stack, cr)
            cr = 0.0
            if j <= ncoarse
                push!(branch_stack, 1)
                push!(cr_stack, cr)
                cr = u[j, i_seq_level - 1]
            else
                push!(branch_stack, 0)
            end
            u[jf, i_seq_level] = 0.5 * (cl + cr)
            push!(cl_stack, cl)
            push!(cr_stack, cr)
        end
        for i_seq_k = 1:nu
            tripcount_stack[(((div(nu - 1, 1) + 1) + (div(num_levels - 2, 1) + 1)) + (div(num_levels - 2, 1) + 1)) + (((i_seq_level - 2) * (div(nu - 1, 1) + 1) + (i_seq_k - 1)) + 1)] = nc
            for i_seq_j = 1:nc
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
                if i_seq_j < nc
                    push!(branch_stack, 1)
                    push!(right_stack, right)
                    right = u[i_seq_j + 1, i_seq_level]
                else
                    push!(branch_stack, 0)
                end
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * rhs[i_seq_j, i_seq_level] + left + right)
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
    hl_stack[(div(num_levels - 2, 1) + 1) + 1] = hl
    hl2_stack[(div(num_levels - 2, 1) + 1) + 1] = hl2
    push!(left_stack, left)
    push!(right_stack, right)
    cl = pop!(cl_stack)
    cr = pop!(cr_stack)
    hl = hl_stack[(div(num_levels - 2, 1) + 1) + 1]
    hl2 = hl2_stack[(div(num_levels - 2, 1) + 1) + 1]
    left = pop!(left_stack)
    right = pop!(right_stack)
    for i_seq_level = num_levels:-1:2
        cl = pop!(cl_stack)
        cr = pop!(cr_stack)
        left = pop!(left_stack)
        right = pop!(right_stack)
        for i_seq_k = nu:-1:1
            left = pop!(left_stack)
            right = pop!(right_stack)
            nc = tripcount_stack[(((div(nu - 1, 1) + 1) + (div(num_levels - 2, 1) + 1)) + (div(num_levels - 2, 1) + 1)) + (((i_seq_level - 2) * (div(nu - 1, 1) + 1) + (i_seq_k - 1)) + 1)]
            for i_seq_j = nc:-1:1
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
                hl2b = hl2b + rhs[i_seq_j, i_seq_level] * (0.5 * ub[i_seq_j, i_seq_level])
                rhsb[i_seq_j, i_seq_level] = rhsb[i_seq_j, i_seq_level] + hl2 * (0.5 * ub[i_seq_j, i_seq_level])
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
        ncoarse = tripcount_stack[((div(nu - 1, 1) + 1) + (div(num_levels - 2, 1) + 1)) + ((i_seq_level - 2) + 1)]
        for j = ncoarse + 1:-1:1
            cl = pop!(cl_stack)
            cr = pop!(cr_stack)
            jf = j * 2 - 1
            __branch_pre_5 = pop!(branch_stack)
            cr = 0.0
            if __branch_pre_5 == 1
                cr = u[j, i_seq_level - 1]
            else
                cr = 0.0
            end
            __branch_pre_3 = pop!(branch_stack)
            cl = 0.0
            if __branch_pre_3 == 1
                cl = u[j - 1, i_seq_level - 1]
            else
                cl = 0.0
            end
            clb = clb + 0.5 * ub[jf, i_seq_level]
            crb = crb + 0.5 * ub[jf, i_seq_level]
            ub[jf, i_seq_level] = 0.0
            if __branch_pre_5 == 1
                ub[j, i_seq_level - 1] = ub[j, i_seq_level - 1] + crb
                crb = 0.0
            end
            cr = pop!(cr_stack)
            crb = 0.0
            if __branch_pre_3 == 1
                ub[j - 1, i_seq_level - 1] = ub[j - 1, i_seq_level - 1] + clb
                clb = 0.0
            end
            cl = pop!(cl_stack)
            clb = 0.0
        end
        ncoarse = tripcount_stack[(div(nu - 1, 1) + 1) + ((i_seq_level - 2) + 1)]
        for j = 1:ncoarse
            jf = j * 2
            ub[j, i_seq_level - 1] = ub[j, i_seq_level - 1] + ub[jf, i_seq_level]
            ub[jf, i_seq_level] = 0.0
        end
        hl2 = hl2_stack[(i_seq_level - 2) + 1]
        hlb = hlb + hl * hl2b
        hlb = hlb + hl * hl2b
        hl2b = 0.0
        hl = hl_stack[(i_seq_level - 2) + 1]
        hlb = 0.5hlb
    end
    for i_seq_k = nu:-1:1
        left = pop!(left_stack)
        right = pop!(right_stack)
        nc = tripcount_stack[(i_seq_k - 1) + 1]
        for i_seq_j = nc:-1:1
            left = pop!(left_stack)
            right = pop!(right_stack)
            __branch_pre_4 = pop!(branch_stack)
            right = 0.0
            if __branch_pre_4 == 1
                right = u[i_seq_j + 1, 1]
            else
                right = 0.0
            end
            __branch_pre_2 = pop!(branch_stack)
            left = 0.0
            if __branch_pre_2 == 1
                left = u[i_seq_j - 1, 1]
            else
                left = 0.0
            end
            hl2b = hl2b + rhs[i_seq_j, 1] * (0.5 * ub[i_seq_j, 1])
            rhsb[i_seq_j, 1] = rhsb[i_seq_j, 1] + hl2 * (0.5 * ub[i_seq_j, 1])
            leftb = leftb + 0.5 * ub[i_seq_j, 1]
            rightb = rightb + 0.5 * ub[i_seq_j, 1]
            ub[i_seq_j, 1] = 0.0
            if __branch_pre_4 == 1
                ub[i_seq_j + 1, 1] = ub[i_seq_j + 1, 1] + rightb
                rightb = 0.0
            end
            right = pop!(right_stack)
            rightb = 0.0
            if __branch_pre_2 == 1
                ub[i_seq_j - 1, 1] = ub[i_seq_j - 1, 1] + leftb
                leftb = 0.0
            end
            left = pop!(left_stack)
            leftb = 0.0
        end
    end
    hlb = hlb + hl * hl2b
    hlb = hlb + hl * hl2b
    hl2b = 0.0
    h_coarseb = h_coarseb + hlb
    hlb = 0.0
    return h_coarseb
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
