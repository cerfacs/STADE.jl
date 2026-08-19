function initstacks_cascadic_mg_prolong_b()
    hl_stack = Vector{Float64}()
    hl2_stack = Vector{Float64}()
    tripcount_stack = Vector{Int64}()
    left_stack = Vector{Float64}()
    branch_stack = Vector{Int64}()
    right_stack = Vector{Float64}()
    cl_stack = Vector{Float64}()
    cr_stack = Vector{Float64}()
    return (hl_stack, hl2_stack, tripcount_stack, left_stack, branch_stack, right_stack, cl_stack, cr_stack)
end

function cascadic_mg_prolong_b(u, ub, rhs, rhsb, h_coarse, h_coarseb, nu, num_levels, hl_stack, hl2_stack, tripcount_stack, left_stack, branch_stack, right_stack, cl_stack, cr_stack)
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
    push!(hl_stack, hl)
    hl = h_coarse
    nc = nl - 1
    push!(hl2_stack, hl2)
    hl2 = hl * hl
    for i_seq_k = 1:nu
        push!(tripcount_stack, nc)
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
        push!(hl_stack, hl)
        hl = hl / 2.0
        nc = nl - 1
        push!(hl2_stack, hl2)
        hl2 = hl * hl
        ncoarse = div(nl, 2) - 1
        push!(tripcount_stack, ncoarse)
        for j = 1:ncoarse
            jf = j * 2
            u[jf, i_seq_level] = u[j, i_seq_level - 1]
        end
        push!(tripcount_stack, ncoarse)
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
            push!(tripcount_stack, nc)
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
    push!(hl_stack, hl)
    push!(hl2_stack, hl2)
    push!(left_stack, left)
    push!(right_stack, right)
    cl = pop!(cl_stack)
    cr = pop!(cr_stack)
    hl = pop!(hl_stack)
    hl2 = pop!(hl2_stack)
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
            nc = pop!(tripcount_stack)
            for i_seq_j = nc:-1:1
                left = pop!(left_stack)
                right = pop!(right_stack)
                __branch_pre_4 = pop!(branch_stack)
                if __branch_pre_4 == 1
                    __snap_discard = pop!(right_stack)
                end
                right = 0.0
                if __branch_pre_4 == 1
                    right = u[i_seq_j + 1, i_seq_level]
                else
                    right = 0.0
                end
                __branch_pre_2 = pop!(branch_stack)
                if __branch_pre_2 == 1
                    __snap_discard = pop!(left_stack)
                end
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
        ncoarse = pop!(tripcount_stack)
        for j = ncoarse + 1:-1:1
            cl = pop!(cl_stack)
            cr = pop!(cr_stack)
            jf = j * 2 - 1
            __branch_pre_5 = pop!(branch_stack)
            if __branch_pre_5 == 1
                __snap_discard = pop!(cr_stack)
            end
            cr = 0.0
            if __branch_pre_5 == 1
                cr = u[j, i_seq_level - 1]
            else
                cr = 0.0
            end
            __branch_pre_3 = pop!(branch_stack)
            if __branch_pre_3 == 1
                __snap_discard = pop!(cl_stack)
            end
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
        ncoarse = pop!(tripcount_stack)
        for j = 1:ncoarse
            jf = j * 2
            ub[j, i_seq_level - 1] = ub[j, i_seq_level - 1] + ub[jf, i_seq_level]
            ub[jf, i_seq_level] = 0.0
        end
        hl2 = pop!(hl2_stack)
        hlb = hlb + hl * hl2b
        hlb = hlb + hl * hl2b
        hl2b = 0.0
        hl = pop!(hl_stack)
        hlb = 0.5hlb
    end
    for i_seq_k = nu:-1:1
        left = pop!(left_stack)
        right = pop!(right_stack)
        nc = pop!(tripcount_stack)
        for i_seq_j = nc:-1:1
            left = pop!(left_stack)
            right = pop!(right_stack)
            __branch_pre_4 = pop!(branch_stack)
            if __branch_pre_4 == 1
                __snap_discard = pop!(right_stack)
            end
            right = 0.0
            if __branch_pre_4 == 1
                right = u[i_seq_j + 1, 1]
            else
                right = 0.0
            end
            __branch_pre_2 = pop!(branch_stack)
            if __branch_pre_2 == 1
                __snap_discard = pop!(left_stack)
            end
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
    hl2 = pop!(hl2_stack)
    hlb = hlb + hl * hl2b
    hlb = hlb + hl * hl2b
    hl2b = 0.0
    hl = pop!(hl_stack)
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
