function initstacks_mg_relax_b()
    left_stack = Vector{Float64}()
    branch_stack = Vector{Int64}()
    right_stack = Vector{Float64}()
    return (left_stack, branch_stack, right_stack)
end

function mg_relax_hv(u, ub, f, fb, n, hl2, hl2b, lvl, nu, ud, ubd, fd, fbd, hl2d, hl2bd, left_stack, branch_stack, right_stack)
    left_stack_d = Vector{Float64}()
    right_stack_d = Vector{Float64}()
    left = 0.0
    right = 0.0
    leftb = 0.0
    rightb = 0.0
    leftd = 0.0
    leftbd = 0.0
    rightd = 0.0
    rightbd = 0.0
    for i_seq_k = 1:nu
        for i_seq_j = 1:n
            leftd = 0.0
            left = 0.0
            if i_seq_j > 1
                push!(branch_stack, 1)
                push!(left_stack_d, leftd)
                push!(left_stack, left)
                leftd = ud[i_seq_j - 1, lvl]
                left = u[i_seq_j - 1, lvl]
            else
                push!(branch_stack, 0)
            end
            rightd = 0.0
            right = 0.0
            if i_seq_j < n
                push!(branch_stack, 1)
                push!(right_stack_d, rightd)
                push!(right_stack, right)
                rightd = ud[i_seq_j + 1, lvl]
                right = u[i_seq_j + 1, lvl]
            else
                push!(branch_stack, 0)
            end
            ud[i_seq_j, lvl] = 0.5 * (((f[i_seq_j, lvl] * hl2d + hl2 * fd[i_seq_j, lvl]) + leftd) + rightd)
            u[i_seq_j, lvl] = 0.5 * (hl2 * f[i_seq_j, lvl] + left + right)
        end
    end
    for i_seq_k = nu:-1:1
        for i_seq_j = n:-1:1
            __branch_pre_4 = pop!(branch_stack)
            if __branch_pre_4 == 1
                __snap_discard = pop!(right_stack)
            end
            rightd = 0.0
            right = 0.0
            if __branch_pre_4 == 1
                rightd = ud[i_seq_j + 1, lvl]
                right = u[i_seq_j + 1, lvl]
            else
                rightd = 0.0
                right = 0.0
            end
            __branch_pre_2 = pop!(branch_stack)
            if __branch_pre_2 == 1
                __snap_discard = pop!(left_stack)
            end
            leftd = 0.0
            left = 0.0
            if __branch_pre_2 == 1
                leftd = ud[i_seq_j - 1, lvl]
                left = u[i_seq_j - 1, lvl]
            else
                leftd = 0.0
                left = 0.0
            end
            hl2bd = hl2bd + ((0.5 * ub[i_seq_j, lvl]) * fd[i_seq_j, lvl] + f[i_seq_j, lvl] * (0.5 * ubd[i_seq_j, lvl]))
            hl2b = hl2b + f[i_seq_j, lvl] * (0.5 * ub[i_seq_j, lvl])
            fbd[i_seq_j, lvl] = fbd[i_seq_j, lvl] + ((0.5 * ub[i_seq_j, lvl]) * hl2d + hl2 * (0.5 * ubd[i_seq_j, lvl]))
            fb[i_seq_j, lvl] = fb[i_seq_j, lvl] + hl2 * (0.5 * ub[i_seq_j, lvl])
            leftbd = leftbd + 0.5 * ubd[i_seq_j, lvl]
            leftb = leftb + 0.5 * ub[i_seq_j, lvl]
            rightbd = rightbd + 0.5 * ubd[i_seq_j, lvl]
            rightb = rightb + 0.5 * ub[i_seq_j, lvl]
            ubd[i_seq_j, lvl] = 0.0
            ub[i_seq_j, lvl] = 0.0
            if __branch_pre_4 == 1
                ubd[i_seq_j + 1, lvl] = ubd[i_seq_j + 1, lvl] + rightbd
                ub[i_seq_j + 1, lvl] = ub[i_seq_j + 1, lvl] + rightb
                rightbd = 0.0
                rightb = 0.0
            end
            rightbd = 0.0
            rightb = 0.0
            if __branch_pre_2 == 1
                ubd[i_seq_j - 1, lvl] = ubd[i_seq_j - 1, lvl] + leftbd
                ub[i_seq_j - 1, lvl] = ub[i_seq_j - 1, lvl] + leftb
                leftbd = 0.0
                leftb = 0.0
            end
            leftbd = 0.0
            leftb = 0.0
        end
    end
    return (hl2b, hl2bd)
end

function mg_relax(u, f, n, hl2, lvl, nu)
    #= none:1 =#
    #= none:2 =#
    for i_seq_k = 1:nu
        #= none:3 =#
        for i_seq_j = 1:n
            #= none:4 =#
            left = 0.0
            #= none:5 =#
            if i_seq_j > 1
                #= none:6 =#
                left = u[i_seq_j - 1, lvl]
            end
            #= none:8 =#
            right = 0.0
            #= none:9 =#
            if i_seq_j < n
                #= none:10 =#
                right = u[i_seq_j + 1, lvl]
            end
            #= none:12 =#
            u[i_seq_j, lvl] = 0.5 * (hl2 * f[i_seq_j, lvl] + left + right)
            #= none:13 =#
        end
        #= none:14 =#
    end
end

function initstacks_mg_vcycle_multi_b()
    hl_stack = Vector{Float64}()
    hl2_stack = Vector{Float64}()
    tripcount_stack = Vector{Int64}()
    left_mg_relax_c1_stack = Vector{Float64}()
    branch_stack = Vector{Int64}()
    right_mg_relax_c1_stack = Vector{Float64}()
    u_stack = Vector{Float64}()
    left_stack = Vector{Float64}()
    right_stack = Vector{Float64}()
    r_stack = Vector{Float64}()
    f_stack = Vector{Float64}()
    cl_stack = Vector{Float64}()
    cr_stack = Vector{Float64}()
    left_mg_relax_c2_stack = Vector{Float64}()
    right_mg_relax_c2_stack = Vector{Float64}()
    return (hl_stack, hl2_stack, tripcount_stack, left_mg_relax_c1_stack, branch_stack, right_mg_relax_c1_stack, u_stack, left_stack, right_stack, r_stack, f_stack, cl_stack, cr_stack, left_mg_relax_c2_stack, right_mg_relax_c2_stack)
end

function mg_vcycle_multi_hv(u, ub, f, fb, r, rb, nfine, num_levels, h1, h1b, nu1, nu2, n, ud, ubd, fd, fbd, rd, rbd, h1d, h1bd, hl_stack, hl2_stack, tripcount_stack, left_mg_relax_c1_stack, branch_stack, right_mg_relax_c1_stack, u_stack, left_stack, right_stack, r_stack, f_stack, cl_stack, cr_stack, left_mg_relax_c2_stack, right_mg_relax_c2_stack)
    hl_stack_d = Vector{Float64}()
    hl2_stack_d = Vector{Float64}()
    left_mg_relax_c1_stack_d = Vector{Float64}()
    right_mg_relax_c1_stack_d = Vector{Float64}()
    u_stack_d = Vector{Float64}()
    left_stack_d = Vector{Float64}()
    right_stack_d = Vector{Float64}()
    r_stack_d = Vector{Float64}()
    f_stack_d = Vector{Float64}()
    cl_stack_d = Vector{Float64}()
    cr_stack_d = Vector{Float64}()
    left_mg_relax_c2_stack_d = Vector{Float64}()
    right_mg_relax_c2_stack_d = Vector{Float64}()
    cl = 0.0
    cr = 0.0
    hl = 0.0
    hl2 = 0.0
    left = 0.0
    left_mg_relax_c1 = 0.0
    left_mg_relax_c2 = 0.0
    right = 0.0
    right_mg_relax_c1 = 0.0
    right_mg_relax_c2 = 0.0
    clb = 0.0
    crb = 0.0
    hlb = 0.0
    hl2b = 0.0
    leftb = 0.0
    left_mg_relax_c1b = 0.0
    left_mg_relax_c2b = 0.0
    rightb = 0.0
    right_mg_relax_c1b = 0.0
    right_mg_relax_c2b = 0.0
    cld = 0.0
    clbd = 0.0
    crd = 0.0
    crbd = 0.0
    hld = 0.0
    hlbd = 0.0
    hl2d = 0.0
    hl2bd = 0.0
    leftd = 0.0
    leftbd = 0.0
    left_mg_relax_c1d = 0.0
    left_mg_relax_c1bd = 0.0
    left_mg_relax_c2d = 0.0
    left_mg_relax_c2bd = 0.0
    rightd = 0.0
    rightbd = 0.0
    right_mg_relax_c1d = 0.0
    right_mg_relax_c1bd = 0.0
    right_mg_relax_c2d = 0.0
    right_mg_relax_c2bd = 0.0
    n = n * 2
    nl = nfine
    push!(hl_stack_d, hld)
    push!(hl_stack, hl)
    hld = h1d
    hl = h1
    for i_seq_level = 1:num_levels - 1
        n = nl - 1
        push!(hl2_stack_d, hl2d)
        push!(hl2_stack, hl2)
        hl2d = hl * hld + hl * hld
        hl2 = hl * hl
        for i_seq_k_mg_relax_c1 = 1:nu1
            push!(tripcount_stack, n)
            for i_seq_j_mg_relax_c1 = 1:n
                left_mg_relax_c1d = 0.0
                left_mg_relax_c1 = 0.0
                if i_seq_j_mg_relax_c1 > 1
                    push!(branch_stack, 1)
                    push!(left_mg_relax_c1_stack_d, left_mg_relax_c1d)
                    push!(left_mg_relax_c1_stack, left_mg_relax_c1)
                    left_mg_relax_c1d = ud[i_seq_j_mg_relax_c1 - 1, i_seq_level]
                    left_mg_relax_c1 = u[i_seq_j_mg_relax_c1 - 1, i_seq_level]
                else
                    push!(branch_stack, 0)
                end
                right_mg_relax_c1d = 0.0
                right_mg_relax_c1 = 0.0
                if i_seq_j_mg_relax_c1 < n
                    push!(branch_stack, 1)
                    push!(right_mg_relax_c1_stack_d, right_mg_relax_c1d)
                    push!(right_mg_relax_c1_stack, right_mg_relax_c1)
                    right_mg_relax_c1d = ud[i_seq_j_mg_relax_c1 + 1, i_seq_level]
                    right_mg_relax_c1 = u[i_seq_j_mg_relax_c1 + 1, i_seq_level]
                else
                    push!(branch_stack, 0)
                end
                push!(u_stack_d, ud[i_seq_j_mg_relax_c1, i_seq_level])
                push!(u_stack, u[i_seq_j_mg_relax_c1, i_seq_level])
                ud[i_seq_j_mg_relax_c1, i_seq_level] = 0.5 * (((f[i_seq_j_mg_relax_c1, i_seq_level] * hl2d + hl2 * fd[i_seq_j_mg_relax_c1, i_seq_level]) + left_mg_relax_c1d) + right_mg_relax_c1d)
                u[i_seq_j_mg_relax_c1, i_seq_level] = 0.5 * (hl2 * f[i_seq_j_mg_relax_c1, i_seq_level] + left_mg_relax_c1 + right_mg_relax_c1)
            end
        end
        push!(tripcount_stack, n)
        for j = 1:n
            leftd = 0.0
            left = 0.0
            if j > 1
                push!(branch_stack, 1)
                push!(left_stack_d, leftd)
                push!(left_stack, left)
                leftd = ud[j - 1, i_seq_level]
                left = u[j - 1, i_seq_level]
            else
                push!(branch_stack, 0)
            end
            rightd = 0.0
            right = 0.0
            if j < n
                push!(branch_stack, 1)
                push!(right_stack_d, rightd)
                push!(right_stack, right)
                rightd = ud[j + 1, i_seq_level]
                right = u[j + 1, i_seq_level]
            else
                push!(branch_stack, 0)
            end
            push!(r_stack_d, rd[j, i_seq_level])
            push!(r_stack, r[j, i_seq_level])
            rd[j, i_seq_level] = fd[j, i_seq_level] + -(((1.0 / hl2) * ((2.0 * ud[j, i_seq_level] + -leftd) + -rightd) + -(((2.0 * u[j, i_seq_level] - left) - right) / hl2 ^ 2) * hl2d))
            r[j, i_seq_level] = f[j, i_seq_level] - ((2.0 * u[j, i_seq_level] - left) - right) / hl2
        end
        ncg = div(nl, 2)
        nc = ncg - 1
        push!(tripcount_stack, nc)
        for j = 1:nc
            jf = j * 2
            push!(f_stack_d, fd[j, i_seq_level + 1])
            push!(f_stack, f[j, i_seq_level + 1])
            fd[j, i_seq_level + 1] = (0.25 * rd[jf - 1, i_seq_level] + 0.5 * rd[jf, i_seq_level]) + 0.25 * rd[jf + 1, i_seq_level]
            f[j, i_seq_level + 1] = 0.25 * r[jf - 1, i_seq_level] + 0.5 * r[jf, i_seq_level] + 0.25 * r[jf + 1, i_seq_level]
        end
        push!(tripcount_stack, nc)
        for j = 1:nc
            ud[j, i_seq_level + 1] = 0.0
            u[j, i_seq_level + 1] = 0.0
        end
        nl = ncg
        push!(hl_stack_d, hld)
        push!(hl_stack, hl)
        hld = 2.0hld
        hl = hl * 2.0
    end
    push!(hl2_stack_d, hl2d)
    push!(hl2_stack, hl2)
    hl2d = hl * hld + hl * hld
    hl2 = hl * hl
    push!(u_stack_d, ud[1, num_levels])
    push!(u_stack, u[1, num_levels])
    ud[1, num_levels] = (0.5 * f[1, num_levels]) * hl2d + (0.5hl2) * fd[1, num_levels]
    u[1, num_levels] = 0.5 * hl2 * f[1, num_levels]
    for i_seq_level = num_levels - 1:-1:1
        nl = nl * 2
        push!(hl_stack_d, hld)
        push!(hl_stack, hl)
        hld = 0.5hld
        hl = hl / 2.0
        n = nl - 1
        ncg = div(nl, 2)
        nc = ncg - 1
        push!(hl2_stack_d, hl2d)
        push!(hl2_stack, hl2)
        hl2d = hl * hld + hl * hld
        hl2 = hl * hl
        push!(tripcount_stack, nc)
        for j = 1:nc
            jf = j * 2
            push!(u_stack_d, ud[jf, i_seq_level])
            push!(u_stack, u[jf, i_seq_level])
            ud[jf, i_seq_level] = ud[jf, i_seq_level] + ud[j, i_seq_level + 1]
            u[jf, i_seq_level] = u[jf, i_seq_level] + u[j, i_seq_level + 1]
        end
        push!(tripcount_stack, nc)
        for j = 1:nc + 1
            jf = j * 2 - 1
            cld = 0.0
            cl = 0.0
            if j > 1
                push!(branch_stack, 1)
                push!(cl_stack_d, cld)
                push!(cl_stack, cl)
                cld = ud[j - 1, i_seq_level + 1]
                cl = u[j - 1, i_seq_level + 1]
            else
                push!(branch_stack, 0)
            end
            crd = 0.0
            cr = 0.0
            if j <= nc
                push!(branch_stack, 1)
                push!(cr_stack_d, crd)
                push!(cr_stack, cr)
                crd = ud[j, i_seq_level + 1]
                cr = u[j, i_seq_level + 1]
            else
                push!(branch_stack, 0)
            end
            push!(u_stack_d, ud[jf, i_seq_level])
            push!(u_stack, u[jf, i_seq_level])
            ud[jf, i_seq_level] = ud[jf, i_seq_level] + 0.5 * (cld + crd)
            u[jf, i_seq_level] = u[jf, i_seq_level] + 0.5 * (cl + cr)
        end
        for i_seq_k_mg_relax_c2 = 1:nu2
            push!(tripcount_stack, n)
            for i_seq_j_mg_relax_c2 = 1:n
                left_mg_relax_c2d = 0.0
                left_mg_relax_c2 = 0.0
                if i_seq_j_mg_relax_c2 > 1
                    push!(branch_stack, 1)
                    push!(left_mg_relax_c2_stack_d, left_mg_relax_c2d)
                    push!(left_mg_relax_c2_stack, left_mg_relax_c2)
                    left_mg_relax_c2d = ud[i_seq_j_mg_relax_c2 - 1, i_seq_level]
                    left_mg_relax_c2 = u[i_seq_j_mg_relax_c2 - 1, i_seq_level]
                else
                    push!(branch_stack, 0)
                end
                right_mg_relax_c2d = 0.0
                right_mg_relax_c2 = 0.0
                if i_seq_j_mg_relax_c2 < n
                    push!(branch_stack, 1)
                    push!(right_mg_relax_c2_stack_d, right_mg_relax_c2d)
                    push!(right_mg_relax_c2_stack, right_mg_relax_c2)
                    right_mg_relax_c2d = ud[i_seq_j_mg_relax_c2 + 1, i_seq_level]
                    right_mg_relax_c2 = u[i_seq_j_mg_relax_c2 + 1, i_seq_level]
                else
                    push!(branch_stack, 0)
                end
                push!(u_stack_d, ud[i_seq_j_mg_relax_c2, i_seq_level])
                push!(u_stack, u[i_seq_j_mg_relax_c2, i_seq_level])
                ud[i_seq_j_mg_relax_c2, i_seq_level] = 0.5 * (((f[i_seq_j_mg_relax_c2, i_seq_level] * hl2d + hl2 * fd[i_seq_j_mg_relax_c2, i_seq_level]) + left_mg_relax_c2d) + right_mg_relax_c2d)
                u[i_seq_j_mg_relax_c2, i_seq_level] = 0.5 * (hl2 * f[i_seq_j_mg_relax_c2, i_seq_level] + left_mg_relax_c2 + right_mg_relax_c2)
            end
        end
    end
    for i_seq_level = 1:num_levels - 1
        for i_seq_k_mg_relax_c2 = nu2:-1:1
            n = pop!(tripcount_stack)
            for i_seq_j_mg_relax_c2 = n:-1:1
                __branch_pre_4 = pop!(branch_stack)
                if __branch_pre_4 == 1
                    __snap_discard = pop!(right_mg_relax_c2_stack)
                end
                right_mg_relax_c2d = 0.0
                right_mg_relax_c2 = 0.0
                if __branch_pre_4 == 1
                    right_mg_relax_c2d = ud[i_seq_j_mg_relax_c2 + 1, i_seq_level]
                    right_mg_relax_c2 = u[i_seq_j_mg_relax_c2 + 1, i_seq_level]
                else
                    right_mg_relax_c2d = 0.0
                    right_mg_relax_c2 = 0.0
                end
                __branch_pre_2 = pop!(branch_stack)
                if __branch_pre_2 == 1
                    __snap_discard = pop!(left_mg_relax_c2_stack)
                end
                left_mg_relax_c2d = 0.0
                left_mg_relax_c2 = 0.0
                if __branch_pre_2 == 1
                    left_mg_relax_c2d = ud[i_seq_j_mg_relax_c2 - 1, i_seq_level]
                    left_mg_relax_c2 = u[i_seq_j_mg_relax_c2 - 1, i_seq_level]
                else
                    left_mg_relax_c2d = 0.0
                    left_mg_relax_c2 = 0.0
                end
                ud[i_seq_j_mg_relax_c2, i_seq_level] = pop!(u_stack_d)
                u[i_seq_j_mg_relax_c2, i_seq_level] = pop!(u_stack)
                hl2bd = hl2bd + ((0.5 * ub[i_seq_j_mg_relax_c2, i_seq_level]) * fd[i_seq_j_mg_relax_c2, i_seq_level] + f[i_seq_j_mg_relax_c2, i_seq_level] * (0.5 * ubd[i_seq_j_mg_relax_c2, i_seq_level]))
                hl2b = hl2b + f[i_seq_j_mg_relax_c2, i_seq_level] * (0.5 * ub[i_seq_j_mg_relax_c2, i_seq_level])
                fbd[i_seq_j_mg_relax_c2, i_seq_level] = fbd[i_seq_j_mg_relax_c2, i_seq_level] + ((0.5 * ub[i_seq_j_mg_relax_c2, i_seq_level]) * hl2d + hl2 * (0.5 * ubd[i_seq_j_mg_relax_c2, i_seq_level]))
                fb[i_seq_j_mg_relax_c2, i_seq_level] = fb[i_seq_j_mg_relax_c2, i_seq_level] + hl2 * (0.5 * ub[i_seq_j_mg_relax_c2, i_seq_level])
                left_mg_relax_c2bd = left_mg_relax_c2bd + 0.5 * ubd[i_seq_j_mg_relax_c2, i_seq_level]
                left_mg_relax_c2b = left_mg_relax_c2b + 0.5 * ub[i_seq_j_mg_relax_c2, i_seq_level]
                right_mg_relax_c2bd = right_mg_relax_c2bd + 0.5 * ubd[i_seq_j_mg_relax_c2, i_seq_level]
                right_mg_relax_c2b = right_mg_relax_c2b + 0.5 * ub[i_seq_j_mg_relax_c2, i_seq_level]
                ubd[i_seq_j_mg_relax_c2, i_seq_level] = 0.0
                ub[i_seq_j_mg_relax_c2, i_seq_level] = 0.0
                if __branch_pre_4 == 1
                    ubd[i_seq_j_mg_relax_c2 + 1, i_seq_level] = ubd[i_seq_j_mg_relax_c2 + 1, i_seq_level] + right_mg_relax_c2bd
                    ub[i_seq_j_mg_relax_c2 + 1, i_seq_level] = ub[i_seq_j_mg_relax_c2 + 1, i_seq_level] + right_mg_relax_c2b
                    right_mg_relax_c2bd = 0.0
                    right_mg_relax_c2b = 0.0
                end
                right_mg_relax_c2bd = 0.0
                right_mg_relax_c2b = 0.0
                if __branch_pre_2 == 1
                    ubd[i_seq_j_mg_relax_c2 - 1, i_seq_level] = ubd[i_seq_j_mg_relax_c2 - 1, i_seq_level] + left_mg_relax_c2bd
                    ub[i_seq_j_mg_relax_c2 - 1, i_seq_level] = ub[i_seq_j_mg_relax_c2 - 1, i_seq_level] + left_mg_relax_c2b
                    left_mg_relax_c2bd = 0.0
                    left_mg_relax_c2b = 0.0
                end
                left_mg_relax_c2bd = 0.0
                left_mg_relax_c2b = 0.0
            end
        end
        nc = pop!(tripcount_stack)
        for j = nc + 1:-1:1
            jf = j * 2 - 1
            __branch_pre_5 = pop!(branch_stack)
            if __branch_pre_5 == 1
                __snap_discard = pop!(cr_stack)
            end
            crd = 0.0
            cr = 0.0
            if __branch_pre_5 == 1
                crd = ud[j, i_seq_level + 1]
                cr = u[j, i_seq_level + 1]
            else
                crd = 0.0
                cr = 0.0
            end
            __branch_pre_3 = pop!(branch_stack)
            if __branch_pre_3 == 1
                __snap_discard = pop!(cl_stack)
            end
            cld = 0.0
            cl = 0.0
            if __branch_pre_3 == 1
                cld = ud[j - 1, i_seq_level + 1]
                cl = u[j - 1, i_seq_level + 1]
            else
                cld = 0.0
                cl = 0.0
            end
            ud[jf, i_seq_level] = pop!(u_stack_d)
            u[jf, i_seq_level] = pop!(u_stack)
            clbd = clbd + 0.5 * ubd[jf, i_seq_level]
            clb = clb + 0.5 * ub[jf, i_seq_level]
            crbd = crbd + 0.5 * ubd[jf, i_seq_level]
            crb = crb + 0.5 * ub[jf, i_seq_level]
            if __branch_pre_5 == 1
                ubd[j, i_seq_level + 1] = ubd[j, i_seq_level + 1] + crbd
                ub[j, i_seq_level + 1] = ub[j, i_seq_level + 1] + crb
                crbd = 0.0
                crb = 0.0
            end
            crbd = 0.0
            crb = 0.0
            if __branch_pre_3 == 1
                ubd[j - 1, i_seq_level + 1] = ubd[j - 1, i_seq_level + 1] + clbd
                ub[j - 1, i_seq_level + 1] = ub[j - 1, i_seq_level + 1] + clb
                clbd = 0.0
                clb = 0.0
            end
            clbd = 0.0
            clb = 0.0
        end
        nc = pop!(tripcount_stack)
        for j = nc:-1:1
            jf = j * 2
            ud[jf, i_seq_level] = pop!(u_stack_d)
            u[jf, i_seq_level] = pop!(u_stack)
            ubd[j, i_seq_level + 1] = ubd[j, i_seq_level + 1] + ubd[jf, i_seq_level]
            ub[j, i_seq_level + 1] = ub[j, i_seq_level + 1] + ub[jf, i_seq_level]
        end
        hl2d = pop!(hl2_stack_d)
        hl2 = pop!(hl2_stack)
        hlbd = hlbd + (hl2b * hld + hl * hl2bd)
        hlb = hlb + hl * hl2b
        hlbd = hlbd + (hl2b * hld + hl * hl2bd)
        hlb = hlb + hl * hl2b
        hl2bd = 0.0
        hl2b = 0.0
        hld = pop!(hl_stack_d)
        hl = pop!(hl_stack)
        hlbd = 0.5hlbd
        hlb = 0.5hlb
    end
    ud[1, num_levels] = pop!(u_stack_d)
    u[1, num_levels] = pop!(u_stack)
    hl2bd = hl2bd + (ub[1, num_levels] * (0.5 * fd[1, num_levels]) + (0.5 * f[1, num_levels]) * ubd[1, num_levels])
    hl2b = hl2b + (0.5 * f[1, num_levels]) * ub[1, num_levels]
    fbd[1, num_levels] = fbd[1, num_levels] + (ub[1, num_levels] * (0.5hl2d) + (0.5hl2) * ubd[1, num_levels])
    fb[1, num_levels] = fb[1, num_levels] + (0.5hl2) * ub[1, num_levels]
    ubd[1, num_levels] = 0.0
    ub[1, num_levels] = 0.0
    hl2d = pop!(hl2_stack_d)
    hl2 = pop!(hl2_stack)
    hlbd = hlbd + (hl2b * hld + hl * hl2bd)
    hlb = hlb + hl * hl2b
    hlbd = hlbd + (hl2b * hld + hl * hl2bd)
    hlb = hlb + hl * hl2b
    hl2bd = 0.0
    hl2b = 0.0
    for i_seq_level = num_levels - 1:-1:1
        hld = pop!(hl_stack_d)
        hl = pop!(hl_stack)
        hlbd = 2.0hlbd
        hlb = 2.0hlb
        nc = pop!(tripcount_stack)
        for j = 1:nc
            ubd[j, i_seq_level + 1] = 0.0
            ub[j, i_seq_level + 1] = 0.0
        end
        nc = pop!(tripcount_stack)
        for j = nc:-1:1
            jf = j * 2
            fd[j, i_seq_level + 1] = pop!(f_stack_d)
            f[j, i_seq_level + 1] = pop!(f_stack)
            rbd[jf - 1, i_seq_level] = rbd[jf - 1, i_seq_level] + 0.25 * fbd[j, i_seq_level + 1]
            rb[jf - 1, i_seq_level] = rb[jf - 1, i_seq_level] + 0.25 * fb[j, i_seq_level + 1]
            rbd[jf, i_seq_level] = rbd[jf, i_seq_level] + 0.5 * fbd[j, i_seq_level + 1]
            rb[jf, i_seq_level] = rb[jf, i_seq_level] + 0.5 * fb[j, i_seq_level + 1]
            rbd[jf + 1, i_seq_level] = rbd[jf + 1, i_seq_level] + 0.25 * fbd[j, i_seq_level + 1]
            rb[jf + 1, i_seq_level] = rb[jf + 1, i_seq_level] + 0.25 * fb[j, i_seq_level + 1]
            fbd[j, i_seq_level + 1] = 0.0
            fb[j, i_seq_level + 1] = 0.0
        end
        n = pop!(tripcount_stack)
        for j = n:-1:1
            __branch_pre_4 = pop!(branch_stack)
            if __branch_pre_4 == 1
                __snap_discard = pop!(right_stack)
            end
            rightd = 0.0
            right = 0.0
            if __branch_pre_4 == 1
                rightd = ud[j + 1, i_seq_level]
                right = u[j + 1, i_seq_level]
            else
                rightd = 0.0
                right = 0.0
            end
            __branch_pre_2 = pop!(branch_stack)
            if __branch_pre_2 == 1
                __snap_discard = pop!(left_stack)
            end
            leftd = 0.0
            left = 0.0
            if __branch_pre_2 == 1
                leftd = ud[j - 1, i_seq_level]
                left = u[j - 1, i_seq_level]
            else
                leftd = 0.0
                left = 0.0
            end
            rd[j, i_seq_level] = pop!(r_stack_d)
            r[j, i_seq_level] = pop!(r_stack)
            fbd[j, i_seq_level] = fbd[j, i_seq_level] + rbd[j, i_seq_level]
            fb[j, i_seq_level] = fb[j, i_seq_level] + rb[j, i_seq_level]
            ubd[j, i_seq_level] = ubd[j, i_seq_level] + 2.0 * (-(rb[j, i_seq_level]) * (-(1.0 / hl2 ^ 2) * hl2d) + (1.0 / hl2) * -(rbd[j, i_seq_level]))
            ub[j, i_seq_level] = ub[j, i_seq_level] + 2.0 * ((1.0 / hl2) * -(rb[j, i_seq_level]))
            leftbd = leftbd + -((-(rb[j, i_seq_level]) * (-(1.0 / hl2 ^ 2) * hl2d) + (1.0 / hl2) * -(rbd[j, i_seq_level])))
            leftb = leftb + -((1.0 / hl2) * -(rb[j, i_seq_level]))
            rightbd = rightbd + -((-(rb[j, i_seq_level]) * (-(1.0 / hl2 ^ 2) * hl2d) + (1.0 / hl2) * -(rbd[j, i_seq_level])))
            rightb = rightb + -((1.0 / hl2) * -(rb[j, i_seq_level]))
            hl2bd = hl2bd + (-(rb[j, i_seq_level]) * -(((1.0 / hl2 ^ 2) * ((2.0 * ud[j, i_seq_level] + -leftd) + -rightd) + -(((2.0 * u[j, i_seq_level] - left) - right) / (hl2 ^ 2) ^ 2) * ((2hl2) * hl2d))) + -(((2.0 * u[j, i_seq_level] - left) - right) / hl2 ^ 2) * -(rbd[j, i_seq_level]))
            hl2b = hl2b + -(((2.0 * u[j, i_seq_level] - left) - right) / hl2 ^ 2) * -(rb[j, i_seq_level])
            rbd[j, i_seq_level] = 0.0
            rb[j, i_seq_level] = 0.0
            if __branch_pre_4 == 1
                ubd[j + 1, i_seq_level] = ubd[j + 1, i_seq_level] + rightbd
                ub[j + 1, i_seq_level] = ub[j + 1, i_seq_level] + rightb
                rightbd = 0.0
                rightb = 0.0
            end
            rightbd = 0.0
            rightb = 0.0
            if __branch_pre_2 == 1
                ubd[j - 1, i_seq_level] = ubd[j - 1, i_seq_level] + leftbd
                ub[j - 1, i_seq_level] = ub[j - 1, i_seq_level] + leftb
                leftbd = 0.0
                leftb = 0.0
            end
            leftbd = 0.0
            leftb = 0.0
        end
        for i_seq_k_mg_relax_c1 = nu1:-1:1
            n = pop!(tripcount_stack)
            for i_seq_j_mg_relax_c1 = n:-1:1
                __branch_pre_4 = pop!(branch_stack)
                if __branch_pre_4 == 1
                    __snap_discard = pop!(right_mg_relax_c1_stack)
                end
                right_mg_relax_c1d = 0.0
                right_mg_relax_c1 = 0.0
                if __branch_pre_4 == 1
                    right_mg_relax_c1d = ud[i_seq_j_mg_relax_c1 + 1, i_seq_level]
                    right_mg_relax_c1 = u[i_seq_j_mg_relax_c1 + 1, i_seq_level]
                else
                    right_mg_relax_c1d = 0.0
                    right_mg_relax_c1 = 0.0
                end
                __branch_pre_2 = pop!(branch_stack)
                if __branch_pre_2 == 1
                    __snap_discard = pop!(left_mg_relax_c1_stack)
                end
                left_mg_relax_c1d = 0.0
                left_mg_relax_c1 = 0.0
                if __branch_pre_2 == 1
                    left_mg_relax_c1d = ud[i_seq_j_mg_relax_c1 - 1, i_seq_level]
                    left_mg_relax_c1 = u[i_seq_j_mg_relax_c1 - 1, i_seq_level]
                else
                    left_mg_relax_c1d = 0.0
                    left_mg_relax_c1 = 0.0
                end
                ud[i_seq_j_mg_relax_c1, i_seq_level] = pop!(u_stack_d)
                u[i_seq_j_mg_relax_c1, i_seq_level] = pop!(u_stack)
                hl2bd = hl2bd + ((0.5 * ub[i_seq_j_mg_relax_c1, i_seq_level]) * fd[i_seq_j_mg_relax_c1, i_seq_level] + f[i_seq_j_mg_relax_c1, i_seq_level] * (0.5 * ubd[i_seq_j_mg_relax_c1, i_seq_level]))
                hl2b = hl2b + f[i_seq_j_mg_relax_c1, i_seq_level] * (0.5 * ub[i_seq_j_mg_relax_c1, i_seq_level])
                fbd[i_seq_j_mg_relax_c1, i_seq_level] = fbd[i_seq_j_mg_relax_c1, i_seq_level] + ((0.5 * ub[i_seq_j_mg_relax_c1, i_seq_level]) * hl2d + hl2 * (0.5 * ubd[i_seq_j_mg_relax_c1, i_seq_level]))
                fb[i_seq_j_mg_relax_c1, i_seq_level] = fb[i_seq_j_mg_relax_c1, i_seq_level] + hl2 * (0.5 * ub[i_seq_j_mg_relax_c1, i_seq_level])
                left_mg_relax_c1bd = left_mg_relax_c1bd + 0.5 * ubd[i_seq_j_mg_relax_c1, i_seq_level]
                left_mg_relax_c1b = left_mg_relax_c1b + 0.5 * ub[i_seq_j_mg_relax_c1, i_seq_level]
                right_mg_relax_c1bd = right_mg_relax_c1bd + 0.5 * ubd[i_seq_j_mg_relax_c1, i_seq_level]
                right_mg_relax_c1b = right_mg_relax_c1b + 0.5 * ub[i_seq_j_mg_relax_c1, i_seq_level]
                ubd[i_seq_j_mg_relax_c1, i_seq_level] = 0.0
                ub[i_seq_j_mg_relax_c1, i_seq_level] = 0.0
                if __branch_pre_4 == 1
                    ubd[i_seq_j_mg_relax_c1 + 1, i_seq_level] = ubd[i_seq_j_mg_relax_c1 + 1, i_seq_level] + right_mg_relax_c1bd
                    ub[i_seq_j_mg_relax_c1 + 1, i_seq_level] = ub[i_seq_j_mg_relax_c1 + 1, i_seq_level] + right_mg_relax_c1b
                    right_mg_relax_c1bd = 0.0
                    right_mg_relax_c1b = 0.0
                end
                right_mg_relax_c1bd = 0.0
                right_mg_relax_c1b = 0.0
                if __branch_pre_2 == 1
                    ubd[i_seq_j_mg_relax_c1 - 1, i_seq_level] = ubd[i_seq_j_mg_relax_c1 - 1, i_seq_level] + left_mg_relax_c1bd
                    ub[i_seq_j_mg_relax_c1 - 1, i_seq_level] = ub[i_seq_j_mg_relax_c1 - 1, i_seq_level] + left_mg_relax_c1b
                    left_mg_relax_c1bd = 0.0
                    left_mg_relax_c1b = 0.0
                end
                left_mg_relax_c1bd = 0.0
                left_mg_relax_c1b = 0.0
            end
        end
        hl2d = pop!(hl2_stack_d)
        hl2 = pop!(hl2_stack)
        hlbd = hlbd + (hl2b * hld + hl * hl2bd)
        hlb = hlb + hl * hl2b
        hlbd = hlbd + (hl2b * hld + hl * hl2bd)
        hlb = hlb + hl * hl2b
        hl2bd = 0.0
        hl2b = 0.0
    end
    hld = pop!(hl_stack_d)
    hl = pop!(hl_stack)
    h1bd = h1bd + hlbd
    h1b = h1b + hlb
    hlbd = 0.0
    hlb = 0.0
    return (h1b, h1bd)
end

function mg_vcycle_multi(u, f, r, nfine, num_levels, h1, nu1, nu2, n)
    #= none:17 =#
    #= none:18 =#
    n = n * 2
    #= none:19 =#
    nl = nfine
    #= none:20 =#
    hl = h1
    #= none:21 =#
    for i_seq_level = 1:num_levels - 1
        #= none:22 =#
        n = nl - 1
        #= none:23 =#
        hl2 = hl * hl
        #= none:24 =#
        mg_relax(u, f, n, hl2, i_seq_level, nu1)
        #= none:25 =#
        for j = 1:n
            #= none:26 =#
            left = 0.0
            #= none:27 =#
            if j > 1
                #= none:28 =#
                left = u[j - 1, i_seq_level]
            end
            #= none:30 =#
            right = 0.0
            #= none:31 =#
            if j < n
                #= none:32 =#
                right = u[j + 1, i_seq_level]
            end
            #= none:34 =#
            r[j, i_seq_level] = f[j, i_seq_level] - ((2.0 * u[j, i_seq_level] - left) - right) / hl2
            #= none:35 =#
        end
        #= none:36 =#
        ncg = div(nl, 2)
        #= none:37 =#
        nc = ncg - 1
        #= none:38 =#
        for j = 1:nc
            #= none:39 =#
            jf = j * 2
            #= none:40 =#
            f[j, i_seq_level + 1] = 0.25 * r[jf - 1, i_seq_level] + 0.5 * r[jf, i_seq_level] + 0.25 * r[jf + 1, i_seq_level]
            #= none:41 =#
        end
        #= none:42 =#
        for j = 1:nc
            #= none:43 =#
            u[j, i_seq_level + 1] = 0.0
            #= none:44 =#
        end
        #= none:45 =#
        nl = ncg
        #= none:46 =#
        hl = hl * 2.0
        #= none:47 =#
    end
    #= none:48 =#
    hl2 = hl * hl
    #= none:49 =#
    u[1, num_levels] = 0.5 * hl2 * f[1, num_levels]
    #= none:50 =#
    for i_seq_level = num_levels - 1:-1:1
        #= none:51 =#
        nl = nl * 2
        #= none:52 =#
        hl = hl / 2.0
        #= none:53 =#
        n = nl - 1
        #= none:54 =#
        ncg = div(nl, 2)
        #= none:55 =#
        nc = ncg - 1
        #= none:56 =#
        hl2 = hl * hl
        #= none:57 =#
        for j = 1:nc
            #= none:58 =#
            jf = j * 2
            #= none:59 =#
            u[jf, i_seq_level] = u[jf, i_seq_level] + u[j, i_seq_level + 1]
            #= none:60 =#
        end
        #= none:61 =#
        for j = 1:nc + 1
            #= none:62 =#
            jf = j * 2 - 1
            #= none:63 =#
            cl = 0.0
            #= none:64 =#
            if j > 1
                #= none:65 =#
                cl = u[j - 1, i_seq_level + 1]
            end
            #= none:67 =#
            cr = 0.0
            #= none:68 =#
            if j <= nc
                #= none:69 =#
                cr = u[j, i_seq_level + 1]
            end
            #= none:71 =#
            u[jf, i_seq_level] = u[jf, i_seq_level] + 0.5 * (cl + cr)
            #= none:72 =#
        end
        #= none:73 =#
        mg_relax(u, f, n, hl2, i_seq_level, nu2)
        #= none:74 =#
    end
end
