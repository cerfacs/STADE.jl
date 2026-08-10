import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_mg_vcycle_hv_1!(__jacc_i, i_seq_level, nc, u, ud)
    j = 1 + (__jacc_i - 1)
    ud[j, i_seq_level + 1] = 0.0
    u[j, i_seq_level + 1] = 0.0
    return nothing
end

function jacc_kernel_mg_vcycle_hv_2!(__jacc_i, i_seq_level, nc, ub, ubd)
    j = 1 + (__jacc_i - 1)
    ubd[j, i_seq_level + 1] = 0.0
    ub[j, i_seq_level + 1] = 0.0
    return nothing
end

function jacc_kernel_mg_vcycle_1!(__jacc_i, f, hl2, i_seq_level, n, r, u)
    j = 1 + (__jacc_i - 1)
    left = 0.0
    if j > 1
        left = u[j - 1, i_seq_level]
    end
    right = 0.0
    if j < n
        right = u[j + 1, i_seq_level]
    end
    r[j, i_seq_level] = f[j, i_seq_level] - ((2.0 * u[j, i_seq_level] - left) - right) / hl2
    return nothing
end

function jacc_kernel_mg_vcycle_2!(__jacc_i, f, i_seq_level, nc, r)
    j = 1 + (__jacc_i - 1)
    jf = j * 2
    f[j, i_seq_level + 1] = 0.25 * r[jf - 1, i_seq_level] + 0.5 * r[jf, i_seq_level] + 0.25 * r[jf + 1, i_seq_level]
    return nothing
end

function jacc_kernel_mg_vcycle_3!(__jacc_i, i_seq_level, nc, u)
    j = 1 + (__jacc_i - 1)
    u[j, i_seq_level + 1] = 0.0
    return nothing
end

function jacc_kernel_mg_vcycle_4!(__jacc_i, i_seq_level, nc, u)
    j = 1 + (__jacc_i - 1)
    jf = j * 2
    Atomix.@atomic u[jf, i_seq_level] += u[j, i_seq_level + 1]
    return nothing
end

function jacc_kernel_mg_vcycle_5!(__jacc_i, i_seq_level, nc, u)
    j = 1 + (__jacc_i - 1)
    jf = j * 2 - 1
    cl = 0.0
    if j > 1
        cl = u[j - 1, i_seq_level + 1]
    end
    cr = 0.0
    if j <= nc
        cr = u[j, i_seq_level + 1]
    end
    Atomix.@atomic u[jf, i_seq_level] += 0.5 * (cl + cr)
    return nothing
end

function initstacks_mg_vcycle_b_jacc()
    hl_stack = Vector{Float64}()
    hl2_stack = Vector{Float64}()
    tripcount_stack = Vector{Int64}()
    left_stack = Vector{Float64}()
    branch_stack = Vector{Int64}()
    right_stack = Vector{Float64}()
    u_stack = Vector{Float64}()
    r_stack = Vector{Float64}()
    f_stack = Vector{Float64}()
    cl_stack = Vector{Float64}()
    cr_stack = Vector{Float64}()
    return (hl_stack, hl2_stack, tripcount_stack, left_stack, branch_stack, right_stack, u_stack, r_stack, f_stack, cl_stack, cr_stack)
end

function mg_vcycle_hv_jacc(u, ub, f, fb, r, rb, nfine, num_levels, h1, h1b, nu1, nu2, n, ud, ubd, fd, fbd, rd, rbd, h1d, h1bd, hl_stack, hl2_stack, tripcount_stack, left_stack, branch_stack, right_stack, u_stack, r_stack, f_stack, cl_stack, cr_stack)
    hl_stack_d = Vector{Float64}()
    hl2_stack_d = Vector{Float64}()
    left_stack_d = Vector{Float64}()
    right_stack_d = Vector{Float64}()
    u_stack_d = Vector{Float64}()
    r_stack_d = Vector{Float64}()
    f_stack_d = Vector{Float64}()
    cl_stack_d = Vector{Float64}()
    cr_stack_d = Vector{Float64}()
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
    rightd = 0.0
    rightbd = 0.0
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
        for i_seq_k = 1:nu1
            push!(tripcount_stack, n)
            for i_seq_j = 1:n
                leftd = 0.0
                left = 0.0
                if i_seq_j > 1
                    push!(branch_stack, 1)
                    push!(left_stack_d, leftd)
                    push!(left_stack, left)
                    leftd = ud[i_seq_j - 1, i_seq_level]
                    left = u[i_seq_j - 1, i_seq_level]
                else
                    push!(branch_stack, 0)
                end
                rightd = 0.0
                right = 0.0
                if i_seq_j < n
                    push!(branch_stack, 1)
                    push!(right_stack_d, rightd)
                    push!(right_stack, right)
                    rightd = ud[i_seq_j + 1, i_seq_level]
                    right = u[i_seq_j + 1, i_seq_level]
                else
                    push!(branch_stack, 0)
                end
                push!(u_stack_d, ud[i_seq_j, i_seq_level])
                push!(u_stack, u[i_seq_j, i_seq_level])
                ud[i_seq_j, i_seq_level] = 0.5 * (((f[i_seq_j, i_seq_level] * hl2d + hl2 * fd[i_seq_j, i_seq_level]) + leftd) + rightd)
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
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
        JACC.parallel_for(div(nc - 1, 1) + 1, jacc_kernel_mg_vcycle_hv_1!, i_seq_level, nc, u, ud)
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
        for i_seq_k = 1:nu2
            push!(tripcount_stack, n)
            for i_seq_j = 1:n
                leftd = 0.0
                left = 0.0
                if i_seq_j > 1
                    push!(branch_stack, 1)
                    push!(left_stack_d, leftd)
                    push!(left_stack, left)
                    leftd = ud[i_seq_j - 1, i_seq_level]
                    left = u[i_seq_j - 1, i_seq_level]
                else
                    push!(branch_stack, 0)
                end
                rightd = 0.0
                right = 0.0
                if i_seq_j < n
                    push!(branch_stack, 1)
                    push!(right_stack_d, rightd)
                    push!(right_stack, right)
                    rightd = ud[i_seq_j + 1, i_seq_level]
                    right = u[i_seq_j + 1, i_seq_level]
                else
                    push!(branch_stack, 0)
                end
                push!(u_stack_d, ud[i_seq_j, i_seq_level])
                push!(u_stack, u[i_seq_j, i_seq_level])
                ud[i_seq_j, i_seq_level] = 0.5 * (((f[i_seq_j, i_seq_level] * hl2d + hl2 * fd[i_seq_j, i_seq_level]) + leftd) + rightd)
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
            end
        end
    end
    for i_seq_level = 1:num_levels - 1
        for i_seq_k = nu2:-1:1
            n = pop!(tripcount_stack)
            for i_seq_j = n:-1:1
                __branch_pre_4 = pop!(branch_stack)
                if __branch_pre_4 == 1
                    __snap_discard = pop!(right_stack)
                end
                rightd = 0.0
                right = 0.0
                if __branch_pre_4 == 1
                    rightd = ud[i_seq_j + 1, i_seq_level]
                    right = u[i_seq_j + 1, i_seq_level]
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
                    leftd = ud[i_seq_j - 1, i_seq_level]
                    left = u[i_seq_j - 1, i_seq_level]
                else
                    leftd = 0.0
                    left = 0.0
                end
                ud[i_seq_j, i_seq_level] = pop!(u_stack_d)
                u[i_seq_j, i_seq_level] = pop!(u_stack)
                hl2bd = hl2bd + ((0.5 * ub[i_seq_j, i_seq_level]) * fd[i_seq_j, i_seq_level] + f[i_seq_j, i_seq_level] * (0.5 * ubd[i_seq_j, i_seq_level]))
                hl2b = hl2b + f[i_seq_j, i_seq_level] * (0.5 * ub[i_seq_j, i_seq_level])
                fbd[i_seq_j, i_seq_level] = fbd[i_seq_j, i_seq_level] + ((0.5 * ub[i_seq_j, i_seq_level]) * hl2d + hl2 * (0.5 * ubd[i_seq_j, i_seq_level]))
                fb[i_seq_j, i_seq_level] = fb[i_seq_j, i_seq_level] + hl2 * (0.5 * ub[i_seq_j, i_seq_level])
                leftbd = leftbd + 0.5 * ubd[i_seq_j, i_seq_level]
                leftb = leftb + 0.5 * ub[i_seq_j, i_seq_level]
                rightbd = rightbd + 0.5 * ubd[i_seq_j, i_seq_level]
                rightb = rightb + 0.5 * ub[i_seq_j, i_seq_level]
                ubd[i_seq_j, i_seq_level] = 0.0
                ub[i_seq_j, i_seq_level] = 0.0
                if __branch_pre_4 == 1
                    ubd[i_seq_j + 1, i_seq_level] = ubd[i_seq_j + 1, i_seq_level] + rightbd
                    ub[i_seq_j + 1, i_seq_level] = ub[i_seq_j + 1, i_seq_level] + rightb
                    rightbd = 0.0
                    rightb = 0.0
                end
                rightbd = 0.0
                rightb = 0.0
                if __branch_pre_2 == 1
                    ubd[i_seq_j - 1, i_seq_level] = ubd[i_seq_j - 1, i_seq_level] + leftbd
                    ub[i_seq_j - 1, i_seq_level] = ub[i_seq_j - 1, i_seq_level] + leftb
                    leftbd = 0.0
                    leftb = 0.0
                end
                leftbd = 0.0
                leftb = 0.0
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
        JACC.parallel_for(div(nc - 1, 1) + 1, jacc_kernel_mg_vcycle_hv_2!, i_seq_level, nc, ub, ubd)
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
        for i_seq_k = nu1:-1:1
            n = pop!(tripcount_stack)
            for i_seq_j = n:-1:1
                __branch_pre_4 = pop!(branch_stack)
                if __branch_pre_4 == 1
                    __snap_discard = pop!(right_stack)
                end
                rightd = 0.0
                right = 0.0
                if __branch_pre_4 == 1
                    rightd = ud[i_seq_j + 1, i_seq_level]
                    right = u[i_seq_j + 1, i_seq_level]
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
                    leftd = ud[i_seq_j - 1, i_seq_level]
                    left = u[i_seq_j - 1, i_seq_level]
                else
                    leftd = 0.0
                    left = 0.0
                end
                ud[i_seq_j, i_seq_level] = pop!(u_stack_d)
                u[i_seq_j, i_seq_level] = pop!(u_stack)
                hl2bd = hl2bd + ((0.5 * ub[i_seq_j, i_seq_level]) * fd[i_seq_j, i_seq_level] + f[i_seq_j, i_seq_level] * (0.5 * ubd[i_seq_j, i_seq_level]))
                hl2b = hl2b + f[i_seq_j, i_seq_level] * (0.5 * ub[i_seq_j, i_seq_level])
                fbd[i_seq_j, i_seq_level] = fbd[i_seq_j, i_seq_level] + ((0.5 * ub[i_seq_j, i_seq_level]) * hl2d + hl2 * (0.5 * ubd[i_seq_j, i_seq_level]))
                fb[i_seq_j, i_seq_level] = fb[i_seq_j, i_seq_level] + hl2 * (0.5 * ub[i_seq_j, i_seq_level])
                leftbd = leftbd + 0.5 * ubd[i_seq_j, i_seq_level]
                leftb = leftb + 0.5 * ub[i_seq_j, i_seq_level]
                rightbd = rightbd + 0.5 * ubd[i_seq_j, i_seq_level]
                rightb = rightb + 0.5 * ub[i_seq_j, i_seq_level]
                ubd[i_seq_j, i_seq_level] = 0.0
                ub[i_seq_j, i_seq_level] = 0.0
                if __branch_pre_4 == 1
                    ubd[i_seq_j + 1, i_seq_level] = ubd[i_seq_j + 1, i_seq_level] + rightbd
                    ub[i_seq_j + 1, i_seq_level] = ub[i_seq_j + 1, i_seq_level] + rightb
                    rightbd = 0.0
                    rightb = 0.0
                end
                rightbd = 0.0
                rightb = 0.0
                if __branch_pre_2 == 1
                    ubd[i_seq_j - 1, i_seq_level] = ubd[i_seq_j - 1, i_seq_level] + leftbd
                    ub[i_seq_j - 1, i_seq_level] = ub[i_seq_j - 1, i_seq_level] + leftb
                    leftbd = 0.0
                    leftb = 0.0
                end
                leftbd = 0.0
                leftb = 0.0
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

function mg_vcycle_jacc(u, f, r, nfine, num_levels, h1, nu1, nu2, n)
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
        JACC.parallel_for(div(n - 1, 1) + 1, jacc_kernel_mg_vcycle_1!, f, hl2, i_seq_level, n, r, u)
        ncg = div(nl, 2)
        nc = ncg - 1
        JACC.parallel_for(div(nc - 1, 1) + 1, jacc_kernel_mg_vcycle_2!, f, i_seq_level, nc, r)
        JACC.parallel_for(div(nc - 1, 1) + 1, jacc_kernel_mg_vcycle_3!, i_seq_level, nc, u)
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
        JACC.parallel_for(div(nc - 1, 1) + 1, jacc_kernel_mg_vcycle_4!, i_seq_level, nc, u)
        JACC.parallel_for(div((nc + 1) - 1, 1) + 1, jacc_kernel_mg_vcycle_5!, i_seq_level, nc, u)
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
    return nothing
end
