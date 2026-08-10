function initstacks_mg_vcycle_b()
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

function mg_vcycle_b(u, ub, f, fb, r, rb, nfine, num_levels, h1, h1b, nu1, nu2, n, hl_stack, hl2_stack, tripcount_stack, left_stack, branch_stack, right_stack, u_stack, r_stack, f_stack, cl_stack, cr_stack)
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
    push!(hl_stack, hl)
    hl = h1
    for i_seq_level = 1:num_levels - 1
        n = nl - 1
        push!(hl2_stack, hl2)
        hl2 = hl * hl
        for i_seq_k = 1:nu1
            push!(tripcount_stack, n)
            for i_seq_j = 1:n
                left = 0.0
                if i_seq_j > 1
                    push!(branch_stack, 1)
                    push!(left_stack, left)
                    left = u[i_seq_j - 1, i_seq_level]
                else
                    push!(branch_stack, 0)
                end
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
            end
        end
        push!(tripcount_stack, n)
        for j = 1:n
            left = 0.0
            if j > 1
                push!(branch_stack, 1)
                push!(left_stack, left)
                left = u[j - 1, i_seq_level]
            else
                push!(branch_stack, 0)
            end
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
        end
        ncg = div(nl, 2)
        nc = ncg - 1
        push!(tripcount_stack, nc)
        for j = 1:nc
            jf = j * 2
            push!(f_stack, f[j, i_seq_level + 1])
            f[j, i_seq_level + 1] = 0.25 * r[jf - 1, i_seq_level] + 0.5 * r[jf, i_seq_level] + 0.25 * r[jf + 1, i_seq_level]
        end
        push!(tripcount_stack, nc)
        for j = 1:nc
            u[j, i_seq_level + 1] = 0.0
        end
        nl = ncg
        push!(hl_stack, hl)
        hl = hl * 2.0
    end
    push!(hl2_stack, hl2)
    hl2 = hl * hl
    push!(u_stack, u[1, num_levels])
    u[1, num_levels] = 0.5 * hl2 * f[1, num_levels]
    for i_seq_level = num_levels - 1:-1:1
        nl = nl * 2
        push!(hl_stack, hl)
        hl = hl / 2.0
        n = nl - 1
        ncg = div(nl, 2)
        nc = ncg - 1
        push!(hl2_stack, hl2)
        hl2 = hl * hl
        push!(tripcount_stack, nc)
        for j = 1:nc
            jf = j * 2
            push!(u_stack, u[jf, i_seq_level])
            u[jf, i_seq_level] = u[jf, i_seq_level] + u[j, i_seq_level + 1]
        end
        push!(tripcount_stack, nc)
        for j = 1:nc + 1
            jf = j * 2 - 1
            cl = 0.0
            if j > 1
                push!(branch_stack, 1)
                push!(cl_stack, cl)
                cl = u[j - 1, i_seq_level + 1]
            else
                push!(branch_stack, 0)
            end
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
        end
        for i_seq_k = 1:nu2
            push!(tripcount_stack, n)
            for i_seq_j = 1:n
                left = 0.0
                if i_seq_j > 1
                    push!(branch_stack, 1)
                    push!(left_stack, left)
                    left = u[i_seq_j - 1, i_seq_level]
                else
                    push!(branch_stack, 0)
                end
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
                rightb = 0.0
                if __branch_pre_2 == 1
                    ub[i_seq_j - 1, i_seq_level] = ub[i_seq_j - 1, i_seq_level] + leftb
                    leftb = 0.0
                end
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
            cr = 0.0
            if __branch_pre_5 == 1
                cr = u[j, i_seq_level + 1]
            else
                cr = 0.0
            end
            __branch_pre_3 = pop!(branch_stack)
            if __branch_pre_3 == 1
                __snap_discard = pop!(cl_stack)
            end
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
            crb = 0.0
            if __branch_pre_3 == 1
                ub[j - 1, i_seq_level + 1] = ub[j - 1, i_seq_level + 1] + clb
                clb = 0.0
            end
            clb = 0.0
        end
        nc = pop!(tripcount_stack)
        for j = nc:-1:1
            jf = j * 2
            u[jf, i_seq_level] = pop!(u_stack)
            ub[j, i_seq_level + 1] = ub[j, i_seq_level + 1] + ub[jf, i_seq_level]
        end
        hl2 = pop!(hl2_stack)
        hlb = hlb + hl * hl2b
        hlb = hlb + hl * hl2b
        hl2b = 0.0
        hl = pop!(hl_stack)
        hlb = 0.5hlb
    end
    u[1, num_levels] = pop!(u_stack)
    hl2b = hl2b + (0.5 * f[1, num_levels]) * ub[1, num_levels]
    fb[1, num_levels] = fb[1, num_levels] + (0.5hl2) * ub[1, num_levels]
    ub[1, num_levels] = 0.0
    hl2 = pop!(hl2_stack)
    hlb = hlb + hl * hl2b
    hlb = hlb + hl * hl2b
    hl2b = 0.0
    for i_seq_level = num_levels - 1:-1:1
        hl = pop!(hl_stack)
        hlb = 2.0hlb
        nc = pop!(tripcount_stack)
        for j = 1:nc
            ub[j, i_seq_level + 1] = 0.0
        end
        nc = pop!(tripcount_stack)
        for j = nc:-1:1
            jf = j * 2
            f[j, i_seq_level + 1] = pop!(f_stack)
            rb[jf - 1, i_seq_level] = rb[jf - 1, i_seq_level] + 0.25 * fb[j, i_seq_level + 1]
            rb[jf, i_seq_level] = rb[jf, i_seq_level] + 0.5 * fb[j, i_seq_level + 1]
            rb[jf + 1, i_seq_level] = rb[jf + 1, i_seq_level] + 0.25 * fb[j, i_seq_level + 1]
            fb[j, i_seq_level + 1] = 0.0
        end
        n = pop!(tripcount_stack)
        for j = n:-1:1
            __branch_pre_4 = pop!(branch_stack)
            if __branch_pre_4 == 1
                __snap_discard = pop!(right_stack)
            end
            right = 0.0
            if __branch_pre_4 == 1
                right = u[j + 1, i_seq_level]
            else
                right = 0.0
            end
            __branch_pre_2 = pop!(branch_stack)
            if __branch_pre_2 == 1
                __snap_discard = pop!(left_stack)
            end
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
            rightb = 0.0
            if __branch_pre_2 == 1
                ub[j - 1, i_seq_level] = ub[j - 1, i_seq_level] + leftb
                leftb = 0.0
            end
            leftb = 0.0
        end
        for i_seq_k = nu1:-1:1
            n = pop!(tripcount_stack)
            for i_seq_j = n:-1:1
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
                rightb = 0.0
                if __branch_pre_2 == 1
                    ub[i_seq_j - 1, i_seq_level] = ub[i_seq_j - 1, i_seq_level] + leftb
                    leftb = 0.0
                end
                leftb = 0.0
            end
        end
        hl2 = pop!(hl2_stack)
        hlb = hlb + hl * hl2b
        hlb = hlb + hl * hl2b
        hl2b = 0.0
    end
    hl = pop!(hl_stack)
    h1b = h1b + hlb
    hlb = 0.0
    return h1b
end

function mg_vcycle(u, f, r, nfine, num_levels, h1, nu1, nu2, n)
    #= none:1 =#
    #= none:2 =#
    n = n * 2
    #= none:3 =#
    nl = nfine
    #= none:4 =#
    hl = h1
    #= none:5 =#
    for i_seq_level = 1:num_levels - 1
        #= none:6 =#
        n = nl - 1
        #= none:7 =#
        hl2 = hl * hl
        #= none:8 =#
        for i_seq_k = 1:nu1
            #= none:9 =#
            for i_seq_j = 1:n
                #= none:10 =#
                left = 0.0
                #= none:11 =#
                if i_seq_j > 1
                    #= none:12 =#
                    left = u[i_seq_j - 1, i_seq_level]
                end
                #= none:14 =#
                right = 0.0
                #= none:15 =#
                if i_seq_j < n
                    #= none:16 =#
                    right = u[i_seq_j + 1, i_seq_level]
                end
                #= none:18 =#
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
                #= none:19 =#
            end
            #= none:20 =#
        end
        #= none:21 =#
        for j = 1:n
            #= none:22 =#
            left = 0.0
            #= none:23 =#
            if j > 1
                #= none:24 =#
                left = u[j - 1, i_seq_level]
            end
            #= none:26 =#
            right = 0.0
            #= none:27 =#
            if j < n
                #= none:28 =#
                right = u[j + 1, i_seq_level]
            end
            #= none:30 =#
            r[j, i_seq_level] = f[j, i_seq_level] - ((2.0 * u[j, i_seq_level] - left) - right) / hl2
            #= none:31 =#
        end
        #= none:32 =#
        ncg = div(nl, 2)
        #= none:33 =#
        nc = ncg - 1
        #= none:34 =#
        for j = 1:nc
            #= none:35 =#
            jf = j * 2
            #= none:36 =#
            f[j, i_seq_level + 1] = 0.25 * r[jf - 1, i_seq_level] + 0.5 * r[jf, i_seq_level] + 0.25 * r[jf + 1, i_seq_level]
            #= none:37 =#
        end
        #= none:38 =#
        for j = 1:nc
            #= none:39 =#
            u[j, i_seq_level + 1] = 0.0
            #= none:40 =#
        end
        #= none:41 =#
        nl = ncg
        #= none:42 =#
        hl = hl * 2.0
        #= none:43 =#
    end
    #= none:44 =#
    hl2 = hl * hl
    #= none:45 =#
    u[1, num_levels] = 0.5 * hl2 * f[1, num_levels]
    #= none:46 =#
    for i_seq_level = num_levels - 1:-1:1
        #= none:47 =#
        nl = nl * 2
        #= none:48 =#
        hl = hl / 2.0
        #= none:49 =#
        n = nl - 1
        #= none:50 =#
        ncg = div(nl, 2)
        #= none:51 =#
        nc = ncg - 1
        #= none:52 =#
        hl2 = hl * hl
        #= none:53 =#
        for j = 1:nc
            #= none:54 =#
            jf = j * 2
            #= none:55 =#
            u[jf, i_seq_level] = u[jf, i_seq_level] + u[j, i_seq_level + 1]
            #= none:56 =#
        end
        #= none:57 =#
        for j = 1:nc + 1
            #= none:58 =#
            jf = j * 2 - 1
            #= none:59 =#
            cl = 0.0
            #= none:60 =#
            if j > 1
                #= none:61 =#
                cl = u[j - 1, i_seq_level + 1]
            end
            #= none:63 =#
            cr = 0.0
            #= none:64 =#
            if j <= nc
                #= none:65 =#
                cr = u[j, i_seq_level + 1]
            end
            #= none:67 =#
            u[jf, i_seq_level] = u[jf, i_seq_level] + 0.5 * (cl + cr)
            #= none:68 =#
        end
        #= none:69 =#
        for i_seq_k = 1:nu2
            #= none:70 =#
            for i_seq_j = 1:n
                #= none:71 =#
                left = 0.0
                #= none:72 =#
                if i_seq_j > 1
                    #= none:73 =#
                    left = u[i_seq_j - 1, i_seq_level]
                end
                #= none:75 =#
                right = 0.0
                #= none:76 =#
                if i_seq_j < n
                    #= none:77 =#
                    right = u[i_seq_j + 1, i_seq_level]
                end
                #= none:79 =#
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
                #= none:80 =#
            end
            #= none:81 =#
        end
        #= none:82 =#
    end
end
