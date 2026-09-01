function initstacks_bnd_branch_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1))
    s_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
    return (branch_stack, s_stack)
end

function bnd_branch_hv(x, xb, flag, flagb, i_n, i_m, out, outb, xd, xbd, flagd, flagbd, outd, outbd, branch_stack, s_stack)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    sd = 0.0
    s = 0.0
    for i_i = 1:i_n
        if flag[i_i] > 0.0
            branch_stack[1] = 1
            sd = 0.0
            s = 0.0
            for i_j = 1:i_m
                __cse_3 = x[i_i]
                __cse_4 = x[i_j]
                sd = sd + (__cse_3 * xd[i_j] + __cse_4 * xd[i_i])
                s = s + __cse_4 * __cse_3
            end
            __cse_5 = s * sd
            outd[i_i] = outd[i_i] + (__cse_5 + __cse_5)
            out[i_i] = out[i_i] + s * s
        else
            branch_stack[1] = 0
        end
        __branch = branch_stack[1]
        if __branch == 1
            __cse_6 = outb[i_i]
            __cse_0d = __cse_6 * sd + s * outbd[i_i]
            __cse_0 = s * __cse_6
            sbd = sbd + __cse_0d
            sb = sb + __cse_0
            sbd = sbd + __cse_0d
            sb = sb + __cse_0
            for i_j = 1:i_m
                __cse_1d = xd[i_j]
                __cse_1 = x[i_j]
                __cse_2d = xd[i_i]
                __cse_2 = x[i_i]
                sd = sd + (__cse_2 * __cse_1d + __cse_1 * __cse_2d)
                s = s + __cse_1 * __cse_2
                xbd[i_j] = xbd[i_j] + (sb * __cse_2d + __cse_2 * sbd)
                xb[i_j] = xb[i_j] + __cse_2 * sb
                xbd[i_i] = xbd[i_i] + (sb * __cse_1d + __cse_1 * sbd)
                xb[i_i] = xb[i_i] + __cse_1 * sb
            end
            sbd = 0.0
            sb = 0.0
        end
    end
    __idx_s_stack_2 = max(0, div(i_n - 1, 1) + 1) + 1
    s_stack_d[__idx_s_stack_2] = sd
    s_stack[__idx_s_stack_2] = s
    __idx_s_stack_0 = max(0, div(i_n - 1, 1) + 1) + 1
    sd = s_stack_d[__idx_s_stack_0]
    s = s_stack[__idx_s_stack_0]
    sbd = 0.0
    sb = 0.0
    return nothing
end

function bnd_branch(x, flag, i_n, i_m, out)
    s = 0.0
    for i_i = 1:i_n
        if flag[i_i] > 0.0
            s = 0.0
            for i_j = 1:i_m
                s = s + x[i_j] * x[i_i]
            end
            out[i_i] = out[i_i] + s * s
        end
    end
end
