function initstacks_bnd_branch_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1))
    s_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
    return (branch_stack, s_stack)
end

function bnd_branch_b(x, xb, flag, flagb, i_n, i_m, out, outb, branch_stack, s_stack)
    s = 0.0
    sb = 0.0
    s = 0.0
    for i_i = 1:i_n
        if flag[i_i] > 0.0
            branch_stack[1] = 1
            s = 0.0
            for i_j = 1:i_m
                s = s + x[i_j] * x[i_i]
            end
            out[i_i] = out[i_i] + s * s
        else
            branch_stack[1] = 0
        end
        __branch = branch_stack[1]
        if __branch == 1
            __cse_0 = s * outb[i_i]
            sb = sb + __cse_0
            sb = sb + __cse_0
            for i_j = 1:i_m
                __cse_1 = x[i_j]
                __cse_2 = x[i_i]
                s = s + __cse_1 * __cse_2
                xb[i_j] = xb[i_j] + __cse_2 * sb
                xb[i_i] = xb[i_i] + __cse_1 * sb
            end
            sb = 0.0
        end
    end
    __icse_3 = max(0, div(i_n - 1, 1) + 1) + 1
    __idx_s_stack_2 = __icse_3
    s_stack[__idx_s_stack_2] = s
    __idx_s_stack_0 = __icse_3
    s = s_stack[__idx_s_stack_0]
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
