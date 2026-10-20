function initstacks_idm_refuse_cond_read_b(i_n)
    v_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1))
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1))
    return (v_stack, branch_stack)
end

function idm_refuse_cond_read_b(v, vb, z, zb, x, xb, i_n, v_stack, branch_stack)
    for i_t = 1:i_n
        __icse_0 = (i_t - 1) + 1
        __idx_v_stack_0 = __icse_0
        __cse_1 = v[i_t]
        v_stack[__idx_v_stack_0] = __cse_1
        __cse_2 = x[i_t]
        v[i_t] = __cse_1 + __cse_2
        __idx_v_stack_3 = max(0, div(i_n - 1, 1) + 1) + __icse_0
        __cse_3 = v[i_t + 1]
        v_stack[__idx_v_stack_3] = __cse_3
        v[i_t + 1] = __cse_3 + __cse_2 * 2.0
        if v[i_t] > 0.0
            __idx_branch_stack_0 = (i_t - 1) + 1
            branch_stack[__idx_branch_stack_0] = 1
            z[i_t] = x[i_t] * 3.0
        else
            __idx_branch_stack_0 = (i_t - 1) + 1
            branch_stack[__idx_branch_stack_0] = 0
            z[i_t] = x[i_t]
        end
    end
    for i_t = i_n:-1:1
        __idx_branch_stack_0 = (i_t - 1) + 1
        __branch = branch_stack[__idx_branch_stack_0]
        if __branch == 1
            __oldb_0 = zb[i_t]
            zb[i_t] = 0.0
            xb[i_t] = xb[i_t] + 3.0__oldb_0
        else
            __oldb_0 = zb[i_t]
            zb[i_t] = 0.0
            xb[i_t] = xb[i_t] + __oldb_0
        end
        __icse_4 = (i_t - 1) + 1
        __idx_v_stack_0 = max(0, div(i_n - 1, 1) + 1) + __icse_4
        v[i_t + 1] = v_stack[__idx_v_stack_0]
        xb[i_t] = xb[i_t] + 2.0 * vb[i_t + 1]
        __idx_v_stack_0 = __icse_4
        v[i_t] = v_stack[__idx_v_stack_0]
        xb[i_t] = xb[i_t] + vb[i_t]
    end
    return nothing
end

function idm_refuse_cond_read(v, z, x, i_n)
    for i_t = 1:i_n
        v[i_t] = v[i_t] + x[i_t]
        v[i_t + 1] = v[i_t + 1] + x[i_t] * 2.0
        if v[i_t] > 0.0
            z[i_t] = x[i_t] * 3.0
        else
            z[i_t] = x[i_t]
        end
    end
end
