function initstacks_idm_refuse_cond_read_b(i_n)
    v_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1))
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1))
    return (v_stack, branch_stack)
end

function idm_refuse_cond_read_hv(v, vb, z, zb, x, xb, i_n, vd, vbd, zd, zbd, xd, xbd, v_stack, branch_stack)
    v_stack_d = Vector{Float64}(undef, length(v_stack))
    for i_t = 1:i_n
        __icse_0 = (i_t - 1) + 1
        __idx_v_stack_0 = __icse_0
        __cse_1d = vd[i_t]
        __cse_1 = v[i_t]
        v_stack_d[__idx_v_stack_0] = __cse_1d
        v_stack[__idx_v_stack_0] = __cse_1
        __cse_2d = xd[i_t]
        __cse_2 = x[i_t]
        vd[i_t] = __cse_1d + __cse_2d
        v[i_t] = __cse_1 + __cse_2
        __idx_v_stack_3 = max(0, div(i_n - 1, 1) + 1) + __icse_0
        __cse_3d = vd[i_t + 1]
        __cse_3 = v[i_t + 1]
        v_stack_d[__idx_v_stack_3] = __cse_3d
        v_stack[__idx_v_stack_3] = __cse_3
        vd[i_t + 1] = __cse_3d + 2.0__cse_2d
        v[i_t + 1] = __cse_3 + __cse_2 * 2.0
        if v[i_t] > 0.0
            __idx_branch_stack_0 = (i_t - 1) + 1
            branch_stack[__idx_branch_stack_0] = 1
            zd[i_t] = 3.0 * xd[i_t]
            z[i_t] = x[i_t] * 3.0
        else
            __idx_branch_stack_0 = (i_t - 1) + 1
            branch_stack[__idx_branch_stack_0] = 0
            zd[i_t] = xd[i_t]
            z[i_t] = x[i_t]
        end
    end
    for i_t = i_n:-1:1
        __idx_branch_stack_0 = (i_t - 1) + 1
        __branch = branch_stack[__idx_branch_stack_0]
        if __branch == 1
            __oldb_0d = zbd[i_t]
            __oldb_0 = zb[i_t]
            zbd[i_t] = 0.0
            zb[i_t] = 0.0
            xbd[i_t] = xbd[i_t] + 3.0__oldb_0d
            xb[i_t] = xb[i_t] + 3.0__oldb_0
        else
            __oldb_0d = zbd[i_t]
            __oldb_0 = zb[i_t]
            zbd[i_t] = 0.0
            zb[i_t] = 0.0
            xbd[i_t] = xbd[i_t] + __oldb_0d
            xb[i_t] = xb[i_t] + __oldb_0
        end
        __icse_4 = (i_t - 1) + 1
        __idx_v_stack_0 = max(0, div(i_n - 1, 1) + 1) + __icse_4
        vd[i_t + 1] = v_stack_d[__idx_v_stack_0]
        v[i_t + 1] = v_stack[__idx_v_stack_0]
        xbd[i_t] = xbd[i_t] + 2.0 * vbd[i_t + 1]
        xb[i_t] = xb[i_t] + 2.0 * vb[i_t + 1]
        __idx_v_stack_0 = __icse_4
        vd[i_t] = v_stack_d[__idx_v_stack_0]
        v[i_t] = v_stack[__idx_v_stack_0]
        xbd[i_t] = xbd[i_t] + vbd[i_t]
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
