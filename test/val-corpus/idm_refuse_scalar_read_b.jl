function initstacks_idm_refuse_scalar_read_b(i_n)
    s_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
    return s_stack
end

function idm_refuse_scalar_read_b(v, vb, z, zb, x, xb, i_n, s_stack)
    s = 0.0
    sb = 0.0
    for i_t = 1:i_n
        __cse_0 = x[i_t]
        v[i_t] = v[i_t] + __cse_0
        v[i_t + 1] = v[i_t + 1] + __cse_0 * 2.0
        __idx_s_stack_2 = (i_t - 1) + 1
        s_stack[__idx_s_stack_2] = s
        s = v[i_t]
        z[i_t] = s * 3.0
    end
    __icse_1 = max(0, div(i_n - 1, 1) + 1) + 1
    __idx_s_stack_1 = __icse_1
    s_stack[__idx_s_stack_1] = s
    __idx_s_stack_0 = __icse_1
    s = s_stack[__idx_s_stack_0]
    for i_t = i_n:-1:1
        __oldb_0 = zb[i_t]
        zb[i_t] = 0.0
        sb = sb + 3.0__oldb_0
        __idx_s_stack_0 = (i_t - 1) + 1
        s = s_stack[__idx_s_stack_0]
        __oldb_2 = sb
        sb = 0.0
        vb[i_t] = vb[i_t] + __oldb_2
        xb[i_t] = xb[i_t] + 2.0 * vb[i_t + 1]
        xb[i_t] = xb[i_t] + vb[i_t]
    end
    return nothing
end

function idm_refuse_scalar_read(v, z, x, i_n)
    for i_t = 1:i_n
        v[i_t] = v[i_t] + x[i_t]
        v[i_t + 1] = v[i_t + 1] + x[i_t] * 2.0
        s = v[i_t]
        z[i_t] = s * 3.0
    end
end
