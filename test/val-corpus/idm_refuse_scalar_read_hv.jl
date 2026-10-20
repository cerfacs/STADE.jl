function initstacks_idm_refuse_scalar_read_b(i_n)
    s_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
    return s_stack
end

function idm_refuse_scalar_read_hv(v, vb, z, zb, x, xb, i_n, vd, vbd, zd, zbd, xd, xbd, s_stack)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    for i_t = 1:i_n
        __cse_0d = xd[i_t]
        __cse_0 = x[i_t]
        vd[i_t] = vd[i_t] + __cse_0d
        v[i_t] = v[i_t] + __cse_0
        vd[i_t + 1] = vd[i_t + 1] + 2.0__cse_0d
        v[i_t + 1] = v[i_t + 1] + __cse_0 * 2.0
        __idx_s_stack_2 = (i_t - 1) + 1
        s_stack_d[__idx_s_stack_2] = sd
        s_stack[__idx_s_stack_2] = s
        sd = vd[i_t]
        s = v[i_t]
        zd[i_t] = 3.0sd
        z[i_t] = s * 3.0
    end
    __ihcse_0 = max(0, div(i_n - 1, 1) + 1) + 1
    __idx_s_stack_1 = __ihcse_0
    s_stack_d[__idx_s_stack_1] = sd
    s_stack[__idx_s_stack_1] = s
    __idx_s_stack_0 = __ihcse_0
    sd = s_stack_d[__idx_s_stack_0]
    s = s_stack[__idx_s_stack_0]
    for i_t = i_n:-1:1
        __oldb_0d = zbd[i_t]
        __oldb_0 = zb[i_t]
        zbd[i_t] = 0.0
        zb[i_t] = 0.0
        sbd = sbd + 3.0__oldb_0d
        sb = sb + 3.0__oldb_0
        __idx_s_stack_0 = (i_t - 1) + 1
        sd = s_stack_d[__idx_s_stack_0]
        s = s_stack[__idx_s_stack_0]
        __oldb_2d = sbd
        __oldb_2 = sb
        sbd = 0.0
        sb = 0.0
        vbd[i_t] = vbd[i_t] + __oldb_2d
        vb[i_t] = vb[i_t] + __oldb_2
        xbd[i_t] = xbd[i_t] + 2.0 * vbd[i_t + 1]
        xb[i_t] = xb[i_t] + 2.0 * vb[i_t + 1]
        xbd[i_t] = xbd[i_t] + vbd[i_t]
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
