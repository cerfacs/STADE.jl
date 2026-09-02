function initstacks_ii_kill_b()
    v_stack = Vector{Float64}(undef, 1)
    return v_stack
end

function ii_kill_hv(x, xb, u, ub, i_n, i_w0, out, outb, acc, accb, xd, xbd, ud, ubd, outd, outbd, accd, accbd, v_stack)
    v_stack_d = Vector{Float64}(undef, length(v_stack))
    v = 0.0
    vb = 0.0
    vd = 0.0
    vbd = 0.0
    vd = 0.0
    v = 0.0
    for i_i = 1:i_n
        __cse_0d = xd[i_i]
        __cse_0 = x[i_i]
        __hcse_0 = __cse_0 * __cse_0d
        vd = __hcse_0 + __hcse_0
        v = __cse_0 * __cse_0
        accd[i_i] = accd[i_i] + vd
        acc[i_i] = acc[i_i] + v
    end
    w = i_w0 - 5
    for i_j = 1:w
        __cse_1d = ud[i_j]
        __cse_1 = u[i_j]
        __hcse_1 = __cse_1 * __cse_1d
        vd = __hcse_1 + __hcse_1
        v = __cse_1 * __cse_1
    end
    __hcse_2 = v * vd
    outd[1] = outd[1] + (__hcse_2 + __hcse_2)
    out[1] = out[1] + v * v
    v_stack_d[1] = vd
    v_stack[1] = v
    vd = v_stack_d[1]
    v = v_stack[1]
    w = i_w0 - 5
    __hcse_3 = outb[1]
    __cse_2d = __hcse_3 * vd + v * outbd[1]
    __cse_2 = v * __hcse_3
    vbd = vbd + __cse_2d
    vb = vb + __cse_2
    vbd = vbd + __cse_2d
    vb = vb + __cse_2
    for i_j = w:-1:1
        __oldb_0d = vbd
        __oldb_0 = vb
        vbd = 0.0
        vb = 0.0
        __hcse_4 = u[i_j]
        __cse_3d = __oldb_0 * ud[i_j] + __hcse_4 * __oldb_0d
        __cse_3 = __hcse_4 * __oldb_0
        ubd[i_j] = ubd[i_j] + __cse_3d
        ub[i_j] = ub[i_j] + __cse_3
        ubd[i_j] = ubd[i_j] + __cse_3d
        ub[i_j] = ub[i_j] + __cse_3
    end
    for i_i = i_n:-1:1
        vbd = vbd + accbd[i_i]
        vb = vb + accb[i_i]
        __oldb_0d = vbd
        __oldb_0 = vb
        vbd = 0.0
        vb = 0.0
        __hcse_5 = x[i_i]
        __cse_4d = __oldb_0 * xd[i_i] + __hcse_5 * __oldb_0d
        __cse_4 = __hcse_5 * __oldb_0
        xbd[i_i] = xbd[i_i] + __cse_4d
        xb[i_i] = xb[i_i] + __cse_4
        xbd[i_i] = xbd[i_i] + __cse_4d
        xb[i_i] = xb[i_i] + __cse_4
    end
    vbd = 0.0
    vb = 0.0
    return nothing
end

function ii_kill(x, u, i_n, i_w0, out, acc)
    v = 0.0
    for i_i = 1:i_n
        v = x[i_i] * x[i_i]
        acc[i_i] = acc[i_i] + v
    end
    w = i_w0 - 5
    for i_j = 1:w
        v = u[i_j] * u[i_j]
    end
    out[1] = out[1] + v * v
end
