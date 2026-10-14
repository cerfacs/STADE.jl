function initstacks_red_two_locals_b()
    p_stack = Vector{Float64}(undef, 1)
    q_stack = Vector{Float64}(undef, 1)
    return (p_stack, q_stack)
end

function red_two_locals_hv(loss, lossb, u, ub, v, vb, i_n, lossd, lossbd, ud, ubd, vd, vbd, p_stack, q_stack)
    p_stack_d = Vector{Float64}(undef, length(p_stack))
    q_stack_d = Vector{Float64}(undef, length(q_stack))
    p = 0.0
    q = 0.0
    pb = 0.0
    qb = 0.0
    pd = 0.0
    pbd = 0.0
    qd = 0.0
    qbd = 0.0
    pd = 0.0
    p = 0.0
    for i_x = 1:i_n
        __hcse_0 = v[i_x]
        __hcse_1 = u[i_x]
        pd = pd + (__hcse_0 * ud[i_x] + __hcse_1 * vd[i_x])
        p = p + __hcse_1 * __hcse_0
    end
    qd = 0.0
    q = 0.0
    for i_y = 1:i_n
        __cse_0d = vd[i_y]
        __cse_0 = v[i_y]
        __hcse_2 = __cse_0 * __cse_0d
        qd = qd + (__hcse_2 + __hcse_2)
        q = q + __cse_0 * __cse_0
    end
    lossd[1] = lossd[1] + (q * pd + p * qd)
    loss[1] = loss[1] + p * q
    p_stack_d[1] = pd
    p_stack[1] = p
    q_stack_d[1] = qd
    q_stack[1] = q
    pd = p_stack_d[1]
    p = p_stack[1]
    qd = q_stack_d[1]
    q = q_stack[1]
    __cse_1d = lossbd[1]
    __cse_1 = lossb[1]
    pbd = pbd + (__cse_1 * qd + q * __cse_1d)
    pb = pb + q * __cse_1
    qbd = qbd + (__cse_1 * pd + p * __cse_1d)
    qb = qb + p * __cse_1
    for i_y = 1:i_n
        __cse_2d = vd[i_y]
        __cse_2 = v[i_y]
        __hcse_3 = __cse_2 * __cse_2d
        qd = qd + (__hcse_3 + __hcse_3)
        q = q + __cse_2 * __cse_2
        __cse_3d = qb * __cse_2d + __cse_2 * qbd
        __cse_3 = __cse_2 * qb
        vbd[i_y] = vbd[i_y] + __cse_3d
        vb[i_y] = vb[i_y] + __cse_3
        vbd[i_y] = vbd[i_y] + __cse_3d
        vb[i_y] = vb[i_y] + __cse_3
    end
    qbd = 0.0
    qb = 0.0
    for i_x = 1:i_n
        __cse_4d = ud[i_x]
        __cse_4 = u[i_x]
        __cse_5d = vd[i_x]
        __cse_5 = v[i_x]
        pd = pd + (__cse_5 * __cse_4d + __cse_4 * __cse_5d)
        p = p + __cse_4 * __cse_5
        ubd[i_x] = ubd[i_x] + (pb * __cse_5d + __cse_5 * pbd)
        ub[i_x] = ub[i_x] + __cse_5 * pb
        vbd[i_x] = vbd[i_x] + (pb * __cse_4d + __cse_4 * pbd)
        vb[i_x] = vb[i_x] + __cse_4 * pb
    end
    pbd = 0.0
    pb = 0.0
    return nothing
end

function red_two_locals(loss, u, v, i_n)
    p = 0.0
    for i_x = 1:i_n
        p = p + u[i_x] * v[i_x]
    end
    q = 0.0
    for i_y = 1:i_n
        q = q + v[i_y] * v[i_y]
    end
    loss[1] = loss[1] + p * q
end
