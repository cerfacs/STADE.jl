function initstacks_red_two_locals_b()
    p_stack = Vector{Float64}(undef, 1)
    q_stack = Vector{Float64}(undef, 1)
    return (p_stack, q_stack)
end

function red_two_locals_b(loss, lossb, u, ub, v, vb, i_n, p_stack, q_stack)
    p = 0.0
    q = 0.0
    pb = 0.0
    qb = 0.0
    p = 0.0
    for i_x = 1:i_n
        p = p + u[i_x] * v[i_x]
    end
    q = 0.0
    for i_y = 1:i_n
        __cse_0 = v[i_y]
        q = q + __cse_0 * __cse_0
    end
    loss[1] = loss[1] + p * q
    p_stack[1] = p
    q_stack[1] = q
    p = p_stack[1]
    q = q_stack[1]
    __cse_1 = lossb[1]
    pb = pb + q * __cse_1
    qb = qb + p * __cse_1
    for i_y = 1:i_n
        __cse_2 = v[i_y]
        q = q + __cse_2 * __cse_2
        __cse_3 = __cse_2 * qb
        vb[i_y] = vb[i_y] + __cse_3
        vb[i_y] = vb[i_y] + __cse_3
    end
    qb = 0.0
    for i_x = 1:i_n
        __cse_4 = u[i_x]
        __cse_5 = v[i_x]
        p = p + __cse_4 * __cse_5
        ub[i_x] = ub[i_x] + __cse_5 * pb
        vb[i_x] = vb[i_x] + __cse_4 * pb
    end
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
