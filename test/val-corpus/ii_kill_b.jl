function initstacks_ii_kill_b()
    v_stack = Vector{Float64}(undef, 1)
    return v_stack
end

function ii_kill_b(x, xb, u, ub, i_n, i_w0, out, outb, acc, accb, v_stack)
    v = 0.0
    vb = 0.0
    v = 0.0
    for i_i = 1:i_n
        __cse_0 = x[i_i]
        v = __cse_0 * __cse_0
        acc[i_i] = acc[i_i] + v
    end
    w = i_w0 - 5
    for i_j = 1:w
        __cse_1 = u[i_j]
        v = __cse_1 * __cse_1
    end
    out[1] = out[1] + v * v
    v_stack[1] = v
    v = v_stack[1]
    w = i_w0 - 5
    __cse_2 = v * outb[1]
    vb = vb + __cse_2
    vb = vb + __cse_2
    for i_j = w:-1:1
        __cse_3 = u[i_j] * vb
        ub[i_j] = ub[i_j] + __cse_3
        ub[i_j] = ub[i_j] + __cse_3
        vb = 0.0
    end
    for i_i = i_n:-1:1
        vb = vb + accb[i_i]
        __cse_4 = x[i_i] * vb
        xb[i_i] = xb[i_i] + __cse_4
        xb[i_i] = xb[i_i] + __cse_4
        vb = 0.0
    end
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
