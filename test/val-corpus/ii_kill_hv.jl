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
        vd = x[i_i] * xd[i_i] + x[i_i] * xd[i_i]
        v = x[i_i] * x[i_i]
        accd[i_i] = accd[i_i] + vd
        acc[i_i] = acc[i_i] + v
    end
    w = i_w0 - 5
    for i_j = 1:w
        vd = u[i_j] * ud[i_j] + u[i_j] * ud[i_j]
        v = u[i_j] * u[i_j]
    end
    outd[1] = outd[1] + (v * vd + v * vd)
    out[1] = out[1] + v * v
    v_stack_d[1] = vd
    v_stack[1] = v
    vd = v_stack_d[1]
    v = v_stack[1]
    w = i_w0 - 5
    vbd = vbd + (outb[1] * vd + v * outbd[1])
    vb = vb + v * outb[1]
    vbd = vbd + (outb[1] * vd + v * outbd[1])
    vb = vb + v * outb[1]
    for i_j = w:-1:1
        ubd[i_j] = ubd[i_j] + (vb * ud[i_j] + u[i_j] * vbd)
        ub[i_j] = ub[i_j] + u[i_j] * vb
        ubd[i_j] = ubd[i_j] + (vb * ud[i_j] + u[i_j] * vbd)
        ub[i_j] = ub[i_j] + u[i_j] * vb
        vbd = 0.0
        vb = 0.0
    end
    for i_i = i_n:-1:1
        vbd = vbd + accbd[i_i]
        vb = vb + accb[i_i]
        xbd[i_i] = xbd[i_i] + (vb * xd[i_i] + x[i_i] * vbd)
        xb[i_i] = xb[i_i] + x[i_i] * vb
        xbd[i_i] = xbd[i_i] + (vb * xd[i_i] + x[i_i] * vbd)
        xb[i_i] = xb[i_i] + x[i_i] * vb
        vbd = 0.0
        vb = 0.0
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
