function ii_kill_d(x, xd, u, ud, i_n, i_w0, out, outd, acc, accd)
    vd = 0.0
    v = 0.0
    for i_i = 1:i_n
        vd = x[i_i] * xd[i_i] + x[i_i] * xd[i_i]
        v = x[i_i] * x[i_i]
        accd[i_i] = accd[i_i] + vd
        acc[i_i] = acc[i_i] + v
    end
    wd = 0.0
    w = i_w0 - 5
    for i_j = 1:w
        vd = u[i_j] * ud[i_j] + u[i_j] * ud[i_j]
        v = u[i_j] * u[i_j]
    end
    outd[1] = outd[1] + (v * vd + v * vd)
    out[1] = out[1] + v * v
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
