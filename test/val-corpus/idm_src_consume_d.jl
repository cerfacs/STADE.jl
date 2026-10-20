function idm_src_consume_d(u, ud, y, yd, i_n)
    for i_t = 1:i_n
        i_bd = 0.0
        i_b = 2 * (i_t - 1)
        sd = ud[i_b + 1]
        s = u[i_b + 1]
        ud[i_b + 1] = 0.0
        u[i_b + 1] = 0.0
        rd = ud[i_b + 2]
        r = u[i_b + 2]
        ud[i_b + 2] = 0.0
        u[i_b + 2] = 0.0
        yd[i_t] = r * sd + s * rd
        y[i_t] = s * r
    end
    return nothing
end

function idm_src_consume(u, y, i_n)
    for i_t = 1:i_n
        i_b = 2 * (i_t - 1)
        s = u[i_b + 1]
        u[i_b + 1] = 0.0
        r = u[i_b + 2]
        u[i_b + 2] = 0.0
        y[i_t] = s * r
    end
end
