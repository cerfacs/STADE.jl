function idm_refuse_consume_gap_d(u, ud, x, xd, i_n)
    for i_t = 1:i_n
        sd = ud[i_t]
        s = u[i_t]
        __cse_0 = x[i_t]
        ud[i_t + 1] = __cse_0 * sd + s * xd[i_t]
        u[i_t + 1] = s * __cse_0 + 1.0
        ud[i_t] = 0.0
        u[i_t] = 0.0
    end
    return nothing
end

function idm_refuse_consume_gap(u, x, i_n)
    for i_t = 1:i_n
        s = u[i_t]
        u[i_t + 1] = s * x[i_t] + 1.0
        u[i_t] = 0.0
    end
end
