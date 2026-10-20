function asnap_direct_d(x, xd, w, wd, node, m, md, i_njac)
    for i_ = 1:i_njac
        n0d = 0.0
        n0 = node[1]
        __cse_0 = x[1]
        __cse_1 = w[1]
        __cse_2 = xd[1]
        md[n0] = md[n0] + ((__cse_0 * 0.1) * wd[1] + (__cse_1 * 0.1) * __cse_2)
        m[n0] = m[n0] + __cse_1 * __cse_0 * 0.1
        __cse_3 = m[n0]
        __cse_4 = (__cse_3 * 0.5) * md[n0]
        xd[1] = __cse_2 + (__cse_4 + __cse_4)
        x[1] = __cse_0 + __cse_3 * __cse_3 * 0.5
    end
    return nothing
end

function asnap_direct(x, w, node, m, i_njac)
    for i_ = 1:i_njac
        n0 = node[1]
        m[n0] = m[n0] + w[1] * x[1] * 0.1
        x[1] = x[1] + m[n0] * m[n0] * 0.5
    end
end
