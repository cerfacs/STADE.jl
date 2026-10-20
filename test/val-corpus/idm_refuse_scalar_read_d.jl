function idm_refuse_scalar_read_d(v, vd, z, zd, x, xd, i_n)
    for i_t = 1:i_n
        __cse_0 = xd[i_t]
        vd[i_t] = vd[i_t] + __cse_0
        __cse_1 = x[i_t]
        v[i_t] = v[i_t] + __cse_1
        vd[i_t + 1] = vd[i_t + 1] + 2.0__cse_0
        v[i_t + 1] = v[i_t + 1] + __cse_1 * 2.0
        sd = vd[i_t]
        s = v[i_t]
        zd[i_t] = 3.0sd
        z[i_t] = s * 3.0
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
