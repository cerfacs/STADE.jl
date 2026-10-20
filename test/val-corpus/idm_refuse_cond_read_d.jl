function idm_refuse_cond_read_d(v, vd, z, zd, x, xd, i_n)
    for i_t = 1:i_n
        __cse_0 = xd[i_t]
        vd[i_t] = vd[i_t] + __cse_0
        __cse_1 = x[i_t]
        v[i_t] = v[i_t] + __cse_1
        vd[i_t + 1] = vd[i_t + 1] + 2.0__cse_0
        v[i_t + 1] = v[i_t + 1] + __cse_1 * 2.0
        if v[i_t] > 0.0
            zd[i_t] = 3.0 * xd[i_t]
            z[i_t] = x[i_t] * 3.0
        else
            zd[i_t] = xd[i_t]
            z[i_t] = x[i_t]
        end
    end
    return nothing
end

function idm_refuse_cond_read(v, z, x, i_n)
    for i_t = 1:i_n
        v[i_t] = v[i_t] + x[i_t]
        v[i_t + 1] = v[i_t + 1] + x[i_t] * 2.0
        if v[i_t] > 0.0
            z[i_t] = x[i_t] * 3.0
        else
            z[i_t] = x[i_t]
        end
    end
end
