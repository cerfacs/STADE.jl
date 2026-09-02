function cse_branch_d(out, outd, u, ud, v, vd, i_flag, i_n)
    for i_r = 1:i_n
        sd = 0.0
        s = 0.0
        if i_flag[i_r] > 0
            __cse_0 = v[i_r]
            __cse_1 = ud[i_r]
            __cse_2 = u[i_r]
            __cse_3 = vd[i_r]
            __cse_4 = __cse_2 * __cse_0
            __cse_5 = __cse_4 * __cse_3
            sd = (__cse_0 * __cse_1 + __cse_2 * __cse_3) + (((__cse_0 * __cse_0) * __cse_1 + __cse_5) + __cse_5)
            s = __cse_4 + __cse_2 * __cse_0 * __cse_0
        else
            __cse_6 = v[i_r]
            __cse_7 = ud[i_r]
            __cse_8 = u[i_r]
            __cse_9 = vd[i_r]
            __cse_10 = __cse_8 * __cse_6
            sd = (__cse_6 * __cse_7 + __cse_8 * __cse_9) + -((((__cse_6 * __cse_8) * __cse_7 + (__cse_8 * __cse_8) * __cse_9) + __cse_10 * __cse_7))
            s = __cse_10 - __cse_8 * __cse_6 * __cse_8
        end
        __cse_11 = u[i_r]
        __cse_12 = v[i_r]
        __cse_13 = __cse_11 * __cse_12
        outd[i_r] = __cse_13 * sd + s * (__cse_12 * ud[i_r] + __cse_11 * vd[i_r])
        out[i_r] = s * __cse_13
    end
    return nothing
end

function cse_branch(out, u, v, i_flag, i_n)
    for i_r = 1:i_n
        s = 0.0
        if i_flag[i_r] > 0
            s = u[i_r] * v[i_r] + u[i_r] * v[i_r] * v[i_r]
        else
            s = u[i_r] * v[i_r] - u[i_r] * v[i_r] * u[i_r]
        end
        out[i_r] = s * (u[i_r] * v[i_r])
    end
end
