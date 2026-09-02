function cse_zerotrip_d(out, outd, u, ud, w, wd, i_npass, i_w0)
    __cse_0 = w[1]
    __cse_1 = u[1]
    sd = __cse_0 * ud[1] + __cse_1 * wd[1]
    s = __cse_1 * __cse_0
    widthd = 0.0
    width = i_w0
    for i_p = 1:i_npass
        __cse_2 = w[1]
        __cse_3 = u[1]
        __cse_4 = ud[1]
        sd = sd + (((__cse_2 * __cse_3) * __cse_4 + (__cse_3 * __cse_3) * wd[1]) + (__cse_3 * __cse_2) * __cse_4)
        s = s + __cse_3 * __cse_2 * __cse_3
        for i_j = 1:width
            __cse_5 = w[i_j]
            __cse_6 = u[i_j]
            sd = sd + (__cse_5 * ud[i_j] + __cse_6 * wd[i_j])
            s = s + __cse_6 * __cse_5
        end
        __cse_7 = u[1]
        __cse_8 = w[1]
        __cse_9 = __cse_7 * __cse_8
        outd[i_p] = __cse_9 * sd + s * (__cse_8 * ud[1] + __cse_7 * wd[1])
        out[i_p] = s * __cse_9
        widthd = 0.0
        width = width - 3
    end
    return nothing
end

function cse_zerotrip(out, u, w, i_npass, i_w0)
    s = u[1] * w[1]
    width = i_w0
    for i_p = 1:i_npass
        s = s + u[1] * w[1] * u[1]
        for i_j = 1:width
            s = s + u[i_j] * w[i_j]
        end
        out[i_p] = s * (u[1] * w[1])
        width = width - 3
    end
end
