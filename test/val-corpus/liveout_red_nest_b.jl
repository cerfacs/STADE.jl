function initstacks_liveout_red_nest_b(i_n)
    return nothing
end

function liveout_red_nest_b(x, xb, w, wb, z, zb, loss, lossb, i_n, i_nc)
    a = 0.0
    xv2 = 0.0
    ab = 0.0
    xv2b = 0.0
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv2 = x[(t + ci) - 1]
            a = a + xv2 * xv2 * w[ci]
        end
        z[t] = a
    end
    for m = 1:i_n
        __cse_0 = z[m]
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    for m = i_n:-1:1
        __cse_1 = z[m] * lossb[1]
        zb[m] = zb[m] + __cse_1
        zb[m] = zb[m] + __cse_1
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv2 = x[(t + ci) - 1]
            a = a + xv2 * xv2 * w[ci]
        end
        __oldb_0 = zb[t]
        zb[t] = 0.0
        ab = ab + __oldb_0
        for ci = 1:i_nc
            __cse_2 = x[(t + ci) - 1]
            xv2 = __cse_2
            __cse_3 = w[ci]
            a = a + xv2 * xv2 * __cse_3
            xv2 = __cse_2
            __cse_4 = (xv2 * __cse_3) * ab
            xv2b = xv2b + __cse_4
            xv2b = xv2b + __cse_4
            wb[ci] = wb[ci] + (xv2 * xv2) * ab
            __oldb_0 = xv2b
            xv2b = 0.0
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + __oldb_0
        end
        ab = 0.0
    end
    return nothing
end

function liveout_red_nest(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv2 = x[(t + ci) - 1]
            a = a + xv2 * xv2 * w[ci]
        end
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
