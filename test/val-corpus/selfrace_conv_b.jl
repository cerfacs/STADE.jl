function initstacks_selfrace_conv_b()
    return nothing
end

function selfrace_conv_b(x, xb, w, wb, z, zb, loss, lossb, i_n, i_nc)
    a = 0.0
    ab = 0.0
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            a = a + x[(t + ci) - 1] * w[ci]
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
    for t = i_n:-1:1
        __oldb_0 = zb[t]
        zb[t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + w[ci] * ab
            wb[ci] = wb[ci] + x[(t + ci) - 1] * ab
        end
        ab = 0.0
    end
    return nothing
end

function selfrace_conv(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            a = a + x[(t + ci) - 1] * w[ci]
        end
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
