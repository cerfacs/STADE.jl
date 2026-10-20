function remat_cross_d(x, xd, w, wd, y, yd, u, ud, z, zd, loss, lossd, i_n, i_nc)
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            xvd = xd[(t + ci) - 1]
            xv = x[(t + ci) - 1]
            wvd = wd[2ci - 1]
            wv = w[2ci - 1]
            __cse_0 = (xv * wv) * wvd
            ad = ad + (((wv * wv) * xvd + __cse_0) + __cse_0)
            a = a + xv * wv * wv
            wvd = wd[2ci]
            wv = w[2ci]
            __cse_1 = (xv * wv) * wvd
            ad = ad + (((wv * wv) * xvd + __cse_1) + __cse_1)
            a = a + xv * wv * wv
        end
        yd[t] = ad
        y[t] = a
    end
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            b1d = 0.0
            b1 = ci * 2
            b2d = 0.0
            b2 = b1 + 1
            xv2d = 2.0 * xd[(t + ci) - 1]
            xv2 = x[(t + ci) - 1] * 2.0
            wv2d = wd[b2]
            wv2 = w[b2]
            __cse_2 = (xv2 * wv2) * wv2d
            ad = ad + (((wv2 * wv2) * xv2d + __cse_2) + __cse_2)
            a = a + xv2 * wv2 * wv2
        end
        zd[t] = ad
        z[t] = a
    end
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 2:2:i_nc
            xv3d = xd[(t + ci) - 1]
            xv3 = x[(t + ci) - 1]
            __cse_3 = w[ci]
            __cse_4 = (xv3 * __cse_3) * xv3d
            ad = ad + ((__cse_4 + __cse_4) + (xv3 * xv3) * wd[ci])
            a = a + xv3 * xv3 * __cse_3
        end
        for cj = i_nc:-1:1
            xv4d = xd[t + cj]
            xv4 = x[t + cj]
            __cse_5 = w[cj + 1]
            __cse_6 = (xv4 * __cse_5) * xv4d
            ad = ad + ((__cse_6 + __cse_6) + (xv4 * xv4) * wd[cj + 1])
            a = a + xv4 * xv4 * __cse_5
        end
        zd[i_n + t] = ad
        z[i_n + t] = a
    end
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        xv5d = 0.5 * xd[t]
        xv5 = x[t] * 0.5
        for ci = 1:i_nc
            __cse_7 = xv5 * xv5d
            ad = ad + (__cse_7 + __cse_7)
            a = a + xv5 * xv5
            __cse_8 = w[ci]
            __cse_9 = x[(t + ci) - 1]
            xv5d = __cse_8 * xd[(t + ci) - 1] + __cse_9 * wd[ci]
            xv5 = __cse_9 * __cse_8
            __cse_10 = w[ci + 1]
            ad = ad + (__cse_10 * xv5d + xv5 * wd[ci + 1])
            a = a + xv5 * __cse_10
        end
        zd[2i_n + t] = ad
        z[2i_n + t] = a
    end
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        __cse_11 = w[1]
        __cse_12 = x[t]
        xv6d = __cse_11 * xd[t] + __cse_12 * wd[1]
        xv6 = __cse_12 * __cse_11
        __cse_13 = xv6 * xv6d
        ad = __cse_13 + __cse_13
        a = xv6 * xv6
        for ci = 1:i_nc
            __cse_14 = w[ci]
            __cse_15 = x[t + ci]
            xv6d = __cse_14 * xd[t + ci] + __cse_15 * wd[ci]
            xv6 = __cse_15 * __cse_14
            __cse_16 = (xv6 * xv6) * xv6d
            ad = ad + ((__cse_16 + __cse_16) + __cse_16)
            a = a + xv6 * xv6 * xv6
        end
        zd[3i_n + t] = ad
        z[3i_n + t] = a
    end
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            xv7d = xd[(t + ci) - 1]
            xv7 = x[(t + ci) - 1]
            __cse_17 = xv7 * xv7d
            ad = ad + (__cse_17 + __cse_17)
            a = a + xv7 * xv7
            for cj = 1:2
                __cse_18 = w[ci + cj]
                xv7d = __cse_18 * xv7d + xv7 * wd[ci + cj]
                xv7 = xv7 * __cse_18
            end
            __cse_19 = xv7 * xv7d
            ad = ad + (__cse_19 + __cse_19)
            a = a + xv7 * xv7
        end
        zd[4i_n + t] = ad
        z[4i_n + t] = a
    end
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            xv8d = ud[ci]
            xv8 = u[ci]
            __cse_20 = x[(t + ci) - 1]
            __cse_21 = (xv8 * __cse_20) * xv8d
            ad = ad + ((__cse_21 + __cse_21) + (xv8 * xv8) * xd[(t + ci) - 1])
            a = a + xv8 * xv8 * __cse_20
            ud[ci + 1] = 0.5 * xd[t + ci]
            u[ci + 1] = x[t + ci] * 0.5
        end
        zd[5i_n + t] = ad
        z[5i_n + t] = a
    end
    for m = 1:i_n
        __cse_22 = y[m]
        __cse_23 = __cse_22 * yd[m]
        lossd[1] = lossd[1] + (__cse_23 + __cse_23)
        loss[1] = loss[1] + __cse_22 * __cse_22
    end
    for m = 1:6i_n
        __cse_24 = z[m]
        __cse_25 = __cse_24 * zd[m]
        lossd[1] = lossd[1] + (__cse_25 + __cse_25)
        loss[1] = loss[1] + __cse_24 * __cse_24
    end
    return nothing
end

function remat_cross(x, w, y, u, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv = x[(t + ci) - 1]
            wv = w[2ci - 1]
            a = a + xv * wv * wv
            wv = w[2ci]
            a = a + xv * wv * wv
        end
        y[t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            b1 = ci * 2
            b2 = b1 + 1
            xv2 = x[(t + ci) - 1] * 2.0
            wv2 = w[b2]
            a = a + xv2 * wv2 * wv2
        end
        z[t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 2:2:i_nc
            xv3 = x[(t + ci) - 1]
            a = a + xv3 * xv3 * w[ci]
        end
        for cj = i_nc:-1:1
            xv4 = x[t + cj]
            a = a + xv4 * xv4 * w[cj + 1]
        end
        z[i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        xv5 = x[t] * 0.5
        for ci = 1:i_nc
            a = a + xv5 * xv5
            xv5 = x[(t + ci) - 1] * w[ci]
            a = a + xv5 * w[ci + 1]
        end
        z[2i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        xv6 = x[t] * w[1]
        a = xv6 * xv6
        for ci = 1:i_nc
            xv6 = x[t + ci] * w[ci]
            a = a + xv6 * xv6 * xv6
        end
        z[3i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv7 = x[(t + ci) - 1]
            a = a + xv7 * xv7
            for cj = 1:2
                xv7 = xv7 * w[ci + cj]
            end
            a = a + xv7 * xv7
        end
        z[4i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv8 = u[ci]
            a = a + xv8 * xv8 * x[(t + ci) - 1]
            u[ci + 1] = x[t + ci] * 0.5
        end
        z[5i_n + t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + y[m] * y[m]
    end
    for m = 1:6i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
