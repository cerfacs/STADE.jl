function remat_shapes_d(x, xd, w, wd, y, yd, z, zd, u, ud, loss, lossd, s, sd, i_n, i_nc)
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            based = 0.0
            base = (ci - 1) * 2
            wvd = wd[base + 1]
            wv = w[base + 1]
            __cse_0 = x[(t + ci) - 1]
            __cse_1 = (wv * __cse_0) * wvd
            ad = ad + ((__cse_1 + __cse_1) + (wv * wv) * xd[(t + ci) - 1])
            a = a + wv * wv * __cse_0
            wvd = wd[base + 2]
            wv = w[base + 2]
            __cse_2 = x[t + ci]
            __cse_3 = (wv * __cse_2) * wvd
            ad = ad + ((__cse_3 + __cse_3) + (wv * wv) * xd[t + ci])
            a = a + wv * wv * __cse_2
        end
        yd[t] = ad
        y[t] = a
    end
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            wvd = wd[ci]
            wv = w[ci]
            __cse_4 = x[(t + ci) - 1]
            __cse_5 = (wv * __cse_4) * wvd
            ad = ad + ((__cse_5 + __cse_5) + (wv * wv) * xd[(t + ci) - 1])
            a = a + wv * wv * __cse_4
            wvd = 0.5wvd
            wv = wv * 0.5
            __cse_6 = x[t + ci]
            __cse_7 = (wv * __cse_6) * wvd
            __cse_8 = xd[t + ci]
            ad = ad + ((__cse_7 + __cse_7) + (wv * wv) * __cse_8)
            a = a + wv * wv * __cse_6
            wvd = wd[ci + 1]
            wv = w[ci + 1]
            __cse_9 = (wv * __cse_6) * wvd
            ad = ad + ((__cse_9 + __cse_9) + (wv * wv) * __cse_8)
            a = a + wv * wv * __cse_6
        end
        zd[t] = ad
        z[t] = a
    end
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            __cse_10 = yd[t]
            wvd = __cse_10
            __cse_11 = y[t]
            wv = __cse_11
            __cse_12 = x[(t + ci) - 1]
            __cse_13 = (wv * __cse_12) * wvd
            ad = ad + ((__cse_13 + __cse_13) + (wv * wv) * xd[(t + ci) - 1])
            a = a + wv * wv * __cse_12
            wvd = 0.5__cse_10
            wv = __cse_11 * 0.5
            __cse_14 = x[t + ci]
            __cse_15 = (wv * __cse_14) * wvd
            ad = ad + ((__cse_15 + __cse_15) + (wv * wv) * xd[t + ci])
            a = a + wv * wv * __cse_14
        end
        zd[i_n + t] = ad
        z[i_n + t] = a
    end
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            wvd = wd[ci]
            wv = w[ci]
            for j = 1:2
                __cse_16 = w[ci + j]
                wvd = __cse_16 * wvd + wv * wd[ci + j]
                wv = wv * __cse_16
            end
            __cse_17 = x[(t + ci) - 1]
            __cse_18 = (wv * __cse_17) * wvd
            ad = ad + ((__cse_18 + __cse_18) + (wv * wv) * xd[(t + ci) - 1])
            a = a + wv * wv * __cse_17
            wvd = wd[ci + 1]
            wv = w[ci + 1]
            __cse_19 = x[t + ci]
            __cse_20 = (wv * __cse_19) * wvd
            ad = ad + ((__cse_20 + __cse_20) + (wv * wv) * xd[t + ci])
            a = a + wv * wv * __cse_19
        end
        zd[2i_n + t] = ad
        z[2i_n + t] = a
    end
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            wvd = 0.0
            wv = 0.0
            if x[(t + ci) - 1] > 0.0
                wvd = wd[ci]
                wv = w[ci]
            end
            __cse_21 = x[(t + ci) - 1]
            __cse_22 = (wv * __cse_21) * wvd
            ad = ad + ((__cse_22 + __cse_22) + (wv * wv) * xd[(t + ci) - 1])
            a = a + wv * wv * __cse_21
            wvd = wd[ci + 1]
            wv = w[ci + 1]
            __cse_23 = x[t + ci]
            __cse_24 = (wv * __cse_23) * wvd
            ad = ad + ((__cse_24 + __cse_24) + (wv * wv) * xd[t + ci])
            a = a + wv * wv * __cse_23
        end
        zd[3i_n + t] = ad
        z[3i_n + t] = a
    end
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            scd = 2.0 * xd[ci]
            sc = x[ci] * 2.0
            __cse_25 = w[ci]
            wvd = sc * wd[ci] + __cse_25 * scd
            wv = __cse_25 * sc
            __cse_26 = x[(t + ci) - 1]
            __cse_27 = (wv * __cse_26) * wvd
            ad = ad + ((__cse_27 + __cse_27) + (wv * wv) * xd[(t + ci) - 1])
            a = a + wv * wv * __cse_26
            __cse_28 = w[ci + 1]
            wvd = sc * wd[ci + 1] + __cse_28 * scd
            wv = __cse_28 * sc
            __cse_29 = x[t + ci]
            __cse_30 = (wv * __cse_29) * wvd
            ad = ad + ((__cse_30 + __cse_30) + (wv * wv) * xd[t + ci])
            a = a + wv * wv * __cse_29
        end
        zd[4i_n + t] = ad
        z[4i_n + t] = a
    end
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            __cse_31 = w[ci]
            wvd = s * wd[ci] + __cse_31 * sd
            wv = __cse_31 * s
            __cse_32 = x[(t + ci) - 1]
            __cse_33 = (wv * __cse_32) * wvd
            ad = ad + ((__cse_33 + __cse_33) + (wv * wv) * xd[(t + ci) - 1])
            a = a + wv * wv * __cse_32
            __cse_34 = w[ci + 1]
            wvd = s * wd[ci + 1] + __cse_34 * sd
            wv = __cse_34 * s
            __cse_35 = x[t + ci]
            __cse_36 = (wv * __cse_35) * wvd
            ad = ad + ((__cse_36 + __cse_36) + (wv * wv) * xd[t + ci])
            a = a + wv * wv * __cse_35
        end
        zd[5i_n + t] = ad
        z[5i_n + t] = a
    end
    for t = 1:i_n
        __cse_37 = ud[t]
        wvd = __cse_37
        __cse_38 = u[t]
        wv = __cse_38
        __cse_39 = xd[t]
        ud[t] = 0.5__cse_37 + __cse_39
        __cse_40 = x[t]
        u[t] = __cse_38 * 0.5 + __cse_40
        __cse_41 = (wv * __cse_40) * wvd
        ad = (__cse_41 + __cse_41) + (wv * wv) * __cse_39
        a = wv * wv * __cse_40
        wvd = ud[t]
        wv = u[t]
        __cse_42 = x[t + 1]
        __cse_43 = (wv * __cse_42) * wvd
        ad = ad + ((__cse_43 + __cse_43) + (wv * wv) * xd[t + 1])
        a = a + wv * wv * __cse_42
        zd[6i_n + t] = ad
        z[6i_n + t] = a
    end
    for t = 1:i_n
        __cse_44 = xd[t]
        __cse_45 = xd[t + 1]
        ud[1] = 0.5__cse_44 + __cse_45
        __cse_46 = x[t]
        __cse_47 = x[t + 1]
        u[1] = __cse_46 * 0.5 + __cse_47
        __cse_48 = ud[1]
        wvd = __cse_48
        __cse_49 = u[1]
        wv = __cse_49
        __cse_50 = (wv * __cse_46) * wvd
        ad = (__cse_50 + __cse_50) + (wv * wv) * __cse_44
        a = wv * wv * __cse_46
        wvd = 2.0__cse_48
        wv = __cse_49 * 2.0
        __cse_51 = (wv * __cse_47) * wvd
        ad = ad + ((__cse_51 + __cse_51) + (wv * wv) * __cse_45)
        a = a + wv * wv * __cse_47
        zd[7i_n + t] = ad
        z[7i_n + t] = a
    end
    for m = 1:8i_n
        __cse_52 = z[m]
        __cse_53 = __cse_52 * zd[m]
        lossd[1] = lossd[1] + (__cse_53 + __cse_53)
        loss[1] = loss[1] + __cse_52 * __cse_52
    end
    return nothing
end

function remat_shapes(x, w, y, z, u, loss, s, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            base = (ci - 1) * 2
            wv = w[base + 1]
            a = a + wv * wv * x[(t + ci) - 1]
            wv = w[base + 2]
            a = a + wv * wv * x[t + ci]
        end
        y[t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = w[ci]
            a = a + wv * wv * x[(t + ci) - 1]
            wv = wv * 0.5
            a = a + wv * wv * x[t + ci]
            wv = w[ci + 1]
            a = a + wv * wv * x[t + ci]
        end
        z[t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = y[t]
            a = a + wv * wv * x[(t + ci) - 1]
            wv = y[t] * 0.5
            a = a + wv * wv * x[t + ci]
        end
        z[i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = w[ci]
            for j = 1:2
                wv = wv * w[ci + j]
            end
            a = a + wv * wv * x[(t + ci) - 1]
            wv = w[ci + 1]
            a = a + wv * wv * x[t + ci]
        end
        z[2i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = 0.0
            if x[(t + ci) - 1] > 0.0
                wv = w[ci]
            end
            a = a + wv * wv * x[(t + ci) - 1]
            wv = w[ci + 1]
            a = a + wv * wv * x[t + ci]
        end
        z[3i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            sc = x[ci] * 2.0
            wv = w[ci] * sc
            a = a + wv * wv * x[(t + ci) - 1]
            wv = w[ci + 1] * sc
            a = a + wv * wv * x[t + ci]
        end
        z[4i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = w[ci] * s
            a = a + wv * wv * x[(t + ci) - 1]
            wv = w[ci + 1] * s
            a = a + wv * wv * x[t + ci]
        end
        z[5i_n + t] = a
    end
    for t = 1:i_n
        wv = u[t]
        u[t] = u[t] * 0.5 + x[t]
        a = wv * wv * x[t]
        wv = u[t]
        a = a + wv * wv * x[t + 1]
        z[6i_n + t] = a
    end
    for t = 1:i_n
        u[1] = x[t] * 0.5 + x[t + 1]
        wv = u[1]
        a = wv * wv * x[t]
        wv = u[1] * 2.0
        a = a + wv * wv * x[t + 1]
        z[7i_n + t] = a
    end
    for m = 1:8i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
