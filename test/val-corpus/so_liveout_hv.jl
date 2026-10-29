function initstacks_so_liveout_b(i_n, i_nb, i_nc)
    wv_stack = Vector{Float64}(undef, (max(0, div(i_nb - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + max(0, div(i_nb - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1)) + max(0, div(i_nb - 1, 1) + 1))
    return wv_stack
end

function so_liveout_hv(x, xb, w, wb, y, yb, z, zb, loss, lossb, i_nb, i_n, i_nc, xd, xbd, wd, wbd, yd, ybd, zd, zbd, lossd, lossbd, wv_stack)
    wv_stack_d = Vector{Float64}(undef, length(wv_stack))
    a = 0.0
    wv = 0.0
    ab = 0.0
    wvb = 0.0
    ad = 0.0
    abd = 0.0
    wvd = 0.0
    wvbd = 0.0
    for bb = 1:i_nb
        wvd = 0.0
        wv = 0.0
        for t = 1:i_n
            ad = 0.0
            a = 0.0
            for ci = 1:i_nc
                __icse_0 = div(i_nc - 1, 1) + 1
                __idx_wv_stack_0 = ((bb - 1) * ((div(i_n - 1, 1) + 1) * __icse_0) + (t - 1) * __icse_0 + (ci - 1)) + 1
                wv_stack_d[__idx_wv_stack_0] = wvd
                wv_stack[__idx_wv_stack_0] = wv
                wvd = wd[(ci - 1) * 3 + 1]
                wv = w[(ci - 1) * 3 + 1]
                __hcse_0 = x[t]
                ad = ad + (wv * xd[t] + __hcse_0 * wvd)
                a = a + __hcse_0 * wv
                wvd = wd[(ci - 1) * 3 + 2]
                wv = w[(ci - 1) * 3 + 2]
                __hcse_1 = x[t + 1]
                ad = ad + (wv * xd[t + 1] + __hcse_1 * wvd)
                a = a + __hcse_1 * wv
                wvd = wd[(ci - 1) * 3 + 3]
                wv = w[(ci - 1) * 3 + 3]
                __hcse_2 = x[t + 2]
                ad = ad + (wv * xd[t + 2] + __hcse_2 * wvd)
                a = a + __hcse_2 * wv
            end
            yd[t] = ad
            y[t] = a
            __icse_1 = div(i_n - 1, 1) + 1
            __idx_wv_stack_3 = max(0, div(i_nb - 1, 1) + 1) * max(0, __icse_1) * max(0, div(i_nc - 1, 1) + 1) + (((bb - 1) * __icse_1 + (t - 1)) + 1)
            wv_stack_d[__idx_wv_stack_3] = wvd
            wv_stack[__idx_wv_stack_3] = wv
        end
        zd[bb] = 3.0wvd
        z[bb] = wv * 3.0
        __icse_2 = max(0, div(i_nb - 1, 1) + 1)
        __icse_3 = max(0, div(i_n - 1, 1) + 1)
        __idx_wv_stack_3 = (__icse_2 * __icse_3 * max(0, div(i_nc - 1, 1) + 1) + __icse_2 * __icse_3) + ((bb - 1) + 1)
        wv_stack_d[__idx_wv_stack_3] = wvd
        wv_stack[__idx_wv_stack_3] = wv
    end
    for k = 1:i_n
        __cse_4d = yd[k]
        __cse_4 = y[k]
        __hcse_3 = __cse_4 * __cse_4d
        lossd[1] = lossd[1] + (__hcse_3 + __hcse_3)
        loss[1] = loss[1] + __cse_4 * __cse_4
    end
    for k = 1:i_nb
        __cse_5d = zd[k]
        __cse_5 = z[k]
        __hcse_4 = __cse_5 * __cse_5d
        lossd[1] = lossd[1] + (__hcse_4 + __hcse_4)
        loss[1] = loss[1] + __cse_5 * __cse_5
    end
    for k = i_nb:-1:1
        __hcse_5 = lossb[1]
        __hcse_6 = z[k]
        __cse_6d = __hcse_5 * zd[k] + __hcse_6 * lossbd[1]
        __cse_6 = __hcse_6 * __hcse_5
        zbd[k] = zbd[k] + __cse_6d
        zb[k] = zb[k] + __cse_6
        zbd[k] = zbd[k] + __cse_6d
        zb[k] = zb[k] + __cse_6
    end
    for k = i_n:-1:1
        __hcse_7 = lossb[1]
        __hcse_8 = y[k]
        __cse_7d = __hcse_7 * yd[k] + __hcse_8 * lossbd[1]
        __cse_7 = __hcse_8 * __hcse_7
        ybd[k] = ybd[k] + __cse_7d
        yb[k] = yb[k] + __cse_7
        ybd[k] = ybd[k] + __cse_7d
        yb[k] = yb[k] + __cse_7
    end
    for bb = i_nb:-1:1
        __icse_8 = max(0, div(i_nb - 1, 1) + 1)
        __icse_9 = max(0, div(i_n - 1, 1) + 1)
        __idx_wv_stack_0 = (__icse_8 * __icse_9 * max(0, div(i_nc - 1, 1) + 1) + __icse_8 * __icse_9) + ((bb - 1) + 1)
        wvd = wv_stack_d[__idx_wv_stack_0]
        wv = wv_stack[__idx_wv_stack_0]
        __oldb_0d = zbd[bb]
        __oldb_0 = zb[bb]
        zbd[bb] = 0.0
        zb[bb] = 0.0
        wvbd = wvbd + 3.0__oldb_0d
        wvb = wvb + 3.0__oldb_0
        for t = i_n:-1:1
            __icse_10 = div(i_n - 1, 1) + 1
            __idx_wv_stack_0 = max(0, div(i_nb - 1, 1) + 1) * max(0, __icse_10) * max(0, div(i_nc - 1, 1) + 1) + (((bb - 1) * __icse_10 + (t - 1)) + 1)
            wvd = wv_stack_d[__idx_wv_stack_0]
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_0d = ybd[t]
            __oldb_0 = yb[t]
            ybd[t] = 0.0
            yb[t] = 0.0
            abd = abd + __oldb_0d
            ab = ab + __oldb_0
            for ci = i_nc:-1:1
                xbd[t + 2] = xbd[t + 2] + (ab * wvd + wv * abd)
                xb[t + 2] = xb[t + 2] + wv * ab
                __hcse_9 = x[t + 2]
                wvbd = wvbd + (ab * xd[t + 2] + __hcse_9 * abd)
                wvb = wvb + __hcse_9 * ab
                wvd = wd[(ci - 1) * 3 + 2]
                wv = w[(ci - 1) * 3 + 2]
                __oldb_1d = wvbd
                __oldb_1 = wvb
                wvbd = 0.0
                wvb = 0.0
                wbd[(ci - 1) * 3 + 3] = wbd[(ci - 1) * 3 + 3] + __oldb_1d
                wb[(ci - 1) * 3 + 3] = wb[(ci - 1) * 3 + 3] + __oldb_1
                xbd[t + 1] = xbd[t + 1] + (ab * wvd + wv * abd)
                xb[t + 1] = xb[t + 1] + wv * ab
                __hcse_10 = x[t + 1]
                wvbd = wvbd + (ab * xd[t + 1] + __hcse_10 * abd)
                wvb = wvb + __hcse_10 * ab
                wvd = wd[(ci - 1) * 3 + 1]
                wv = w[(ci - 1) * 3 + 1]
                __oldb_1d = wvbd
                __oldb_1 = wvb
                wvbd = 0.0
                wvb = 0.0
                wbd[(ci - 1) * 3 + 2] = wbd[(ci - 1) * 3 + 2] + __oldb_1d
                wb[(ci - 1) * 3 + 2] = wb[(ci - 1) * 3 + 2] + __oldb_1
                xbd[t] = xbd[t] + (ab * wvd + wv * abd)
                xb[t] = xb[t] + wv * ab
                __hcse_11 = x[t]
                wvbd = wvbd + (ab * xd[t] + __hcse_11 * abd)
                wvb = wvb + __hcse_11 * ab
                __icse_11 = div(i_nc - 1, 1) + 1
                __idx_wv_stack_0 = ((bb - 1) * ((div(i_n - 1, 1) + 1) * __icse_11) + (t - 1) * __icse_11 + (ci - 1)) + 1
                wvd = wv_stack_d[__idx_wv_stack_0]
                wv = wv_stack[__idx_wv_stack_0]
                __oldb_2d = wvbd
                __oldb_2 = wvb
                wvbd = 0.0
                wvb = 0.0
                wbd[(ci - 1) * 3 + 1] = wbd[(ci - 1) * 3 + 1] + __oldb_2d
                wb[(ci - 1) * 3 + 1] = wb[(ci - 1) * 3 + 1] + __oldb_2
            end
            abd = 0.0
            ab = 0.0
        end
        wvbd = 0.0
        wvb = 0.0
    end
    return nothing
end

function so_liveout(x, w, y, z, loss, i_nb, i_n, i_nc)
    for bb = 1:i_nb
        wv = 0.0
        for t = 1:i_n
            a = 0.0
            for ci = 1:i_nc
                wv = w[(ci - 1) * 3 + 1]
                a = a + x[t] * wv
                wv = w[(ci - 1) * 3 + 2]
                a = a + x[t + 1] * wv
                wv = w[(ci - 1) * 3 + 3]
                a = a + x[t + 2] * wv
            end
            y[t] = a
        end
        z[bb] = wv * 3.0
    end
    for k = 1:i_n
        loss[1] = loss[1] + y[k] * y[k]
    end
    for k = 1:i_nb
        loss[1] = loss[1] + z[k] * z[k]
    end
end
