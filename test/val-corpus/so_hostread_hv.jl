function initstacks_so_hostread_b(i_n, i_nc)
    wv_stack = Vector{Float64}(undef, (max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1)
    return wv_stack
end

function so_hostread_hv(x, xb, w, wb, y, yb, q, qb, loss, lossb, i_n, i_nc, xd, xbd, wd, wbd, yd, ybd, qd, qbd, lossd, lossbd, wv_stack)
    wv_stack_d = Vector{Float64}(undef, length(wv_stack))
    a = 0.0
    wv = 0.0
    ab = 0.0
    wvb = 0.0
    ad = 0.0
    abd = 0.0
    wvd = 0.0
    wvbd = 0.0
    wvd = 2.0 * wd[1]
    wv = w[1] * 2.0
    __hcse_0 = wv * wvd
    qd[1] = __hcse_0 + __hcse_0
    q[1] = wv * wv
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            __idx_wv_stack_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
            wv_stack_d[__idx_wv_stack_0] = wvd
            wv_stack[__idx_wv_stack_0] = wv
            wvd = wd[(ci - 1) * 3 + 1]
            wv = w[(ci - 1) * 3 + 1]
            __hcse_1 = x[t]
            ad = ad + (wv * xd[t] + __hcse_1 * wvd)
            a = a + __hcse_1 * wv
            wvd = wd[(ci - 1) * 3 + 2]
            wv = w[(ci - 1) * 3 + 2]
            __hcse_2 = x[t + 1]
            ad = ad + (wv * xd[t + 1] + __hcse_2 * wvd)
            a = a + __hcse_2 * wv
            wvd = wd[(ci - 1) * 3 + 3]
            wv = w[(ci - 1) * 3 + 3]
            __hcse_3 = x[t + 2]
            ad = ad + (wv * xd[t + 2] + __hcse_3 * wvd)
            a = a + __hcse_3 * wv
        end
        yd[t] = ad
        y[t] = a
        __idx_wv_stack_3 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + ((t - 1) + 1)
        wv_stack_d[__idx_wv_stack_3] = wvd
        wv_stack[__idx_wv_stack_3] = wv
    end
    for k = 1:i_n
        __cse_0d = yd[k]
        __cse_0 = y[k]
        __hcse_4 = __cse_0 * __cse_0d
        lossd[1] = (lossd[1] + (__hcse_4 + __hcse_4)) + qd[1]
        loss[1] = loss[1] + __cse_0 * __cse_0 + q[1]
    end
    __ihcse_5 = max(0, div(i_n - 1, 1) + 1)
    __icse_1 = __ihcse_5
    __ihcse_6 = max(0, div(i_nc - 1, 1) + 1)
    __idx_wv_stack_4 = (__icse_1 * __ihcse_6 + __icse_1) + 1
    wv_stack_d[__idx_wv_stack_4] = wvd
    wv_stack[__idx_wv_stack_4] = wv
    __icse_2 = __ihcse_5
    __idx_wv_stack_0 = (__icse_2 * __ihcse_6 + __icse_2) + 1
    wvd = wv_stack_d[__idx_wv_stack_0]
    wv = wv_stack[__idx_wv_stack_0]
    for k = i_n:-1:1
        __cse_3d = lossbd[1]
        __cse_3 = lossb[1]
        __hcse_7 = y[k]
        __cse_4d = __cse_3 * yd[k] + __hcse_7 * __cse_3d
        __cse_4 = __hcse_7 * __cse_3
        ybd[k] = ybd[k] + __cse_4d
        yb[k] = yb[k] + __cse_4
        ybd[k] = ybd[k] + __cse_4d
        yb[k] = yb[k] + __cse_4
        qbd[1] = qbd[1] + __cse_3d
        qb[1] = qb[1] + __cse_3
    end
    for t = i_n:-1:1
        __idx_wv_stack_0 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + ((t - 1) + 1)
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
            __hcse_8 = x[t + 2]
            wvbd = wvbd + (ab * xd[t + 2] + __hcse_8 * abd)
            wvb = wvb + __hcse_8 * ab
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
            __hcse_9 = x[t + 1]
            wvbd = wvbd + (ab * xd[t + 1] + __hcse_9 * abd)
            wvb = wvb + __hcse_9 * ab
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
            __hcse_10 = x[t]
            wvbd = wvbd + (ab * xd[t] + __hcse_10 * abd)
            wvb = wvb + __hcse_10 * ab
            __idx_wv_stack_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
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
    __oldb_0d = qbd[1]
    __oldb_0 = qb[1]
    qbd[1] = 0.0
    qb[1] = 0.0
    __cse_5d = __oldb_0 * wvd + wv * __oldb_0d
    __cse_5 = wv * __oldb_0
    wvbd = wvbd + __cse_5d
    wvb = wvb + __cse_5
    wvbd = wvbd + __cse_5d
    wvb = wvb + __cse_5
    __oldb_0d = wvbd
    __oldb_0 = wvb
    wvbd = 0.0
    wvb = 0.0
    wbd[1] = wbd[1] + 2.0__oldb_0d
    wb[1] = wb[1] + 2.0__oldb_0
    return nothing
end

function so_hostread(x, w, y, q, loss, i_n, i_nc)
    wv = w[1] * 2.0
    q[1] = wv * wv
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
    for k = 1:i_n
        loss[1] = loss[1] + y[k] * y[k] + q[1]
    end
end
