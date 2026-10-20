function initstacks_so_hostread_b(i_n, i_nc)
    wv_stack = Vector{Float64}(undef, (((max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1)
    return wv_stack
end

function so_hostread_b(x, xb, w, wb, y, yb, q, qb, loss, lossb, i_n, i_nc, wv_stack)
    a = 0.0
    wv = 0.0
    ab = 0.0
    wvb = 0.0
    wv = w[1] * 2.0
    q[1] = wv * wv
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            __icse_0 = div(i_nc - 1, 1) + 1
            __icse_1 = ((t - 1) * __icse_0 + (ci - 1)) + 1
            __idx_wv_stack_0 = __icse_1
            wv_stack[__idx_wv_stack_0] = wv
            wv = w[(ci - 1) * 3 + 1]
            a = a + x[t] * wv
            __icse_2 = max(0, div(i_n - 1, 1) + 1) * max(0, __icse_0)
            __idx_wv_stack_4 = __icse_2 + __icse_1
            wv_stack[__idx_wv_stack_4] = wv
            wv = w[(ci - 1) * 3 + 2]
            a = a + x[t + 1] * wv
            __idx_wv_stack_8 = (__icse_2 + __icse_2) + __icse_1
            wv_stack[__idx_wv_stack_8] = wv
            wv = w[(ci - 1) * 3 + 3]
            a = a + x[t + 2] * wv
        end
        y[t] = a
        __icse_3 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)
        __idx_wv_stack_3 = ((__icse_3 + __icse_3) + __icse_3) + ((t - 1) + 1)
        wv_stack[__idx_wv_stack_3] = wv
    end
    for k = 1:i_n
        __cse_4 = y[k]
        loss[1] = loss[1] + __cse_4 * __cse_4 + q[1]
    end
    __icse_5 = max(0, div(i_n - 1, 1) + 1)
    __icse_6 = __icse_5 * max(0, div(i_nc - 1, 1) + 1)
    __icse_7 = (((__icse_6 + __icse_6) + __icse_6) + __icse_5) + 1
    __idx_wv_stack_4 = __icse_7
    wv_stack[__idx_wv_stack_4] = wv
    __idx_wv_stack_0 = __icse_7
    wv = wv_stack[__idx_wv_stack_0]
    for k = i_n:-1:1
        __cse_8 = lossb[1]
        __cse_9 = y[k] * __cse_8
        yb[k] = yb[k] + __cse_9
        yb[k] = yb[k] + __cse_9
        qb[1] = qb[1] + __cse_8
    end
    for t = i_n:-1:1
        __icse_10 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)
        __idx_wv_stack_0 = ((__icse_10 + __icse_10) + __icse_10) + ((t - 1) + 1)
        wv = wv_stack[__idx_wv_stack_0]
        __oldb_0 = yb[t]
        yb[t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            xb[t + 2] = xb[t + 2] + wv * ab
            wvb = wvb + x[t + 2] * ab
            __icse_11 = div(i_nc - 1, 1) + 1
            __icse_12 = max(0, div(i_n - 1, 1) + 1) * max(0, __icse_11)
            __icse_13 = ((t - 1) * __icse_11 + (ci - 1)) + 1
            __idx_wv_stack_0 = (__icse_12 + __icse_12) + __icse_13
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[(ci - 1) * 3 + 3] = wb[(ci - 1) * 3 + 3] + __oldb_2
            xb[t + 1] = xb[t + 1] + wv * ab
            wvb = wvb + x[t + 1] * ab
            __idx_wv_stack_0 = __icse_12 + __icse_13
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[(ci - 1) * 3 + 2] = wb[(ci - 1) * 3 + 2] + __oldb_2
            xb[t] = xb[t] + wv * ab
            wvb = wvb + x[t] * ab
            __idx_wv_stack_0 = __icse_13
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[(ci - 1) * 3 + 1] = wb[(ci - 1) * 3 + 1] + __oldb_2
        end
        ab = 0.0
    end
    __oldb_0 = qb[1]
    qb[1] = 0.0
    __cse_14 = wv * __oldb_0
    wvb = wvb + __cse_14
    wvb = wvb + __cse_14
    __oldb_0 = wvb
    wvb = 0.0
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
