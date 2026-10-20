function initstacks_so_liveout_b(i_n, i_nb, i_nc)
    wv_stack = Vector{Float64}(undef, (((max(0, div(i_nb - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + max(0, div(i_nb - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_nb - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_nb - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1)) + max(0, div(i_nb - 1, 1) + 1))
    return wv_stack
end

function so_liveout_b(x, xb, w, wb, y, yb, z, zb, loss, lossb, i_nb, i_n, i_nc, wv_stack)
    a = 0.0
    wv = 0.0
    ab = 0.0
    wvb = 0.0
    for bb = 1:i_nb
        wv = 0.0
        for t = 1:i_n
            a = 0.0
            for ci = 1:i_nc
                __icse_0 = div(i_n - 1, 1) + 1
                __icse_1 = div(i_nc - 1, 1) + 1
                __icse_2 = ((bb - 1) * (__icse_0 * __icse_1) + (t - 1) * __icse_1 + (ci - 1)) + 1
                __idx_wv_stack_0 = __icse_2
                wv_stack[__idx_wv_stack_0] = wv
                wv = w[(ci - 1) * 3 + 1]
                a = a + x[t] * wv
                __icse_3 = max(0, div(i_nb - 1, 1) + 1) * max(0, __icse_0) * max(0, __icse_1)
                __idx_wv_stack_4 = __icse_3 + __icse_2
                wv_stack[__idx_wv_stack_4] = wv
                wv = w[(ci - 1) * 3 + 2]
                a = a + x[t + 1] * wv
                __idx_wv_stack_8 = (__icse_3 + __icse_3) + __icse_2
                wv_stack[__idx_wv_stack_8] = wv
                wv = w[(ci - 1) * 3 + 3]
                a = a + x[t + 2] * wv
            end
            y[t] = a
            __icse_4 = div(i_n - 1, 1) + 1
            __icse_5 = max(0, div(i_nb - 1, 1) + 1) * max(0, __icse_4) * max(0, div(i_nc - 1, 1) + 1)
            __idx_wv_stack_3 = ((__icse_5 + __icse_5) + __icse_5) + (((bb - 1) * __icse_4 + (t - 1)) + 1)
            wv_stack[__idx_wv_stack_3] = wv
        end
        z[bb] = wv * 3.0
        __icse_6 = max(0, div(i_nb - 1, 1) + 1)
        __icse_7 = max(0, div(i_n - 1, 1) + 1)
        __icse_8 = __icse_6 * __icse_7 * max(0, div(i_nc - 1, 1) + 1)
        __idx_wv_stack_3 = (((__icse_8 + __icse_8) + __icse_8) + __icse_6 * __icse_7) + ((bb - 1) + 1)
        wv_stack[__idx_wv_stack_3] = wv
    end
    for k = 1:i_n
        __cse_9 = y[k]
        loss[1] = loss[1] + __cse_9 * __cse_9
    end
    for k = 1:i_nb
        __cse_10 = z[k]
        loss[1] = loss[1] + __cse_10 * __cse_10
    end
    for k = i_nb:-1:1
        __cse_11 = z[k] * lossb[1]
        zb[k] = zb[k] + __cse_11
        zb[k] = zb[k] + __cse_11
    end
    for k = i_n:-1:1
        __cse_12 = y[k] * lossb[1]
        yb[k] = yb[k] + __cse_12
        yb[k] = yb[k] + __cse_12
    end
    for bb = i_nb:-1:1
        __icse_13 = max(0, div(i_nb - 1, 1) + 1)
        __icse_14 = max(0, div(i_n - 1, 1) + 1)
        __icse_15 = __icse_13 * __icse_14 * max(0, div(i_nc - 1, 1) + 1)
        __idx_wv_stack_0 = (((__icse_15 + __icse_15) + __icse_15) + __icse_13 * __icse_14) + ((bb - 1) + 1)
        wv = wv_stack[__idx_wv_stack_0]
        __oldb_0 = zb[bb]
        zb[bb] = 0.0
        wvb = wvb + 3.0__oldb_0
        for t = i_n:-1:1
            __icse_16 = div(i_n - 1, 1) + 1
            __icse_17 = max(0, div(i_nb - 1, 1) + 1) * max(0, __icse_16) * max(0, div(i_nc - 1, 1) + 1)
            __idx_wv_stack_0 = ((__icse_17 + __icse_17) + __icse_17) + (((bb - 1) * __icse_16 + (t - 1)) + 1)
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_0 = yb[t]
            yb[t] = 0.0
            ab = ab + __oldb_0
            for ci = i_nc:-1:1
                xb[t + 2] = xb[t + 2] + wv * ab
                wvb = wvb + x[t + 2] * ab
                __icse_18 = div(i_n - 1, 1) + 1
                __icse_19 = div(i_nc - 1, 1) + 1
                __icse_20 = max(0, div(i_nb - 1, 1) + 1) * max(0, __icse_18) * max(0, __icse_19)
                __icse_21 = ((bb - 1) * (__icse_18 * __icse_19) + (t - 1) * __icse_19 + (ci - 1)) + 1
                __idx_wv_stack_0 = (__icse_20 + __icse_20) + __icse_21
                wv = wv_stack[__idx_wv_stack_0]
                __oldb_2 = wvb
                wvb = 0.0
                wb[(ci - 1) * 3 + 3] = wb[(ci - 1) * 3 + 3] + __oldb_2
                xb[t + 1] = xb[t + 1] + wv * ab
                wvb = wvb + x[t + 1] * ab
                __idx_wv_stack_0 = __icse_20 + __icse_21
                wv = wv_stack[__idx_wv_stack_0]
                __oldb_2 = wvb
                wvb = 0.0
                wb[(ci - 1) * 3 + 2] = wb[(ci - 1) * 3 + 2] + __oldb_2
                xb[t] = xb[t] + wv * ab
                wvb = wvb + x[t] * ab
                __idx_wv_stack_0 = __icse_21
                wv = wv_stack[__idx_wv_stack_0]
                __oldb_2 = wvb
                wvb = 0.0
                wb[(ci - 1) * 3 + 1] = wb[(ci - 1) * 3 + 1] + __oldb_2
            end
            ab = 0.0
        end
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
