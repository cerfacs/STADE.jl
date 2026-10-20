function initstacks_so_carry_b(i_n, i_nc)
    wv_stack = Vector{Float64}(undef, (((max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1)
    run_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
    return (wv_stack, run_stack)
end

function so_carry_b(x, xb, w, wb, y, yb, loss, lossb, i_n, i_nc, wv_stack, run_stack)
    a = 0.0
    run = 0.0
    wv = 0.0
    ab = 0.0
    runb = 0.0
    wvb = 0.0
    run = 0.0
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
        __icse_3 = (t - 1) + 1
        __idx_run_stack_2 = __icse_3
        run_stack[__idx_run_stack_2] = run
        run = run * 0.5 + a
        y[t] = run
        __icse_4 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)
        __idx_wv_stack_6 = ((__icse_4 + __icse_4) + __icse_4) + __icse_3
        wv_stack[__idx_wv_stack_6] = wv
    end
    for k = 1:i_n
        __cse_5 = y[k]
        loss[1] = loss[1] + __cse_5 * __cse_5
    end
    __icse_6 = max(0, div(i_n - 1, 1) + 1)
    __icse_7 = __icse_6 + 1
    __idx_run_stack_3 = __icse_7
    run_stack[__idx_run_stack_3] = run
    __icse_8 = __icse_6 * max(0, div(i_nc - 1, 1) + 1)
    __icse_9 = (((__icse_8 + __icse_8) + __icse_8) + __icse_6) + 1
    __idx_wv_stack_5 = __icse_9
    wv_stack[__idx_wv_stack_5] = wv
    __idx_run_stack_0 = __icse_7
    run = run_stack[__idx_run_stack_0]
    __idx_wv_stack_2 = __icse_9
    wv = wv_stack[__idx_wv_stack_2]
    for k = i_n:-1:1
        __cse_10 = y[k] * lossb[1]
        yb[k] = yb[k] + __cse_10
        yb[k] = yb[k] + __cse_10
    end
    for t = i_n:-1:1
        __icse_11 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)
        __icse_12 = (t - 1) + 1
        __idx_wv_stack_0 = ((__icse_11 + __icse_11) + __icse_11) + __icse_12
        wv = wv_stack[__idx_wv_stack_0]
        __oldb_0 = yb[t]
        yb[t] = 0.0
        runb = runb + __oldb_0
        __idx_run_stack_0 = __icse_12
        run = run_stack[__idx_run_stack_0]
        __oldb_2 = runb
        runb = 0.0
        runb = runb + 0.5__oldb_2
        ab = ab + __oldb_2
        for ci = i_nc:-1:1
            xb[t + 2] = xb[t + 2] + wv * ab
            wvb = wvb + x[t + 2] * ab
            __icse_13 = div(i_nc - 1, 1) + 1
            __icse_14 = max(0, div(i_n - 1, 1) + 1) * max(0, __icse_13)
            __icse_15 = ((t - 1) * __icse_13 + (ci - 1)) + 1
            __idx_wv_stack_0 = (__icse_14 + __icse_14) + __icse_15
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[(ci - 1) * 3 + 3] = wb[(ci - 1) * 3 + 3] + __oldb_2
            xb[t + 1] = xb[t + 1] + wv * ab
            wvb = wvb + x[t + 1] * ab
            __idx_wv_stack_0 = __icse_14 + __icse_15
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[(ci - 1) * 3 + 2] = wb[(ci - 1) * 3 + 2] + __oldb_2
            xb[t] = xb[t] + wv * ab
            wvb = wvb + x[t] * ab
            __idx_wv_stack_0 = __icse_15
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[(ci - 1) * 3 + 1] = wb[(ci - 1) * 3 + 1] + __oldb_2
        end
        ab = 0.0
    end
    runb = 0.0
    return nothing
end

function so_carry(x, w, y, loss, i_n, i_nc)
    run = 0.0
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
        run = run * 0.5 + a
        y[t] = run
    end
    for k = 1:i_n
        loss[1] = loss[1] + y[k] * y[k]
    end
end
