function initstacks_so_carry_b(i_n)
    wv_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
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
            wv = w[(ci - 1) * 3 + 1]
            a = a + x[t] * wv
            wv = w[(ci - 1) * 3 + 2]
            a = a + x[t + 1] * wv
            wv = w[(ci - 1) * 3 + 3]
            a = a + x[t + 2] * wv
        end
        __icse_0 = (t - 1) + 1
        __idx_run_stack_2 = __icse_0
        run_stack[__idx_run_stack_2] = run
        run = run * 0.5 + a
        y[t] = run
        __idx_wv_stack_6 = __icse_0
        wv_stack[__idx_wv_stack_6] = wv
    end
    for k = 1:i_n
        __cse_1 = y[k]
        loss[1] = loss[1] + __cse_1 * __cse_1
    end
    __icse_2 = max(0, div(i_n - 1, 1) + 1) + 1
    __idx_run_stack_3 = __icse_2
    run_stack[__idx_run_stack_3] = run
    __idx_wv_stack_5 = __icse_2
    wv_stack[__idx_wv_stack_5] = wv
    __idx_run_stack_0 = __icse_2
    run = run_stack[__idx_run_stack_0]
    __idx_wv_stack_2 = __icse_2
    wv = wv_stack[__idx_wv_stack_2]
    for k = i_n:-1:1
        __cse_3 = y[k] * lossb[1]
        yb[k] = yb[k] + __cse_3
        yb[k] = yb[k] + __cse_3
    end
    for t = i_n:-1:1
        __icse_4 = (t - 1) + 1
        __idx_wv_stack_0 = __icse_4
        wv = wv_stack[__idx_wv_stack_0]
        __oldb_0 = yb[t]
        yb[t] = 0.0
        runb = runb + __oldb_0
        __idx_run_stack_0 = __icse_4
        run = run_stack[__idx_run_stack_0]
        __oldb_2 = runb
        runb = 0.0
        runb = runb + 0.5__oldb_2
        ab = ab + __oldb_2
        for ci = i_nc:-1:1
            wv = w[(ci - 1) * 3 + 3]
            xb[t + 2] = xb[t + 2] + wv * ab
            wvb = wvb + x[t + 2] * ab
            wv = w[(ci - 1) * 3 + 2]
            __oldb_1 = wvb
            wvb = 0.0
            wb[(ci - 1) * 3 + 3] = wb[(ci - 1) * 3 + 3] + __oldb_1
            xb[t + 1] = xb[t + 1] + wv * ab
            wvb = wvb + x[t + 1] * ab
            wv = w[(ci - 1) * 3 + 1]
            __oldb_1 = wvb
            wvb = 0.0
            wb[(ci - 1) * 3 + 2] = wb[(ci - 1) * 3 + 2] + __oldb_1
            xb[t] = xb[t] + wv * ab
            wvb = wvb + x[t] * ab
            __oldb_0 = wvb
            wvb = 0.0
            wb[(ci - 1) * 3 + 1] = wb[(ci - 1) * 3 + 1] + __oldb_0
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
