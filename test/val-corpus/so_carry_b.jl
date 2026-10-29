function initstacks_so_carry_b(i_n)
    run_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
    return run_stack
end

function so_carry_b(x, xb, w, wb, y, yb, loss, lossb, i_n, i_nc, run_stack)
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
        __idx_run_stack_2 = (t - 1) + 1
        run_stack[__idx_run_stack_2] = run
        run = run * 0.5 + a
        y[t] = run
    end
    for k = 1:i_n
        __cse_0 = y[k]
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    __icse_1 = max(0, div(i_n - 1, 1) + 1) + 1
    __idx_run_stack_3 = __icse_1
    run_stack[__idx_run_stack_3] = run
    __idx_run_stack_0 = __icse_1
    run = run_stack[__idx_run_stack_0]
    for k = i_n:-1:1
        __cse_2 = y[k] * lossb[1]
        yb[k] = yb[k] + __cse_2
        yb[k] = yb[k] + __cse_2
    end
    for t = i_n:-1:1
        __oldb_0 = yb[t]
        yb[t] = 0.0
        runb = runb + __oldb_0
        __idx_run_stack_0 = (t - 1) + 1
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
