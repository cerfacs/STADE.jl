function initstacks_so_inner_b(i_n, i_nc)
    xv_stack = Vector{Float64}(undef, (max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1)
    wv_stack = Vector{Float64}(undef, (max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1)
    run_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
    return (xv_stack, wv_stack, run_stack)
end

function so_inner_b(x, xb, w, wb, q, qb, y, yb, loss, lossb, i_n, i_nc, xv_stack, wv_stack, run_stack)
    run = 0.0
    wv = 0.0
    xv = 0.0
    runb = 0.0
    wvb = 0.0
    xvb = 0.0
    run = 0.0
    for t = 1:i_n
        for ci = 1:i_nc
            __icse_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
            __idx_xv_stack_0 = __icse_0
            xv_stack[__idx_xv_stack_0] = xv
            xv = x[ci]
            __idx_wv_stack_3 = __icse_0
            wv_stack[__idx_wv_stack_3] = wv
            wv = w[ci]
            q[(t - 1) * i_nc + ci] = xv * wv
            __cse_1 = 1.0 + run
            xv = x[ci + 1] * __cse_1
            wv = w[ci + 1] * __cse_1
            q[(t - 1) * i_nc + ci] = q[(t - 1) * i_nc + ci] + xv * wv * run
        end
        __icse_2 = (t - 1) + 1
        __idx_run_stack_1 = __icse_2
        run_stack[__idx_run_stack_1] = run
        run = run * 0.5 + q[(t - 1) * i_nc + 1]
        y[t] = run
        __icse_3 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + __icse_2
        __idx_wv_stack_5 = __icse_3
        wv_stack[__idx_wv_stack_5] = wv
        __idx_xv_stack_7 = __icse_3
        xv_stack[__idx_xv_stack_7] = xv
    end
    for k = 1:i_n
        __cse_4 = y[k]
        loss[1] = loss[1] + __cse_4 * __cse_4
    end
    __icse_5 = max(0, div(i_n - 1, 1) + 1)
    __icse_6 = __icse_5 + 1
    __idx_run_stack_3 = __icse_6
    run_stack[__idx_run_stack_3] = run
    __icse_7 = (__icse_5 * max(0, div(i_nc - 1, 1) + 1) + __icse_5) + 1
    __idx_wv_stack_5 = __icse_7
    wv_stack[__idx_wv_stack_5] = wv
    __idx_xv_stack_7 = __icse_7
    xv_stack[__idx_xv_stack_7] = xv
    __idx_run_stack_0 = __icse_6
    run = run_stack[__idx_run_stack_0]
    __idx_wv_stack_2 = __icse_7
    wv = wv_stack[__idx_wv_stack_2]
    __idx_xv_stack_4 = __icse_7
    xv = xv_stack[__idx_xv_stack_4]
    for k = i_n:-1:1
        __cse_8 = y[k] * lossb[1]
        yb[k] = yb[k] + __cse_8
        yb[k] = yb[k] + __cse_8
    end
    for t = i_n:-1:1
        __icse_9 = (t - 1) + 1
        __icse_10 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + __icse_9
        __idx_wv_stack_0 = __icse_10
        wv = wv_stack[__idx_wv_stack_0]
        __idx_xv_stack_2 = __icse_10
        xv = xv_stack[__idx_xv_stack_2]
        __oldb_0 = yb[t]
        yb[t] = 0.0
        runb = runb + __oldb_0
        __idx_run_stack_0 = __icse_9
        run = run_stack[__idx_run_stack_0]
        __oldb_2 = runb
        runb = 0.0
        runb = runb + 0.5__oldb_2
        qb[(t - 1) * i_nc + 1] = qb[(t - 1) * i_nc + 1] + __oldb_2
        for ci = i_nc:-1:1
            __cse_11 = qb[(t - 1) * i_nc + ci]
            xvb = xvb + (wv * run) * __cse_11
            wvb = wvb + (xv * run) * __cse_11
            runb = runb + (xv * wv) * __cse_11
            wv = w[ci]
            __oldb_1 = wvb
            wvb = 0.0
            __cse_12 = 1.0 + run
            wb[ci + 1] = wb[ci + 1] + __cse_12 * __oldb_1
            runb = runb + w[ci + 1] * __oldb_1
            xv = x[ci]
            __oldb_1 = xvb
            xvb = 0.0
            xb[ci + 1] = xb[ci + 1] + __cse_12 * __oldb_1
            runb = runb + x[ci + 1] * __oldb_1
            __oldb_0 = __cse_11
            qb[(t - 1) * i_nc + ci] = 0.0
            xvb = xvb + wv * __oldb_0
            wvb = wvb + xv * __oldb_0
            __icse_13 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
            __idx_wv_stack_0 = __icse_13
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[ci] = wb[ci] + __oldb_2
            __idx_xv_stack_0 = __icse_13
            xv = xv_stack[__idx_xv_stack_0]
            __oldb_2 = xvb
            xvb = 0.0
            xb[ci] = xb[ci] + __oldb_2
        end
    end
    runb = 0.0
    return nothing
end

function so_inner(x, w, q, y, loss, i_n, i_nc)
    run = 0.0
    for t = 1:i_n
        for ci = 1:i_nc
            xv = x[ci]
            wv = w[ci]
            q[(t - 1) * i_nc + ci] = xv * wv
            xv = x[ci + 1] * (1.0 + run)
            wv = w[ci + 1] * (1.0 + run)
            q[(t - 1) * i_nc + ci] = q[(t - 1) * i_nc + ci] + xv * wv * run
        end
        run = run * 0.5 + q[(t - 1) * i_nc + 1]
        y[t] = run
    end
    for k = 1:i_n
        loss[1] = loss[1] + y[k] * y[k]
    end
end
