function initstacks_so_carry_b(i_n)
    run_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
    return run_stack
end

function so_carry_hv(x, xb, w, wb, y, yb, loss, lossb, i_n, i_nc, xd, xbd, wd, wbd, yd, ybd, lossd, lossbd, run_stack)
    run_stack_d = Vector{Float64}(undef, length(run_stack))
    a = 0.0
    run = 0.0
    wv = 0.0
    ab = 0.0
    runb = 0.0
    wvb = 0.0
    ad = 0.0
    abd = 0.0
    rund = 0.0
    runbd = 0.0
    wvd = 0.0
    wvbd = 0.0
    rund = 0.0
    run = 0.0
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
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
        __idx_run_stack_2 = (t - 1) + 1
        run_stack_d[__idx_run_stack_2] = rund
        run_stack[__idx_run_stack_2] = run
        rund = 0.5rund + ad
        run = run * 0.5 + a
        yd[t] = rund
        y[t] = run
    end
    for k = 1:i_n
        __cse_0d = yd[k]
        __cse_0 = y[k]
        __hcse_3 = __cse_0 * __cse_0d
        lossd[1] = lossd[1] + (__hcse_3 + __hcse_3)
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    __ihcse_4 = max(0, div(i_n - 1, 1) + 1) + 1
    __idx_run_stack_3 = __ihcse_4
    run_stack_d[__idx_run_stack_3] = rund
    run_stack[__idx_run_stack_3] = run
    __idx_run_stack_0 = __ihcse_4
    rund = run_stack_d[__idx_run_stack_0]
    run = run_stack[__idx_run_stack_0]
    for k = i_n:-1:1
        __hcse_5 = lossb[1]
        __hcse_6 = y[k]
        __cse_1d = __hcse_5 * yd[k] + __hcse_6 * lossbd[1]
        __cse_1 = __hcse_6 * __hcse_5
        ybd[k] = ybd[k] + __cse_1d
        yb[k] = yb[k] + __cse_1
        ybd[k] = ybd[k] + __cse_1d
        yb[k] = yb[k] + __cse_1
    end
    for t = i_n:-1:1
        __oldb_0d = ybd[t]
        __oldb_0 = yb[t]
        ybd[t] = 0.0
        yb[t] = 0.0
        runbd = runbd + __oldb_0d
        runb = runb + __oldb_0
        __idx_run_stack_0 = (t - 1) + 1
        rund = run_stack_d[__idx_run_stack_0]
        run = run_stack[__idx_run_stack_0]
        __oldb_2d = runbd
        __oldb_2 = runb
        runbd = 0.0
        runb = 0.0
        runbd = runbd + 0.5__oldb_2d
        runb = runb + 0.5__oldb_2
        abd = abd + __oldb_2d
        ab = ab + __oldb_2
        for ci = i_nc:-1:1
            wvd = wd[(ci - 1) * 3 + 3]
            wv = w[(ci - 1) * 3 + 3]
            xbd[t + 2] = xbd[t + 2] + (ab * wvd + wv * abd)
            xb[t + 2] = xb[t + 2] + wv * ab
            __hcse_7 = x[t + 2]
            wvbd = wvbd + (ab * xd[t + 2] + __hcse_7 * abd)
            wvb = wvb + __hcse_7 * ab
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
            __hcse_8 = x[t + 1]
            wvbd = wvbd + (ab * xd[t + 1] + __hcse_8 * abd)
            wvb = wvb + __hcse_8 * ab
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
            __hcse_9 = x[t]
            wvbd = wvbd + (ab * xd[t] + __hcse_9 * abd)
            wvb = wvb + __hcse_9 * ab
            __oldb_0d = wvbd
            __oldb_0 = wvb
            wvbd = 0.0
            wvb = 0.0
            wbd[(ci - 1) * 3 + 1] = wbd[(ci - 1) * 3 + 1] + __oldb_0d
            wb[(ci - 1) * 3 + 1] = wb[(ci - 1) * 3 + 1] + __oldb_0
        end
        abd = 0.0
        ab = 0.0
    end
    runbd = 0.0
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
