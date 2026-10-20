function initstacks_if_scalar_rewrite_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1))
    wv_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1))
    return (branch_stack, wv_stack)
end

function if_scalar_rewrite_hv(x, xb, w, wb, y, yb, loss, lossb, i_n, xd, xbd, wd, wbd, yd, ybd, lossd, lossbd, branch_stack, wv_stack)
    wv_stack_d = Vector{Float64}(undef, length(wv_stack))
    wv = 0.0
    wvb = 0.0
    wvd = 0.0
    wvbd = 0.0
    for t = 1:i_n
        wvd = 0.0
        wv = 0.0
        if x[t] > 0.0
            __idx_branch_stack_0 = (t - 1) + 1
            branch_stack[__idx_branch_stack_0] = 1
            wvd = wd[t]
            wv = w[t]
        else
            __idx_branch_stack_0 = (t - 1) + 1
            branch_stack[__idx_branch_stack_0] = 0
        end
        __cse_0d = xd[t]
        __cse_0 = x[t]
        __hcse_0 = (wv * __cse_0) * wvd
        yd[t] = (__hcse_0 + __hcse_0) + (wv * wv) * __cse_0d
        y[t] = wv * wv * __cse_0
        __icse_1 = (t - 1) + 1
        __idx_wv_stack_3 = __icse_1
        wv_stack_d[__idx_wv_stack_3] = wvd
        wv_stack[__idx_wv_stack_3] = wv
        wvd = wd[t + 1]
        wv = w[t + 1]
        __hcse_1 = (wv * __cse_0) * wvd
        yd[t] = yd[t] + ((__hcse_1 + __hcse_1) + (wv * wv) * __cse_0d)
        y[t] = y[t] + wv * wv * __cse_0
        __idx_wv_stack_7 = max(0, div(i_n - 1, 1) + 1) + __icse_1
        wv_stack_d[__idx_wv_stack_7] = wvd
        wv_stack[__idx_wv_stack_7] = wv
    end
    for m = 1:i_n
        __cse_2d = yd[m]
        __cse_2 = y[m]
        __hcse_2 = __cse_2 * __cse_2d
        lossd[1] = lossd[1] + (__hcse_2 + __hcse_2)
        loss[1] = loss[1] + __cse_2 * __cse_2
    end
    for m = i_n:-1:1
        __hcse_3 = lossb[1]
        __hcse_4 = y[m]
        __cse_3d = __hcse_3 * yd[m] + __hcse_4 * lossbd[1]
        __cse_3 = __hcse_4 * __hcse_3
        ybd[m] = ybd[m] + __cse_3d
        yb[m] = yb[m] + __cse_3
        ybd[m] = ybd[m] + __cse_3d
        yb[m] = yb[m] + __cse_3
    end
    for t = i_n:-1:1
        __icse_4 = (t - 1) + 1
        __idx_wv_stack_0 = max(0, div(i_n - 1, 1) + 1) + __icse_4
        wvd = wv_stack_d[__idx_wv_stack_0]
        wv = wv_stack[__idx_wv_stack_0]
        __cse_5d = xd[t]
        __cse_5 = x[t]
        __cse_6d = ybd[t]
        __cse_6 = yb[t]
        __hcse_5 = wv * __cse_5
        __cse_7d = __cse_6 * (__cse_5 * wvd + wv * __cse_5d) + __hcse_5 * __cse_6d
        __cse_7 = __hcse_5 * __cse_6
        wvbd = wvbd + __cse_7d
        wvb = wvb + __cse_7
        wvbd = wvbd + __cse_7d
        wvb = wvb + __cse_7
        __hcse_6 = wv * wvd
        __hcse_7 = wv * wv
        xbd[t] = xbd[t] + (__cse_6 * (__hcse_6 + __hcse_6) + __hcse_7 * __cse_6d)
        xb[t] = xb[t] + __hcse_7 * __cse_6
        __idx_wv_stack_0 = __icse_4
        wvd = wv_stack_d[__idx_wv_stack_0]
        wv = wv_stack[__idx_wv_stack_0]
        __oldb_2d = wvbd
        __oldb_2 = wvb
        wvbd = 0.0
        wvb = 0.0
        wbd[t + 1] = wbd[t + 1] + __oldb_2d
        wb[t + 1] = wb[t + 1] + __oldb_2
        __oldb_0d = __cse_6d
        __oldb_0 = __cse_6
        ybd[t] = 0.0
        yb[t] = 0.0
        __hcse_8 = wv * __cse_5
        __cse_8d = __oldb_0 * (__cse_5 * wvd + wv * __cse_5d) + __hcse_8 * __oldb_0d
        __cse_8 = __hcse_8 * __oldb_0
        wvbd = wvbd + __cse_8d
        wvb = wvb + __cse_8
        wvbd = wvbd + __cse_8d
        wvb = wvb + __cse_8
        __hcse_9 = wv * wvd
        __hcse_10 = wv * wv
        xbd[t] = xbd[t] + (__oldb_0 * (__hcse_9 + __hcse_9) + __hcse_10 * __oldb_0d)
        xb[t] = xb[t] + __hcse_10 * __oldb_0
        __idx_branch_stack_15 = __icse_4
        __branch = branch_stack[__idx_branch_stack_15]
        if __branch == 1
            __oldb_0d = wvbd
            __oldb_0 = wvb
            wvbd = 0.0
            wvb = 0.0
            wbd[t] = wbd[t] + __oldb_0d
            wb[t] = wb[t] + __oldb_0
        end
        wvbd = 0.0
        wvb = 0.0
    end
    return nothing
end

function if_scalar_rewrite(x, w, y, loss, i_n)
    for t = 1:i_n
        wv = 0.0
        if x[t] > 0.0
            wv = w[t]
        end
        y[t] = wv * wv * x[t]
        wv = w[t + 1]
        y[t] = y[t] + wv * wv * x[t]
    end
    for m = 1:i_n
        loss[1] = loss[1] + y[m] * y[m]
    end
end
