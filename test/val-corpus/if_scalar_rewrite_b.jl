function initstacks_if_scalar_rewrite_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1))
    wv_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1))
    return (branch_stack, wv_stack)
end

function if_scalar_rewrite_b(x, xb, w, wb, y, yb, loss, lossb, i_n, branch_stack, wv_stack)
    wv = 0.0
    wvb = 0.0
    for t = 1:i_n
        wv = 0.0
        if x[t] > 0.0
            __idx_branch_stack_0 = (t - 1) + 1
            branch_stack[__idx_branch_stack_0] = 1
            wv = w[t]
        else
            __idx_branch_stack_0 = (t - 1) + 1
            branch_stack[__idx_branch_stack_0] = 0
        end
        __cse_0 = x[t]
        y[t] = wv * wv * __cse_0
        __icse_1 = (t - 1) + 1
        __idx_wv_stack_3 = __icse_1
        wv_stack[__idx_wv_stack_3] = wv
        wv = w[t + 1]
        y[t] = y[t] + wv * wv * __cse_0
        __idx_wv_stack_7 = max(0, div(i_n - 1, 1) + 1) + __icse_1
        wv_stack[__idx_wv_stack_7] = wv
    end
    for m = 1:i_n
        __cse_2 = y[m]
        loss[1] = loss[1] + __cse_2 * __cse_2
    end
    for m = i_n:-1:1
        __cse_3 = y[m] * lossb[1]
        yb[m] = yb[m] + __cse_3
        yb[m] = yb[m] + __cse_3
    end
    for t = i_n:-1:1
        __icse_4 = (t - 1) + 1
        __idx_wv_stack_0 = max(0, div(i_n - 1, 1) + 1) + __icse_4
        wv = wv_stack[__idx_wv_stack_0]
        __cse_5 = x[t]
        __cse_6 = yb[t]
        __cse_7 = (wv * __cse_5) * __cse_6
        wvb = wvb + __cse_7
        wvb = wvb + __cse_7
        xb[t] = xb[t] + (wv * wv) * __cse_6
        __idx_wv_stack_0 = __icse_4
        wv = wv_stack[__idx_wv_stack_0]
        __oldb_2 = wvb
        wvb = 0.0
        wb[t + 1] = wb[t + 1] + __oldb_2
        __oldb_0 = __cse_6
        yb[t] = 0.0
        __cse_8 = (wv * __cse_5) * __oldb_0
        wvb = wvb + __cse_8
        wvb = wvb + __cse_8
        xb[t] = xb[t] + (wv * wv) * __oldb_0
        __idx_branch_stack_15 = __icse_4
        __branch = branch_stack[__idx_branch_stack_15]
        if __branch == 1
            __oldb_0 = wvb
            wvb = 0.0
            wb[t] = wb[t] + __oldb_0
        end
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
