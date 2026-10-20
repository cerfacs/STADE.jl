function initstacks_if_in_outer_loop_b()
    branch_stack = Vector{Int64}(undef, max(0, div(2 - 1, 1) + 1))
    return branch_stack
end

function if_in_outer_loop_hv(x, xb, y, yb, i_n, xd, xbd, yd, ybd, branch_stack)
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    for i_k = 1:2
        sd = 0.0
        s = 0.0
        if i_n > 2
            branch_stack[1] = 1
            __hcse_0 = x[i_k + 1]
            __hcse_1 = x[i_k]
            sd = __hcse_0 * xd[i_k] + __hcse_1 * xd[i_k + 1]
            s = __hcse_1 * __hcse_0
        else
            branch_stack[1] = 0
        end
        for i = 1:i_n
            __hcse_2 = x[i]
            __hcse_3 = (s * __hcse_2) * sd
            yd[i] = yd[i] + ((__hcse_3 + __hcse_3) + (s * s) * xd[i])
            y[i] = y[i] + s * s * __hcse_2
        end
        for i = i_n:-1:1
            __cse_0d = ybd[i]
            __cse_0 = yb[i]
            __hcse_4 = x[i]
            __hcse_5 = s * __hcse_4
            __cse_1d = __cse_0 * (__hcse_4 * sd + s * xd[i]) + __hcse_5 * __cse_0d
            __cse_1 = __hcse_5 * __cse_0
            sbd = sbd + __cse_1d
            sb = sb + __cse_1
            sbd = sbd + __cse_1d
            sb = sb + __cse_1
            __hcse_6 = s * sd
            __hcse_7 = s * s
            xbd[i] = xbd[i] + (__cse_0 * (__hcse_6 + __hcse_6) + __hcse_7 * __cse_0d)
            xb[i] = xb[i] + __hcse_7 * __cse_0
        end
        __branch = branch_stack[1]
        if __branch == 1
            __oldb_0d = sbd
            __oldb_0 = sb
            sbd = 0.0
            sb = 0.0
            __hcse_8 = x[i_k + 1]
            xbd[i_k] = xbd[i_k] + (__oldb_0 * xd[i_k + 1] + __hcse_8 * __oldb_0d)
            xb[i_k] = xb[i_k] + __hcse_8 * __oldb_0
            __hcse_9 = x[i_k]
            xbd[i_k + 1] = xbd[i_k + 1] + (__oldb_0 * xd[i_k] + __hcse_9 * __oldb_0d)
            xb[i_k + 1] = xb[i_k + 1] + __hcse_9 * __oldb_0
        end
        sbd = 0.0
        sb = 0.0
    end
    return nothing
end

function if_in_outer_loop(x, y, i_n)
    for i_k = 1:2
        s = 0.0
        if i_n > 2
            s = x[i_k] * x[i_k + 1]
        end
        for i = 1:i_n
            y[i] = y[i] + s * s * x[i]
        end
    end
end
