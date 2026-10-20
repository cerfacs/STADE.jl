function initstacks_if_in_outer_loop_b()
    branch_stack = Vector{Int64}(undef, max(0, div(2 - 1, 1) + 1))
    return branch_stack
end

function if_in_outer_loop_b(x, xb, y, yb, i_n, branch_stack)
    s = 0.0
    sb = 0.0
    for i_k = 1:2
        s = 0.0
        if i_n > 2
            branch_stack[1] = 1
            s = x[i_k] * x[i_k + 1]
        else
            branch_stack[1] = 0
        end
        for i = 1:i_n
            y[i] = y[i] + s * s * x[i]
        end
        for i = i_n:-1:1
            __cse_0 = yb[i]
            __cse_1 = (s * x[i]) * __cse_0
            sb = sb + __cse_1
            sb = sb + __cse_1
            xb[i] = xb[i] + (s * s) * __cse_0
        end
        __branch = branch_stack[1]
        if __branch == 1
            __oldb_0 = sb
            sb = 0.0
            xb[i_k] = xb[i_k] + x[i_k + 1] * __oldb_0
            xb[i_k + 1] = xb[i_k + 1] + x[i_k] * __oldb_0
        end
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
