function initstacks_liveout_local_last_b(i_n)
    xv2_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
    return xv2_stack
end

function liveout_local_last_b(x, xb, w, wb, z, zb, loss, lossb, i_n, i_nc, xv2_stack)
    a = 0.0
    xv2 = 0.0
    ab = 0.0
    xv2b = 0.0
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv2 = x[(t + ci) - 1]
            a = xv2 * xv2 * w[ci]
        end
        z[t] = a
        __idx_xv2_stack_3 = (t - 1) + 1
        xv2_stack[__idx_xv2_stack_3] = xv2
    end
    for m = 1:i_n
        __cse_0 = z[m]
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    __icse_1 = max(0, div(i_n - 1, 1) + 1) + 1
    __idx_xv2_stack_2 = __icse_1
    xv2_stack[__idx_xv2_stack_2] = xv2
    __idx_xv2_stack_0 = __icse_1
    xv2 = xv2_stack[__idx_xv2_stack_0]
    for m = i_n:-1:1
        __cse_2 = z[m] * lossb[1]
        zb[m] = zb[m] + __cse_2
        zb[m] = zb[m] + __cse_2
    end
    for t = i_n:-1:1
        __idx_xv2_stack_0 = (t - 1) + 1
        xv2 = xv2_stack[__idx_xv2_stack_0]
        __oldb_0 = zb[t]
        zb[t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            xv2 = x[(t + ci) - 1]
            __oldb_0 = ab
            ab = 0.0
            __cse_3 = (xv2 * w[ci]) * __oldb_0
            xv2b = xv2b + __cse_3
            xv2b = xv2b + __cse_3
            wb[ci] = wb[ci] + (xv2 * xv2) * __oldb_0
            __oldb_0 = xv2b
            xv2b = 0.0
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + __oldb_0
        end
        ab = 0.0
    end
    return nothing
end

function liveout_local_last(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv2 = x[(t + ci) - 1]
            a = xv2 * xv2 * w[ci]
        end
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
