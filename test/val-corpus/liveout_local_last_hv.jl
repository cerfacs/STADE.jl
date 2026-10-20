function initstacks_liveout_local_last_b(i_n)
    xv2_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
    return xv2_stack
end

function liveout_local_last_hv(x, xb, w, wb, z, zb, loss, lossb, i_n, i_nc, xd, xbd, wd, wbd, zd, zbd, lossd, lossbd, xv2_stack)
    xv2_stack_d = Vector{Float64}(undef, length(xv2_stack))
    a = 0.0
    xv2 = 0.0
    ab = 0.0
    xv2b = 0.0
    ad = 0.0
    abd = 0.0
    xv2d = 0.0
    xv2bd = 0.0
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            xv2d = xd[(t + ci) - 1]
            xv2 = x[(t + ci) - 1]
            __hcse_0 = w[ci]
            __hcse_1 = (xv2 * __hcse_0) * xv2d
            ad = (__hcse_1 + __hcse_1) + (xv2 * xv2) * wd[ci]
            a = xv2 * xv2 * __hcse_0
        end
        zd[t] = ad
        z[t] = a
        __idx_xv2_stack_3 = (t - 1) + 1
        xv2_stack_d[__idx_xv2_stack_3] = xv2d
        xv2_stack[__idx_xv2_stack_3] = xv2
    end
    for m = 1:i_n
        __cse_0d = zd[m]
        __cse_0 = z[m]
        __hcse_2 = __cse_0 * __cse_0d
        lossd[1] = lossd[1] + (__hcse_2 + __hcse_2)
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    __ihcse_3 = max(0, div(i_n - 1, 1) + 1) + 1
    __idx_xv2_stack_2 = __ihcse_3
    xv2_stack_d[__idx_xv2_stack_2] = xv2d
    xv2_stack[__idx_xv2_stack_2] = xv2
    __idx_xv2_stack_0 = __ihcse_3
    xv2d = xv2_stack_d[__idx_xv2_stack_0]
    xv2 = xv2_stack[__idx_xv2_stack_0]
    for m = i_n:-1:1
        __hcse_4 = lossb[1]
        __hcse_5 = z[m]
        __cse_1d = __hcse_4 * zd[m] + __hcse_5 * lossbd[1]
        __cse_1 = __hcse_5 * __hcse_4
        zbd[m] = zbd[m] + __cse_1d
        zb[m] = zb[m] + __cse_1
        zbd[m] = zbd[m] + __cse_1d
        zb[m] = zb[m] + __cse_1
    end
    for t = i_n:-1:1
        __idx_xv2_stack_0 = (t - 1) + 1
        xv2d = xv2_stack_d[__idx_xv2_stack_0]
        xv2 = xv2_stack[__idx_xv2_stack_0]
        __oldb_0d = zbd[t]
        __oldb_0 = zb[t]
        zbd[t] = 0.0
        zb[t] = 0.0
        abd = abd + __oldb_0d
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            xv2d = xd[(t + ci) - 1]
            xv2 = x[(t + ci) - 1]
            __oldb_0d = abd
            __oldb_0 = ab
            abd = 0.0
            ab = 0.0
            __hcse_6 = w[ci]
            __hcse_7 = xv2 * __hcse_6
            __cse_2d = __oldb_0 * (__hcse_6 * xv2d + xv2 * wd[ci]) + __hcse_7 * __oldb_0d
            __cse_2 = __hcse_7 * __oldb_0
            xv2bd = xv2bd + __cse_2d
            xv2b = xv2b + __cse_2
            xv2bd = xv2bd + __cse_2d
            xv2b = xv2b + __cse_2
            __hcse_8 = xv2 * xv2d
            __hcse_9 = xv2 * xv2
            wbd[ci] = wbd[ci] + (__oldb_0 * (__hcse_8 + __hcse_8) + __hcse_9 * __oldb_0d)
            wb[ci] = wb[ci] + __hcse_9 * __oldb_0
            __oldb_0d = xv2bd
            __oldb_0 = xv2b
            xv2bd = 0.0
            xv2b = 0.0
            xbd[(t + ci) - 1] = xbd[(t + ci) - 1] + __oldb_0d
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + __oldb_0
        end
        abd = 0.0
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
