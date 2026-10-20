function initstacks_cross_entry_b(i_n, i_nc)
    xv_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1))
    return xv_stack
end

function cross_entry_hv(x, xb, w, wb, z, zb, loss, lossb, i_n, i_nc, xd, xbd, wd, wbd, zd, zbd, lossd, lossbd, xv_stack)
    xv_stack_d = Vector{Float64}(undef, length(xv_stack))
    a = 0.0
    xv = 0.0
    ab = 0.0
    xvb = 0.0
    ad = 0.0
    abd = 0.0
    xvd = 0.0
    xvbd = 0.0
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        xvd = 0.5 * xd[t]
        xv = x[t] * 0.5
        __hcse_0 = xv * xvd
        ad = ad + (__hcse_0 + __hcse_0)
        a = a + xv * xv
        for ci = 1:i_nc
            __idx_xv_stack_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
            xv_stack_d[__idx_xv_stack_0] = xvd
            xv_stack[__idx_xv_stack_0] = xv
            xvd = xd[(t + ci) - 1]
            xv = x[(t + ci) - 1]
            __hcse_1 = w[ci]
            __hcse_2 = (xv * __hcse_1) * xvd
            ad = ad + ((__hcse_2 + __hcse_2) + (xv * xv) * wd[ci])
            a = a + xv * xv * __hcse_1
        end
        zd[t] = ad
        z[t] = a
        __idx_xv_stack_5 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + ((t - 1) + 1)
        xv_stack_d[__idx_xv_stack_5] = xvd
        xv_stack[__idx_xv_stack_5] = xv
    end
    for m = 1:i_n
        __cse_0d = zd[m]
        __cse_0 = z[m]
        __hcse_3 = __cse_0 * __cse_0d
        lossd[1] = lossd[1] + (__hcse_3 + __hcse_3)
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
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
        __idx_xv_stack_0 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + ((t - 1) + 1)
        xvd = xv_stack_d[__idx_xv_stack_0]
        xv = xv_stack[__idx_xv_stack_0]
        __oldb_0d = zbd[t]
        __oldb_0 = zb[t]
        zbd[t] = 0.0
        zb[t] = 0.0
        abd = abd + __oldb_0d
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            __hcse_6 = w[ci]
            __hcse_7 = xv * __hcse_6
            __cse_2d = ab * (__hcse_6 * xvd + xv * wd[ci]) + __hcse_7 * abd
            __cse_2 = __hcse_7 * ab
            xvbd = xvbd + __cse_2d
            xvb = xvb + __cse_2
            xvbd = xvbd + __cse_2d
            xvb = xvb + __cse_2
            __hcse_8 = xv * xvd
            __hcse_9 = xv * xv
            wbd[ci] = wbd[ci] + (ab * (__hcse_8 + __hcse_8) + __hcse_9 * abd)
            wb[ci] = wb[ci] + __hcse_9 * ab
            __idx_xv_stack_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
            xvd = xv_stack_d[__idx_xv_stack_0]
            xv = xv_stack[__idx_xv_stack_0]
            __oldb_2d = xvbd
            __oldb_2 = xvb
            xvbd = 0.0
            xvb = 0.0
            xbd[(t + ci) - 1] = xbd[(t + ci) - 1] + __oldb_2d
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + __oldb_2
        end
        __cse_3d = ab * xvd + xv * abd
        __cse_3 = xv * ab
        xvbd = xvbd + __cse_3d
        xvb = xvb + __cse_3
        xvbd = xvbd + __cse_3d
        xvb = xvb + __cse_3
        __oldb_0d = xvbd
        __oldb_0 = xvb
        xvbd = 0.0
        xvb = 0.0
        xbd[t] = xbd[t] + 0.5__oldb_0d
        xb[t] = xb[t] + 0.5__oldb_0
        abd = 0.0
        ab = 0.0
    end
    return nothing
end

function cross_entry(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        xv = x[t] * 0.5
        a = a + xv * xv
        for ci = 1:i_nc
            xv = x[(t + ci) - 1]
            a = a + xv * xv * w[ci]
        end
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
