function initstacks_cross_nested_b(i_n, i_nc)
    xv_stack = Vector{Float64}(undef, (max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) * max(0, div(2 - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1))
    return xv_stack
end

function cross_nested_hv(x, xb, w, wb, z, zb, loss, lossb, i_n, i_nc, xd, xbd, wd, wbd, zd, zbd, lossd, lossbd, xv_stack)
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
        for ci = 1:i_nc
            xvd = xd[(t + ci) - 1]
            xv = x[(t + ci) - 1]
            __hcse_0 = w[ci]
            __hcse_1 = (xv * __hcse_0) * xvd
            ad = ad + ((__hcse_1 + __hcse_1) + (xv * xv) * wd[ci])
            a = a + xv * xv * __hcse_0
            for cj = 1:2
                __idx_xv_stack_0 = ((t - 1) + ((ci - 1) * (div(2 - 1, 1) + 1) + (cj - 1)) * (div(i_n - 1, 1) + 1)) + 1
                xv_stack_d[__idx_xv_stack_0] = xvd
                xv_stack[__idx_xv_stack_0] = xv
                __hcse_2 = w[cj]
                __hcse_3 = x[(t + ci + cj) - 1]
                xvd = __hcse_2 * xd[(t + ci + cj) - 1] + __hcse_3 * wd[cj]
                xv = __hcse_3 * __hcse_2
                __hcse_4 = xv * xvd
                ad = ad + (__hcse_4 + __hcse_4)
                a = a + xv * xv
            end
            __icse_0 = div(i_n - 1, 1) + 1
            __icse_1 = max(0, __icse_0)
            __icse_2 = max(0, div(i_nc - 1, 1) + 1)
            __icse_3 = __icse_1 * __icse_2 * max(0, div(2 - 1, 1) + 1)
            __icse_4 = ((t - 1) + (ci - 1) * __icse_0) + 1
            __idx_xv_stack_3 = __icse_3 + __icse_4
            xv_stack_d[__idx_xv_stack_3] = xvd
            xv_stack[__idx_xv_stack_3] = xv
            xvd = 0.5 * xd[(t + ci) - 1]
            xv = x[(t + ci) - 1] * 0.5
            __hcse_5 = w[ci + 1]
            __hcse_6 = (xv * __hcse_5) * xvd
            ad = ad + ((__hcse_6 + __hcse_6) + (xv * xv) * wd[ci + 1])
            a = a + xv * xv * __hcse_5
            __idx_xv_stack_7 = (__icse_3 + __icse_1 * __icse_2) + __icse_4
            xv_stack_d[__idx_xv_stack_7] = xvd
            xv_stack[__idx_xv_stack_7] = xv
        end
        zd[t] = ad
        z[t] = a
    end
    for m = 1:i_n
        __cse_5d = zd[m]
        __cse_5 = z[m]
        __hcse_7 = __cse_5 * __cse_5d
        lossd[1] = lossd[1] + (__hcse_7 + __hcse_7)
        loss[1] = loss[1] + __cse_5 * __cse_5
    end
    for m = i_n:-1:1
        __hcse_8 = lossb[1]
        __hcse_9 = z[m]
        __cse_6d = __hcse_8 * zd[m] + __hcse_9 * lossbd[1]
        __cse_6 = __hcse_9 * __hcse_8
        zbd[m] = zbd[m] + __cse_6d
        zb[m] = zb[m] + __cse_6
        zbd[m] = zbd[m] + __cse_6d
        zb[m] = zb[m] + __cse_6
    end
    for t = i_n:-1:1
        __oldb_0d = zbd[t]
        __oldb_0 = zb[t]
        zbd[t] = 0.0
        zb[t] = 0.0
        abd = abd + __oldb_0d
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            __icse_7 = div(i_n - 1, 1) + 1
            __icse_8 = max(0, __icse_7)
            __icse_9 = max(0, div(i_nc - 1, 1) + 1)
            __icse_10 = __icse_8 * __icse_9 * max(0, div(2 - 1, 1) + 1)
            __icse_11 = ((t - 1) + (ci - 1) * __icse_7) + 1
            __idx_xv_stack_0 = (__icse_10 + __icse_8 * __icse_9) + __icse_11
            xvd = xv_stack_d[__idx_xv_stack_0]
            xv = xv_stack[__idx_xv_stack_0]
            __hcse_10 = w[ci + 1]
            __hcse_11 = xv * __hcse_10
            __cse_12d = ab * (__hcse_10 * xvd + xv * wd[ci + 1]) + __hcse_11 * abd
            __cse_12 = __hcse_11 * ab
            xvbd = xvbd + __cse_12d
            xvb = xvb + __cse_12
            xvbd = xvbd + __cse_12d
            xvb = xvb + __cse_12
            __hcse_12 = xv * xvd
            __hcse_13 = xv * xv
            wbd[ci + 1] = wbd[ci + 1] + (ab * (__hcse_12 + __hcse_12) + __hcse_13 * abd)
            wb[ci + 1] = wb[ci + 1] + __hcse_13 * ab
            __idx_xv_stack_0 = __icse_10 + __icse_11
            xvd = xv_stack_d[__idx_xv_stack_0]
            xv = xv_stack[__idx_xv_stack_0]
            __oldb_2d = xvbd
            __oldb_2 = xvb
            xvbd = 0.0
            xvb = 0.0
            xbd[(t + ci) - 1] = xbd[(t + ci) - 1] + 0.5__oldb_2d
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + 0.5__oldb_2
            for cj = 2:-1:1
                __cse_13d = ab * xvd + xv * abd
                __cse_13 = xv * ab
                xvbd = xvbd + __cse_13d
                xvb = xvb + __cse_13
                xvbd = xvbd + __cse_13d
                xvb = xvb + __cse_13
                __idx_xv_stack_0 = ((t - 1) + ((ci - 1) * (div(2 - 1, 1) + 1) + (cj - 1)) * (div(i_n - 1, 1) + 1)) + 1
                xvd = xv_stack_d[__idx_xv_stack_0]
                xv = xv_stack[__idx_xv_stack_0]
                __oldb_2d = xvbd
                __oldb_2 = xvb
                xvbd = 0.0
                xvb = 0.0
                __hcse_14 = w[cj]
                xbd[(t + ci + cj) - 1] = xbd[(t + ci + cj) - 1] + (__oldb_2 * wd[cj] + __hcse_14 * __oldb_2d)
                xb[(t + ci + cj) - 1] = xb[(t + ci + cj) - 1] + __hcse_14 * __oldb_2
                __hcse_15 = x[(t + ci + cj) - 1]
                wbd[cj] = wbd[cj] + (__oldb_2 * xd[(t + ci + cj) - 1] + __hcse_15 * __oldb_2d)
                wb[cj] = wb[cj] + __hcse_15 * __oldb_2
            end
            __hcse_16 = w[ci]
            __hcse_17 = xv * __hcse_16
            __cse_14d = ab * (__hcse_16 * xvd + xv * wd[ci]) + __hcse_17 * abd
            __cse_14 = __hcse_17 * ab
            xvbd = xvbd + __cse_14d
            xvb = xvb + __cse_14
            xvbd = xvbd + __cse_14d
            xvb = xvb + __cse_14
            __hcse_18 = xv * xvd
            __hcse_19 = xv * xv
            wbd[ci] = wbd[ci] + (ab * (__hcse_18 + __hcse_18) + __hcse_19 * abd)
            wb[ci] = wb[ci] + __hcse_19 * ab
            __oldb_0d = xvbd
            __oldb_0 = xvb
            xvbd = 0.0
            xvb = 0.0
            xbd[(t + ci) - 1] = xbd[(t + ci) - 1] + __oldb_0d
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + __oldb_0
        end
        abd = 0.0
        ab = 0.0
    end
    return nothing
end

function cross_nested(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv = x[(t + ci) - 1]
            a = a + xv * xv * w[ci]
            for cj = 1:2
                xv = x[(t + ci + cj) - 1] * w[cj]
                a = a + xv * xv
            end
            xv = x[(t + ci) - 1] * 0.5
            a = a + xv * xv * w[ci + 1]
        end
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
