function initstacks_cross_branchlocal_b(i_n, i_nc)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1))
    xv_stack = Vector{Float64}(undef, ((((max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1)
    return (branch_stack, xv_stack)
end

function cross_branchlocal_hv(x, xb, w, wb, z, zb, loss, lossb, i_n, i_nc, xd, xbd, wd, wbd, zd, zbd, lossd, lossbd, branch_stack, xv_stack)
    xv_stack_d = Vector{Float64}(undef, length(xv_stack))
    a = 0.0
    b = 0.0
    xv = 0.0
    ab = 0.0
    bb = 0.0
    xvb = 0.0
    ad = 0.0
    abd = 0.0
    bd = 0.0
    bbd = 0.0
    xvd = 0.0
    xvbd = 0.0
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            if x[(t + ci) - 1] > 0.5
                __icse_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
                __idx_branch_stack_0 = __icse_0
                branch_stack[__idx_branch_stack_0] = 1
                __idx_xv_stack_0 = __icse_0
                xv_stack_d[__idx_xv_stack_0] = xvd
                xv_stack[__idx_xv_stack_0] = xv
                xvd = 2.0 * xd[(t + ci) - 1]
                xv = x[(t + ci) - 1] * 2.0
                __hcse_0 = w[ci]
                __hcse_1 = (xv * __hcse_0) * xvd
                ad = ad + ((__hcse_1 + __hcse_1) + (xv * xv) * wd[ci])
                a = a + xv * xv * __hcse_0
            else
                __idx_branch_stack_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
                branch_stack[__idx_branch_stack_0] = 0
            end
            __icse_1 = div(i_nc - 1, 1) + 1
            __idx_xv_stack_1 = max(0, div(i_n - 1, 1) + 1) * max(0, __icse_1) + (((t - 1) * __icse_1 + (ci - 1)) + 1)
            xv_stack_d[__idx_xv_stack_1] = xvd
            xv_stack[__idx_xv_stack_1] = xv
        end
        zd[t] = ad
        z[t] = a
        __icse_2 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)
        __idx_xv_stack_3 = (__icse_2 + __icse_2) + ((t - 1) + 1)
        xv_stack_d[__idx_xv_stack_3] = xvd
        xv_stack[__idx_xv_stack_3] = xv
    end
    for t = 1:i_n
        bd = 0.0
        b = 0.0
        for ci = 1:i_nc
            __icse_3 = max(0, div(i_n - 1, 1) + 1)
            __icse_4 = div(i_nc - 1, 1) + 1
            __icse_5 = __icse_3 * max(0, __icse_4)
            __idx_xv_stack_0 = ((__icse_5 + __icse_5) + __icse_3) + (((t - 1) * __icse_4 + (ci - 1)) + 1)
            xv_stack_d[__idx_xv_stack_0] = xvd
            xv_stack[__idx_xv_stack_0] = xv
            xvd = xd[(t + ci) - 1]
            xv = x[(t + ci) - 1]
            __hcse_2 = w[ci]
            __hcse_3 = (xv * __hcse_2) * xvd
            bd = bd + ((__hcse_3 + __hcse_3) + (xv * xv) * wd[ci])
            b = b + xv * xv * __hcse_2
        end
        zd[i_n + t] = bd
        z[i_n + t] = b
        __icse_6 = max(0, div(i_n - 1, 1) + 1)
        __icse_7 = __icse_6 * max(0, div(i_nc - 1, 1) + 1)
        __idx_xv_stack_3 = (((__icse_7 + __icse_7) + __icse_6) + __icse_7) + ((t - 1) + 1)
        xv_stack_d[__idx_xv_stack_3] = xvd
        xv_stack[__idx_xv_stack_3] = xv
    end
    for m = 1:2i_n
        __cse_8d = zd[m]
        __cse_8 = z[m]
        __hcse_4 = __cse_8 * __cse_8d
        lossd[1] = lossd[1] + (__hcse_4 + __hcse_4)
        loss[1] = loss[1] + __cse_8 * __cse_8
    end
    __ihcse_5 = max(0, div(i_n - 1, 1) + 1)
    __icse_9 = __ihcse_5
    __ihcse_6 = max(0, div(i_nc - 1, 1) + 1)
    __icse_10 = __icse_9 * __ihcse_6
    __idx_xv_stack_3 = ((((__icse_10 + __icse_10) + __icse_9) + __icse_10) + __icse_9) + 1
    xv_stack_d[__idx_xv_stack_3] = xvd
    xv_stack[__idx_xv_stack_3] = xv
    __icse_11 = __ihcse_5
    __icse_12 = __icse_11 * __ihcse_6
    __idx_xv_stack_0 = ((((__icse_12 + __icse_12) + __icse_11) + __icse_12) + __icse_11) + 1
    xvd = xv_stack_d[__idx_xv_stack_0]
    xv = xv_stack[__idx_xv_stack_0]
    for m = 2i_n:-1:1
        __hcse_7 = lossb[1]
        __hcse_8 = z[m]
        __cse_13d = __hcse_7 * zd[m] + __hcse_8 * lossbd[1]
        __cse_13 = __hcse_8 * __hcse_7
        zbd[m] = zbd[m] + __cse_13d
        zb[m] = zb[m] + __cse_13
        zbd[m] = zbd[m] + __cse_13d
        zb[m] = zb[m] + __cse_13
    end
    for t = i_n:-1:1
        __icse_14 = max(0, div(i_n - 1, 1) + 1)
        __icse_15 = __icse_14 * max(0, div(i_nc - 1, 1) + 1)
        __idx_xv_stack_0 = (((__icse_15 + __icse_15) + __icse_14) + __icse_15) + ((t - 1) + 1)
        xvd = xv_stack_d[__idx_xv_stack_0]
        xv = xv_stack[__idx_xv_stack_0]
        __oldb_0d = zbd[i_n + t]
        __oldb_0 = zb[i_n + t]
        zbd[i_n + t] = 0.0
        zb[i_n + t] = 0.0
        bbd = bbd + __oldb_0d
        bb = bb + __oldb_0
        for ci = i_nc:-1:1
            __hcse_9 = w[ci]
            __hcse_10 = xv * __hcse_9
            __cse_16d = bb * (__hcse_9 * xvd + xv * wd[ci]) + __hcse_10 * bbd
            __cse_16 = __hcse_10 * bb
            xvbd = xvbd + __cse_16d
            xvb = xvb + __cse_16
            xvbd = xvbd + __cse_16d
            xvb = xvb + __cse_16
            __hcse_11 = xv * xvd
            __hcse_12 = xv * xv
            wbd[ci] = wbd[ci] + (bb * (__hcse_11 + __hcse_11) + __hcse_12 * bbd)
            wb[ci] = wb[ci] + __hcse_12 * bb
            __icse_17 = max(0, div(i_n - 1, 1) + 1)
            __icse_18 = div(i_nc - 1, 1) + 1
            __icse_19 = __icse_17 * max(0, __icse_18)
            __idx_xv_stack_0 = ((__icse_19 + __icse_19) + __icse_17) + (((t - 1) * __icse_18 + (ci - 1)) + 1)
            xvd = xv_stack_d[__idx_xv_stack_0]
            xv = xv_stack[__idx_xv_stack_0]
            __oldb_2d = xvbd
            __oldb_2 = xvb
            xvbd = 0.0
            xvb = 0.0
            xbd[(t + ci) - 1] = xbd[(t + ci) - 1] + __oldb_2d
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + __oldb_2
        end
        bbd = 0.0
        bb = 0.0
    end
    for t = i_n:-1:1
        __icse_20 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)
        __idx_xv_stack_0 = (__icse_20 + __icse_20) + ((t - 1) + 1)
        xvd = xv_stack_d[__idx_xv_stack_0]
        xv = xv_stack[__idx_xv_stack_0]
        __oldb_0d = zbd[t]
        __oldb_0 = zb[t]
        zbd[t] = 0.0
        zb[t] = 0.0
        abd = abd + __oldb_0d
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            __icse_21 = div(i_nc - 1, 1) + 1
            __icse_22 = ((t - 1) * __icse_21 + (ci - 1)) + 1
            __idx_xv_stack_0 = max(0, div(i_n - 1, 1) + 1) * max(0, __icse_21) + __icse_22
            xvd = xv_stack_d[__idx_xv_stack_0]
            xv = xv_stack[__idx_xv_stack_0]
            __idx_branch_stack_2 = __icse_22
            __branch = branch_stack[__idx_branch_stack_2]
            if __branch == 1
                __hcse_13 = w[ci]
                __hcse_14 = xv * __hcse_13
                __cse_23d = ab * (__hcse_13 * xvd + xv * wd[ci]) + __hcse_14 * abd
                __cse_23 = __hcse_14 * ab
                xvbd = xvbd + __cse_23d
                xvb = xvb + __cse_23
                xvbd = xvbd + __cse_23d
                xvb = xvb + __cse_23
                __hcse_15 = xv * xvd
                __hcse_16 = xv * xv
                wbd[ci] = wbd[ci] + (ab * (__hcse_15 + __hcse_15) + __hcse_16 * abd)
                wb[ci] = wb[ci] + __hcse_16 * ab
                __idx_xv_stack_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
                xvd = xv_stack_d[__idx_xv_stack_0]
                xv = xv_stack[__idx_xv_stack_0]
                __oldb_2d = xvbd
                __oldb_2 = xvb
                xvbd = 0.0
                xvb = 0.0
                xbd[(t + ci) - 1] = xbd[(t + ci) - 1] + 2.0__oldb_2d
                xb[(t + ci) - 1] = xb[(t + ci) - 1] + 2.0__oldb_2
            end
        end
        abd = 0.0
        ab = 0.0
    end
    return nothing
end

function cross_branchlocal(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            if x[(t + ci) - 1] > 0.5
                xv = x[(t + ci) - 1] * 2.0
                a = a + xv * xv * w[ci]
            end
        end
        z[t] = a
    end
    for t = 1:i_n
        b = 0.0
        for ci = 1:i_nc
            xv = x[(t + ci) - 1]
            b = b + xv * xv * w[ci]
        end
        z[i_n + t] = b
    end
    for m = 1:2i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
