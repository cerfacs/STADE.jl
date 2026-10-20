function initstacks_cross_ifwrite_b(i_n, i_nc)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1))
    xv_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1))
    return (branch_stack, xv_stack)
end

function cross_ifwrite_hv(x, xb, w, wb, z, zb, loss, lossb, i_n, i_nc, xd, xbd, wd, wbd, zd, zbd, lossd, lossbd, branch_stack, xv_stack)
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
            if x[t + ci] > 0.5
                __icse_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
                __idx_branch_stack_0 = __icse_0
                branch_stack[__idx_branch_stack_0] = 1
                __idx_xv_stack_0 = __icse_0
                xv_stack_d[__idx_xv_stack_0] = xvd
                xv_stack[__idx_xv_stack_0] = xv
                xvd = 2.0xvd
                xv = xv * 2.0
            else
                __idx_branch_stack_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
                branch_stack[__idx_branch_stack_0] = 0
            end
            __hcse_2 = xv * xvd
            ad = ad + (__hcse_2 + __hcse_2)
            a = a + xv * xv
            __icse_1 = div(i_nc - 1, 1) + 1
            __idx_xv_stack_4 = max(0, div(i_n - 1, 1) + 1) * max(0, __icse_1) + (((t - 1) * __icse_1 + (ci - 1)) + 1)
            xv_stack_d[__idx_xv_stack_4] = xvd
            xv_stack[__idx_xv_stack_4] = xv
        end
        zd[t] = ad
        z[t] = a
    end
    for m = 1:i_n
        __cse_2d = zd[m]
        __cse_2 = z[m]
        __hcse_3 = __cse_2 * __cse_2d
        lossd[1] = lossd[1] + (__hcse_3 + __hcse_3)
        loss[1] = loss[1] + __cse_2 * __cse_2
    end
    for m = i_n:-1:1
        __hcse_4 = lossb[1]
        __hcse_5 = z[m]
        __cse_3d = __hcse_4 * zd[m] + __hcse_5 * lossbd[1]
        __cse_3 = __hcse_5 * __hcse_4
        zbd[m] = zbd[m] + __cse_3d
        zb[m] = zb[m] + __cse_3
        zbd[m] = zbd[m] + __cse_3d
        zb[m] = zb[m] + __cse_3
    end
    for t = i_n:-1:1
        __oldb_0d = zbd[t]
        __oldb_0 = zb[t]
        zbd[t] = 0.0
        zb[t] = 0.0
        abd = abd + __oldb_0d
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            __icse_4 = div(i_nc - 1, 1) + 1
            __icse_5 = ((t - 1) * __icse_4 + (ci - 1)) + 1
            __idx_xv_stack_0 = max(0, div(i_n - 1, 1) + 1) * max(0, __icse_4) + __icse_5
            xvd = xv_stack_d[__idx_xv_stack_0]
            xv = xv_stack[__idx_xv_stack_0]
            __cse_6d = ab * xvd + xv * abd
            __cse_6 = xv * ab
            xvbd = xvbd + __cse_6d
            xvb = xvb + __cse_6
            xvbd = xvbd + __cse_6d
            xvb = xvb + __cse_6
            __idx_branch_stack_4 = __icse_5
            __branch = branch_stack[__idx_branch_stack_4]
            if __branch == 1
                __idx_xv_stack_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
                xvd = xv_stack_d[__idx_xv_stack_0]
                xv = xv_stack[__idx_xv_stack_0]
                __oldb_2d = xvbd
                __oldb_2 = xvb
                xvbd = 0.0
                xvb = 0.0
                xvbd = xvbd + 2.0__oldb_2d
                xvb = xvb + 2.0__oldb_2
            end
            __hcse_6 = w[ci]
            __hcse_7 = xv * __hcse_6
            __cse_7d = ab * (__hcse_6 * xvd + xv * wd[ci]) + __hcse_7 * abd
            __cse_7 = __hcse_7 * ab
            xvbd = xvbd + __cse_7d
            xvb = xvb + __cse_7
            xvbd = xvbd + __cse_7d
            xvb = xvb + __cse_7
            __hcse_8 = xv * xvd
            __hcse_9 = xv * xv
            wbd[ci] = wbd[ci] + (ab * (__hcse_8 + __hcse_8) + __hcse_9 * abd)
            wb[ci] = wb[ci] + __hcse_9 * ab
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

function cross_ifwrite(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv = x[(t + ci) - 1]
            a = a + xv * xv * w[ci]
            if x[t + ci] > 0.5
                xv = xv * 2.0
            end
            a = a + xv * xv
        end
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
