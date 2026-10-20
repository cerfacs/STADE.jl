function initstacks_cross_branchlocal_b(i_n, i_nc)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1))
    xv_stack = Vector{Float64}(undef, ((((max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1)
    return (branch_stack, xv_stack)
end

function cross_branchlocal_b(x, xb, w, wb, z, zb, loss, lossb, i_n, i_nc, branch_stack, xv_stack)
    a = 0.0
    b = 0.0
    xv = 0.0
    ab = 0.0
    bb = 0.0
    xvb = 0.0
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            if x[(t + ci) - 1] > 0.5
                __icse_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
                __idx_branch_stack_0 = __icse_0
                branch_stack[__idx_branch_stack_0] = 1
                __idx_xv_stack_0 = __icse_0
                xv_stack[__idx_xv_stack_0] = xv
                xv = x[(t + ci) - 1] * 2.0
                a = a + xv * xv * w[ci]
            else
                __idx_branch_stack_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
                branch_stack[__idx_branch_stack_0] = 0
            end
            __icse_1 = div(i_nc - 1, 1) + 1
            __idx_xv_stack_1 = max(0, div(i_n - 1, 1) + 1) * max(0, __icse_1) + (((t - 1) * __icse_1 + (ci - 1)) + 1)
            xv_stack[__idx_xv_stack_1] = xv
        end
        z[t] = a
        __icse_2 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)
        __idx_xv_stack_3 = (__icse_2 + __icse_2) + ((t - 1) + 1)
        xv_stack[__idx_xv_stack_3] = xv
    end
    for t = 1:i_n
        b = 0.0
        for ci = 1:i_nc
            __icse_3 = max(0, div(i_n - 1, 1) + 1)
            __icse_4 = div(i_nc - 1, 1) + 1
            __icse_5 = __icse_3 * max(0, __icse_4)
            __idx_xv_stack_0 = ((__icse_5 + __icse_5) + __icse_3) + (((t - 1) * __icse_4 + (ci - 1)) + 1)
            xv_stack[__idx_xv_stack_0] = xv
            xv = x[(t + ci) - 1]
            b = b + xv * xv * w[ci]
        end
        z[i_n + t] = b
        __icse_6 = max(0, div(i_n - 1, 1) + 1)
        __icse_7 = __icse_6 * max(0, div(i_nc - 1, 1) + 1)
        __idx_xv_stack_3 = (((__icse_7 + __icse_7) + __icse_6) + __icse_7) + ((t - 1) + 1)
        xv_stack[__idx_xv_stack_3] = xv
    end
    for m = 1:2i_n
        __cse_8 = z[m]
        loss[1] = loss[1] + __cse_8 * __cse_8
    end
    __icse_9 = max(0, div(i_n - 1, 1) + 1)
    __icse_10 = __icse_9 * max(0, div(i_nc - 1, 1) + 1)
    __icse_11 = ((((__icse_10 + __icse_10) + __icse_9) + __icse_10) + __icse_9) + 1
    __idx_xv_stack_3 = __icse_11
    xv_stack[__idx_xv_stack_3] = xv
    __idx_xv_stack_0 = __icse_11
    xv = xv_stack[__idx_xv_stack_0]
    for m = 2i_n:-1:1
        __cse_12 = z[m] * lossb[1]
        zb[m] = zb[m] + __cse_12
        zb[m] = zb[m] + __cse_12
    end
    for t = i_n:-1:1
        __icse_13 = max(0, div(i_n - 1, 1) + 1)
        __icse_14 = __icse_13 * max(0, div(i_nc - 1, 1) + 1)
        __idx_xv_stack_0 = (((__icse_14 + __icse_14) + __icse_13) + __icse_14) + ((t - 1) + 1)
        xv = xv_stack[__idx_xv_stack_0]
        __oldb_0 = zb[i_n + t]
        zb[i_n + t] = 0.0
        bb = bb + __oldb_0
        for ci = i_nc:-1:1
            __cse_15 = (xv * w[ci]) * bb
            xvb = xvb + __cse_15
            xvb = xvb + __cse_15
            wb[ci] = wb[ci] + (xv * xv) * bb
            __icse_16 = max(0, div(i_n - 1, 1) + 1)
            __icse_17 = div(i_nc - 1, 1) + 1
            __icse_18 = __icse_16 * max(0, __icse_17)
            __idx_xv_stack_0 = ((__icse_18 + __icse_18) + __icse_16) + (((t - 1) * __icse_17 + (ci - 1)) + 1)
            xv = xv_stack[__idx_xv_stack_0]
            __oldb_2 = xvb
            xvb = 0.0
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + __oldb_2
        end
        bb = 0.0
    end
    for t = i_n:-1:1
        __icse_19 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)
        __idx_xv_stack_0 = (__icse_19 + __icse_19) + ((t - 1) + 1)
        xv = xv_stack[__idx_xv_stack_0]
        __oldb_0 = zb[t]
        zb[t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            __icse_20 = div(i_nc - 1, 1) + 1
            __icse_21 = ((t - 1) * __icse_20 + (ci - 1)) + 1
            __idx_xv_stack_0 = max(0, div(i_n - 1, 1) + 1) * max(0, __icse_20) + __icse_21
            xv = xv_stack[__idx_xv_stack_0]
            __idx_branch_stack_2 = __icse_21
            __branch = branch_stack[__idx_branch_stack_2]
            if __branch == 1
                __cse_22 = (xv * w[ci]) * ab
                xvb = xvb + __cse_22
                xvb = xvb + __cse_22
                wb[ci] = wb[ci] + (xv * xv) * ab
                __idx_xv_stack_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
                xv = xv_stack[__idx_xv_stack_0]
                __oldb_2 = xvb
                xvb = 0.0
                xb[(t + ci) - 1] = xb[(t + ci) - 1] + 2.0__oldb_2
            end
        end
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
