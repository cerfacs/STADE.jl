function initstacks_cross_nested_b(i_n, i_nc)
    xv_stack = Vector{Float64}(undef, (max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) * max(0, div(2 - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1))
    return xv_stack
end

function cross_nested_b(x, xb, w, wb, z, zb, loss, lossb, i_n, i_nc, xv_stack)
    a = 0.0
    xv = 0.0
    ab = 0.0
    xvb = 0.0
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv = x[(t + ci) - 1]
            a = a + xv * xv * w[ci]
            for cj = 1:2
                __icse_0 = div(2 - 1, 1) + 1
                __idx_xv_stack_0 = ((t - 1) * ((div(i_nc - 1, 1) + 1) * __icse_0) + (ci - 1) * __icse_0 + (cj - 1)) + 1
                xv_stack[__idx_xv_stack_0] = xv
                xv = x[(t + ci + cj) - 1] * w[cj]
                a = a + xv * xv
            end
            __icse_1 = max(0, div(i_n - 1, 1) + 1)
            __icse_2 = div(i_nc - 1, 1) + 1
            __icse_3 = max(0, __icse_2)
            __icse_4 = __icse_1 * __icse_3 * max(0, div(2 - 1, 1) + 1)
            __icse_5 = ((t - 1) * __icse_2 + (ci - 1)) + 1
            __idx_xv_stack_3 = __icse_4 + __icse_5
            xv_stack[__idx_xv_stack_3] = xv
            xv = x[(t + ci) - 1] * 0.5
            a = a + xv * xv * w[ci + 1]
            __idx_xv_stack_7 = (__icse_4 + __icse_1 * __icse_3) + __icse_5
            xv_stack[__idx_xv_stack_7] = xv
        end
        z[t] = a
    end
    for m = 1:i_n
        __cse_6 = z[m]
        loss[1] = loss[1] + __cse_6 * __cse_6
    end
    for m = i_n:-1:1
        __cse_7 = z[m] * lossb[1]
        zb[m] = zb[m] + __cse_7
        zb[m] = zb[m] + __cse_7
    end
    for t = i_n:-1:1
        __oldb_0 = zb[t]
        zb[t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            __icse_8 = max(0, div(i_n - 1, 1) + 1)
            __icse_9 = div(i_nc - 1, 1) + 1
            __icse_10 = max(0, __icse_9)
            __icse_11 = __icse_8 * __icse_10 * max(0, div(2 - 1, 1) + 1)
            __icse_12 = ((t - 1) * __icse_9 + (ci - 1)) + 1
            __idx_xv_stack_0 = (__icse_11 + __icse_8 * __icse_10) + __icse_12
            xv = xv_stack[__idx_xv_stack_0]
            __cse_13 = (xv * w[ci + 1]) * ab
            xvb = xvb + __cse_13
            xvb = xvb + __cse_13
            wb[ci + 1] = wb[ci + 1] + (xv * xv) * ab
            __idx_xv_stack_0 = __icse_11 + __icse_12
            xv = xv_stack[__idx_xv_stack_0]
            __oldb_2 = xvb
            xvb = 0.0
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + 0.5__oldb_2
            for cj = 2:-1:1
                __cse_14 = xv * ab
                xvb = xvb + __cse_14
                xvb = xvb + __cse_14
                __icse_15 = div(2 - 1, 1) + 1
                __idx_xv_stack_0 = ((t - 1) * ((div(i_nc - 1, 1) + 1) * __icse_15) + (ci - 1) * __icse_15 + (cj - 1)) + 1
                xv = xv_stack[__idx_xv_stack_0]
                __oldb_2 = xvb
                xvb = 0.0
                xb[(t + ci + cj) - 1] = xb[(t + ci + cj) - 1] + w[cj] * __oldb_2
                wb[cj] = wb[cj] + x[(t + ci + cj) - 1] * __oldb_2
            end
            __cse_16 = (xv * w[ci]) * ab
            xvb = xvb + __cse_16
            xvb = xvb + __cse_16
            wb[ci] = wb[ci] + (xv * xv) * ab
            __oldb_0 = xvb
            xvb = 0.0
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + __oldb_0
        end
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
