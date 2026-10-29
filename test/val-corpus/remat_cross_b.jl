function initstacks_remat_cross_b(i_n, i_nc)
    xv5_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1))
    xv6_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1))
    xv7_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) * max(0, div(2 - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1))
    xv8_stack = Vector{Float64}(undef, (max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1)
    return (xv5_stack, xv6_stack, xv7_stack, xv8_stack)
end

function remat_cross_b(x, xb, w, wb, y, yb, u, ub, z, zb, loss, lossb, i_n, i_nc, xv5_stack, xv6_stack, xv7_stack, xv8_stack)
    a = 0.0
    wv = 0.0
    wv2 = 0.0
    xv = 0.0
    xv2 = 0.0
    xv3 = 0.0
    xv4 = 0.0
    xv5 = 0.0
    xv6 = 0.0
    xv7 = 0.0
    xv8 = 0.0
    ab = 0.0
    wvb = 0.0
    wv2b = 0.0
    xvb = 0.0
    xv2b = 0.0
    xv3b = 0.0
    xv4b = 0.0
    xv5b = 0.0
    xv6b = 0.0
    xv7b = 0.0
    xv8b = 0.0
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv = x[(t + ci) - 1]
            wv = w[2ci - 1]
            a = a + xv * wv * wv
            wv = w[2ci]
            a = a + xv * wv * wv
        end
        y[t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            b1 = ci * 2
            b2 = b1 + 1
            xv2 = x[(t + ci) - 1] * 2.0
            wv2 = w[b2]
            a = a + xv2 * wv2 * wv2
        end
        z[t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 2:2:i_nc
            xv3 = x[(t + ci) - 1]
            a = a + xv3 * xv3 * w[ci]
        end
        for cj = i_nc:-1:1
            xv4 = x[t + cj]
            a = a + xv4 * xv4 * w[cj + 1]
        end
        z[i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        xv5 = x[t] * 0.5
        for ci = 1:i_nc
            a = a + xv5 * xv5
            __idx_xv5_stack_1 = ((t - 1) + (ci - 1) * (div(i_n - 1, 1) + 1)) + 1
            xv5_stack[__idx_xv5_stack_1] = xv5
            xv5 = x[(t + ci) - 1] * w[ci]
            a = a + xv5 * w[ci + 1]
        end
        z[2i_n + t] = a
        __idx_xv5_stack_4 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + ((t - 1) + 1)
        xv5_stack[__idx_xv5_stack_4] = xv5
    end
    for t = 1:i_n
        a = 0.0
        xv6 = x[t] * w[1]
        a = xv6 * xv6
        for ci = 1:i_nc
            __idx_xv6_stack_0 = ((t - 1) + (ci - 1) * (div(i_n - 1, 1) + 1)) + 1
            xv6_stack[__idx_xv6_stack_0] = xv6
            xv6 = x[t + ci] * w[ci]
            a = a + xv6 * xv6 * xv6
        end
        z[3i_n + t] = a
        __idx_xv6_stack_5 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + ((t - 1) + 1)
        xv6_stack[__idx_xv6_stack_5] = xv6
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv7 = x[(t + ci) - 1]
            a = a + xv7 * xv7
            for cj = 1:2
                __idx_xv7_stack_0 = ((t - 1) + ((ci - 1) * (div(2 - 1, 1) + 1) + (cj - 1)) * (div(i_n - 1, 1) + 1)) + 1
                xv7_stack[__idx_xv7_stack_0] = xv7
                xv7 = xv7 * w[ci + cj]
            end
            a = a + xv7 * xv7
            __icse_0 = div(i_n - 1, 1) + 1
            __idx_xv7_stack_4 = max(0, __icse_0) * max(0, div(i_nc - 1, 1) + 1) * max(0, div(2 - 1, 1) + 1) + (((t - 1) + (ci - 1) * __icse_0) + 1)
            xv7_stack[__idx_xv7_stack_4] = xv7
        end
        z[4i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            __idx_xv8_stack_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
            xv8_stack[__idx_xv8_stack_0] = xv8
            xv8 = u[ci]
            a = a + xv8 * xv8 * x[(t + ci) - 1]
            u[ci + 1] = x[t + ci] * 0.5
        end
        z[5i_n + t] = a
        __idx_xv8_stack_3 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + ((t - 1) + 1)
        xv8_stack[__idx_xv8_stack_3] = xv8
    end
    for m = 1:i_n
        __cse_1 = y[m]
        loss[1] = loss[1] + __cse_1 * __cse_1
    end
    for m = 1:6i_n
        __cse_2 = z[m]
        loss[1] = loss[1] + __cse_2 * __cse_2
    end
    __icse_3 = max(0, div(i_n - 1, 1) + 1)
    __icse_4 = (__icse_3 * max(0, div(i_nc - 1, 1) + 1) + __icse_3) + 1
    __idx_xv8_stack_9 = __icse_4
    xv8_stack[__idx_xv8_stack_9] = xv8
    __idx_xv8_stack_0 = __icse_4
    xv8 = xv8_stack[__idx_xv8_stack_0]
    for m = 6i_n:-1:1
        __cse_5 = z[m] * lossb[1]
        zb[m] = zb[m] + __cse_5
        zb[m] = zb[m] + __cse_5
    end
    for m = i_n:-1:1
        __cse_6 = y[m] * lossb[1]
        yb[m] = yb[m] + __cse_6
        yb[m] = yb[m] + __cse_6
    end
    for t = i_n:-1:1
        __idx_xv8_stack_0 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + ((t - 1) + 1)
        xv8 = xv8_stack[__idx_xv8_stack_0]
        __oldb_0 = zb[5i_n + t]
        zb[5i_n + t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            __oldb_0 = ub[ci + 1]
            ub[ci + 1] = 0.0
            xb[t + ci] = xb[t + ci] + 0.5__oldb_0
            __cse_7 = (xv8 * x[(t + ci) - 1]) * ab
            xv8b = xv8b + __cse_7
            xv8b = xv8b + __cse_7
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + (xv8 * xv8) * ab
            __idx_xv8_stack_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
            xv8 = xv8_stack[__idx_xv8_stack_0]
            __oldb_2 = xv8b
            xv8b = 0.0
            ub[ci] = ub[ci] + __oldb_2
        end
        ab = 0.0
    end
    for t = i_n:-1:1
        __oldb_0 = zb[4i_n + t]
        zb[4i_n + t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            __icse_8 = div(i_n - 1, 1) + 1
            __idx_xv7_stack_0 = max(0, __icse_8) * max(0, div(i_nc - 1, 1) + 1) * max(0, div(2 - 1, 1) + 1) + (((t - 1) + (ci - 1) * __icse_8) + 1)
            xv7 = xv7_stack[__idx_xv7_stack_0]
            __cse_9 = xv7 * ab
            xv7b = xv7b + __cse_9
            xv7b = xv7b + __cse_9
            for cj = 2:-1:1
                __idx_xv7_stack_0 = ((t - 1) + ((ci - 1) * (div(2 - 1, 1) + 1) + (cj - 1)) * (div(i_n - 1, 1) + 1)) + 1
                xv7 = xv7_stack[__idx_xv7_stack_0]
                __oldb_2 = xv7b
                xv7b = 0.0
                xv7b = xv7b + w[ci + cj] * __oldb_2
                wb[ci + cj] = wb[ci + cj] + xv7 * __oldb_2
            end
            __cse_10 = xv7 * ab
            xv7b = xv7b + __cse_10
            xv7b = xv7b + __cse_10
            __oldb_0 = xv7b
            xv7b = 0.0
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + __oldb_0
        end
        ab = 0.0
    end
    for t = i_n:-1:1
        __idx_xv6_stack_0 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + ((t - 1) + 1)
        xv6 = xv6_stack[__idx_xv6_stack_0]
        __oldb_0 = zb[3i_n + t]
        zb[3i_n + t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            __cse_11 = (xv6 * xv6) * ab
            xv6b = xv6b + __cse_11
            xv6b = xv6b + __cse_11
            xv6b = xv6b + __cse_11
            __idx_xv6_stack_0 = ((t - 1) + (ci - 1) * (div(i_n - 1, 1) + 1)) + 1
            xv6 = xv6_stack[__idx_xv6_stack_0]
            __oldb_2 = xv6b
            xv6b = 0.0
            xb[t + ci] = xb[t + ci] + w[ci] * __oldb_2
            wb[ci] = wb[ci] + x[t + ci] * __oldb_2
        end
        __oldb_0 = ab
        ab = 0.0
        __cse_12 = xv6 * __oldb_0
        xv6b = xv6b + __cse_12
        xv6b = xv6b + __cse_12
        __oldb_0 = xv6b
        xv6b = 0.0
        xb[t] = xb[t] + w[1] * __oldb_0
        wb[1] = wb[1] + x[t] * __oldb_0
        ab = 0.0
    end
    for t = i_n:-1:1
        __idx_xv5_stack_0 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + ((t - 1) + 1)
        xv5 = xv5_stack[__idx_xv5_stack_0]
        __oldb_0 = zb[2i_n + t]
        zb[2i_n + t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            xv5b = xv5b + w[ci + 1] * ab
            wb[ci + 1] = wb[ci + 1] + xv5 * ab
            __idx_xv5_stack_0 = ((t - 1) + (ci - 1) * (div(i_n - 1, 1) + 1)) + 1
            xv5 = xv5_stack[__idx_xv5_stack_0]
            __oldb_2 = xv5b
            xv5b = 0.0
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + w[ci] * __oldb_2
            wb[ci] = wb[ci] + x[(t + ci) - 1] * __oldb_2
            __cse_13 = xv5 * ab
            xv5b = xv5b + __cse_13
            xv5b = xv5b + __cse_13
        end
        __oldb_0 = xv5b
        xv5b = 0.0
        xb[t] = xb[t] + 0.5__oldb_0
        ab = 0.0
    end
    for t = 1:i_n
        a = 0.0
        for ci = 2:2:i_nc
            xv3 = x[(t + ci) - 1]
            a = a + xv3 * xv3 * w[ci]
        end
        for cj = i_nc:-1:1
            xv4 = x[t + cj]
            a = a + xv4 * xv4 * w[cj + 1]
        end
        __oldb_0 = zb[i_n + t]
        zb[i_n + t] = 0.0
        ab = ab + __oldb_0
        for cj = i_nc:-1:1
            __cse_14 = x[t + cj]
            xv4 = __cse_14
            __cse_15 = w[cj + 1]
            a = a + xv4 * xv4 * __cse_15
            xv4 = __cse_14
            __cse_16 = (xv4 * __cse_15) * ab
            xv4b = xv4b + __cse_16
            xv4b = xv4b + __cse_16
            wb[cj + 1] = wb[cj + 1] + (xv4 * xv4) * ab
            __oldb_0 = xv4b
            xv4b = 0.0
            xb[t + cj] = xb[t + cj] + __oldb_0
        end
        for ci = 2:2:i_nc
            __cse_17 = x[(t + ci) - 1]
            xv3 = __cse_17
            __cse_18 = w[ci]
            a = a + xv3 * xv3 * __cse_18
            xv3 = __cse_17
            __cse_19 = (xv3 * __cse_18) * ab
            xv3b = xv3b + __cse_19
            xv3b = xv3b + __cse_19
            wb[ci] = wb[ci] + (xv3 * xv3) * ab
            __oldb_0 = xv3b
            xv3b = 0.0
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + __oldb_0
        end
        ab = 0.0
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            b1 = ci * 2
            b2 = b1 + 1
            xv2 = x[(t + ci) - 1] * 2.0
            wv2 = w[b2]
            a = a + xv2 * wv2 * wv2
        end
        __oldb_0 = zb[t]
        zb[t] = 0.0
        ab = ab + __oldb_0
        for ci = 1:i_nc
            __icse_20 = ci * 2
            b1 = __icse_20
            b2 = b1 + 1
            __cse_21 = x[(t + ci) - 1] * 2.0
            xv2 = __cse_21
            wv2 = w[b2]
            a = a + xv2 * wv2 * wv2
            b1 = __icse_20
            b2 = b1 + 1
            wv2 = w[b2]
            xv2 = __cse_21
            xv2b = xv2b + (wv2 * wv2) * ab
            __cse_22 = (xv2 * wv2) * ab
            wv2b = wv2b + __cse_22
            wv2b = wv2b + __cse_22
            __oldb_0 = wv2b
            wv2b = 0.0
            wb[b2] = wb[b2] + __oldb_0
            __oldb_0 = xv2b
            xv2b = 0.0
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + 2.0__oldb_0
        end
        ab = 0.0
    end
    for t = i_n:-1:1
        __oldb_0 = yb[t]
        yb[t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            wv = w[2ci]
            xv = x[(t + ci) - 1]
            xvb = xvb + (wv * wv) * ab
            __cse_23 = (xv * wv) * ab
            wvb = wvb + __cse_23
            wvb = wvb + __cse_23
            wv = w[2ci - 1]
            __oldb_1 = wvb
            wvb = 0.0
            wb[2ci] = wb[2ci] + __oldb_1
            xvb = xvb + (wv * wv) * ab
            __cse_24 = (xv * wv) * ab
            wvb = wvb + __cse_24
            wvb = wvb + __cse_24
            __oldb_0 = wvb
            wvb = 0.0
            wb[2ci - 1] = wb[2ci - 1] + __oldb_0
            __oldb_0 = xvb
            xvb = 0.0
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + __oldb_0
        end
        ab = 0.0
    end
    return nothing
end

function remat_cross(x, w, y, u, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv = x[(t + ci) - 1]
            wv = w[2ci - 1]
            a = a + xv * wv * wv
            wv = w[2ci]
            a = a + xv * wv * wv
        end
        y[t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            b1 = ci * 2
            b2 = b1 + 1
            xv2 = x[(t + ci) - 1] * 2.0
            wv2 = w[b2]
            a = a + xv2 * wv2 * wv2
        end
        z[t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 2:2:i_nc
            xv3 = x[(t + ci) - 1]
            a = a + xv3 * xv3 * w[ci]
        end
        for cj = i_nc:-1:1
            xv4 = x[t + cj]
            a = a + xv4 * xv4 * w[cj + 1]
        end
        z[i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        xv5 = x[t] * 0.5
        for ci = 1:i_nc
            a = a + xv5 * xv5
            xv5 = x[(t + ci) - 1] * w[ci]
            a = a + xv5 * w[ci + 1]
        end
        z[2i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        xv6 = x[t] * w[1]
        a = xv6 * xv6
        for ci = 1:i_nc
            xv6 = x[t + ci] * w[ci]
            a = a + xv6 * xv6 * xv6
        end
        z[3i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv7 = x[(t + ci) - 1]
            a = a + xv7 * xv7
            for cj = 1:2
                xv7 = xv7 * w[ci + cj]
            end
            a = a + xv7 * xv7
        end
        z[4i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv8 = u[ci]
            a = a + xv8 * xv8 * x[(t + ci) - 1]
            u[ci + 1] = x[t + ci] * 0.5
        end
        z[5i_n + t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + y[m] * y[m]
    end
    for m = 1:6i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
