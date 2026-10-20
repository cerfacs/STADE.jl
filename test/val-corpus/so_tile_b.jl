function initstacks_so_tile_b(i_n, i_nc)
    xv_stack = Vector{Float64}(undef, (((((max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1)
    wv_stack = Vector{Float64}(undef, (((((max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1)
    a_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1))
    return (xv_stack, wv_stack, a_stack)
end

function so_tile_b(x, xb, w, wb, b, bb, y, yb, z, zb, loss, lossb, i_n, i_nc, xv_stack, wv_stack, a_stack)
    a = 0.0
    wv = 0.0
    xv = 0.0
    ab = 0.0
    wvb = 0.0
    xvb = 0.0
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            __icse_0 = div(i_nc - 1, 1) + 1
            __icse_1 = ((t - 1) * __icse_0 + (ci - 1)) + 1
            __idx_xv_stack_0 = __icse_1
            xv_stack[__idx_xv_stack_0] = xv
            xv = x[(t + ci) - 1]
            __idx_wv_stack_3 = __icse_1
            wv_stack[__idx_wv_stack_3] = wv
            wv = w[(ci - 1) * 2 + 1]
            a = a + xv * wv
            __icse_2 = max(0, div(i_n - 1, 1) + 1) * max(0, __icse_0) + __icse_1
            __idx_xv_stack_7 = __icse_2
            xv_stack[__idx_xv_stack_7] = xv
            xv = x[t + ci]
            __idx_wv_stack_10 = __icse_2
            wv_stack[__idx_wv_stack_10] = wv
            wv = w[(ci - 1) * 2 + 2]
            a = a + xv * wv
        end
        y[t] = max(a + b[1], 0.0)
        __icse_3 = (t - 1) + 1
        __idx_a_stack_3 = __icse_3
        a_stack[__idx_a_stack_3] = a
        __icse_4 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)
        __icse_5 = (__icse_4 + __icse_4) + __icse_3
        __idx_wv_stack_5 = __icse_5
        wv_stack[__idx_wv_stack_5] = wv
        __idx_xv_stack_7 = __icse_5
        xv_stack[__idx_xv_stack_7] = xv
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            __icse_6 = max(0, div(i_n - 1, 1) + 1)
            __icse_7 = div(i_nc - 1, 1) + 1
            __icse_8 = __icse_6 * max(0, __icse_7)
            __icse_9 = (__icse_8 + __icse_8) + __icse_6
            __icse_10 = ((t - 1) * __icse_7 + (ci - 1)) + 1
            __icse_11 = __icse_9 + __icse_10
            __idx_xv_stack_0 = __icse_11
            xv_stack[__idx_xv_stack_0] = xv
            xv = y[(t + ci) - 1]
            __idx_wv_stack_3 = __icse_11
            wv_stack[__idx_wv_stack_3] = wv
            wv = w[(ci - 1) * 2 + 2]
            a = a + xv * wv
            __icse_12 = (__icse_9 + __icse_8) + __icse_10
            __idx_xv_stack_7 = __icse_12
            xv_stack[__idx_xv_stack_7] = xv
            xv = y[t + ci]
            __idx_wv_stack_10 = __icse_12
            wv_stack[__idx_wv_stack_10] = wv
            wv = w[(ci - 1) * 2 + 1]
            a = a + xv * wv
        end
        z[t] = a
        __icse_13 = max(0, div(i_n - 1, 1) + 1)
        __icse_14 = (t - 1) + 1
        __idx_a_stack_3 = __icse_13 + __icse_14
        a_stack[__idx_a_stack_3] = a
        __icse_15 = __icse_13 * max(0, div(i_nc - 1, 1) + 1)
        __icse_16 = ((((__icse_15 + __icse_15) + __icse_13) + __icse_15) + __icse_15) + __icse_14
        __idx_wv_stack_5 = __icse_16
        wv_stack[__idx_wv_stack_5] = wv
        __idx_xv_stack_7 = __icse_16
        xv_stack[__idx_xv_stack_7] = xv
    end
    for k = 1:i_n
        __cse_17 = z[k]
        loss[1] = loss[1] + __cse_17 * __cse_17
    end
    __icse_18 = max(0, div(i_n - 1, 1) + 1)
    __icse_19 = __icse_18 * max(0, div(i_nc - 1, 1) + 1)
    __icse_20 = (((((__icse_19 + __icse_19) + __icse_18) + __icse_19) + __icse_19) + __icse_18) + 1
    __idx_wv_stack_3 = __icse_20
    wv_stack[__idx_wv_stack_3] = wv
    __idx_xv_stack_5 = __icse_20
    xv_stack[__idx_xv_stack_5] = xv
    __idx_wv_stack_0 = __icse_20
    wv = wv_stack[__idx_wv_stack_0]
    __idx_xv_stack_2 = __icse_20
    xv = xv_stack[__idx_xv_stack_2]
    for k = i_n:-1:1
        __cse_21 = z[k] * lossb[1]
        zb[k] = zb[k] + __cse_21
        zb[k] = zb[k] + __cse_21
    end
    for t = i_n:-1:1
        __icse_22 = max(0, div(i_n - 1, 1) + 1)
        __icse_23 = (t - 1) + 1
        __idx_a_stack_0 = __icse_22 + __icse_23
        a = a_stack[__idx_a_stack_0]
        __icse_24 = __icse_22 * max(0, div(i_nc - 1, 1) + 1)
        __icse_25 = ((((__icse_24 + __icse_24) + __icse_22) + __icse_24) + __icse_24) + __icse_23
        __idx_wv_stack_2 = __icse_25
        wv = wv_stack[__idx_wv_stack_2]
        __idx_xv_stack_4 = __icse_25
        xv = xv_stack[__idx_xv_stack_4]
        __oldb_0 = zb[t]
        zb[t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            xvb = xvb + wv * ab
            wvb = wvb + xv * ab
            __icse_26 = max(0, div(i_n - 1, 1) + 1)
            __icse_27 = div(i_nc - 1, 1) + 1
            __icse_28 = __icse_26 * max(0, __icse_27)
            __icse_29 = (__icse_28 + __icse_28) + __icse_26
            __icse_30 = ((t - 1) * __icse_27 + (ci - 1)) + 1
            __icse_31 = (__icse_29 + __icse_28) + __icse_30
            __idx_wv_stack_0 = __icse_31
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[(ci - 1) * 2 + 1] = wb[(ci - 1) * 2 + 1] + __oldb_2
            __idx_xv_stack_0 = __icse_31
            xv = xv_stack[__idx_xv_stack_0]
            __oldb_2 = xvb
            xvb = 0.0
            yb[t + ci] = yb[t + ci] + __oldb_2
            xvb = xvb + wv * ab
            wvb = wvb + xv * ab
            __icse_32 = __icse_29 + __icse_30
            __idx_wv_stack_0 = __icse_32
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[(ci - 1) * 2 + 2] = wb[(ci - 1) * 2 + 2] + __oldb_2
            __idx_xv_stack_0 = __icse_32
            xv = xv_stack[__idx_xv_stack_0]
            __oldb_2 = xvb
            xvb = 0.0
            yb[(t + ci) - 1] = yb[(t + ci) - 1] + __oldb_2
        end
        ab = 0.0
    end
    for t = i_n:-1:1
        __icse_33 = (t - 1) + 1
        __idx_a_stack_0 = __icse_33
        a = a_stack[__idx_a_stack_0]
        __icse_34 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)
        __icse_35 = (__icse_34 + __icse_34) + __icse_33
        __idx_wv_stack_2 = __icse_35
        wv = wv_stack[__idx_wv_stack_2]
        __idx_xv_stack_4 = __icse_35
        xv = xv_stack[__idx_xv_stack_4]
        __oldb_0 = yb[t]
        yb[t] = 0.0
        __cse_36 = (0.5 * (1.0 + sign(a + b[1]))) * __oldb_0
        ab = ab + __cse_36
        bb[1] = bb[1] + __cse_36
        for ci = i_nc:-1:1
            xvb = xvb + wv * ab
            wvb = wvb + xv * ab
            __icse_37 = div(i_nc - 1, 1) + 1
            __icse_38 = ((t - 1) * __icse_37 + (ci - 1)) + 1
            __icse_39 = max(0, div(i_n - 1, 1) + 1) * max(0, __icse_37) + __icse_38
            __idx_wv_stack_0 = __icse_39
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[(ci - 1) * 2 + 2] = wb[(ci - 1) * 2 + 2] + __oldb_2
            __idx_xv_stack_0 = __icse_39
            xv = xv_stack[__idx_xv_stack_0]
            __oldb_2 = xvb
            xvb = 0.0
            xb[t + ci] = xb[t + ci] + __oldb_2
            xvb = xvb + wv * ab
            wvb = wvb + xv * ab
            __idx_wv_stack_0 = __icse_38
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[(ci - 1) * 2 + 1] = wb[(ci - 1) * 2 + 1] + __oldb_2
            __idx_xv_stack_0 = __icse_38
            xv = xv_stack[__idx_xv_stack_0]
            __oldb_2 = xvb
            xvb = 0.0
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + __oldb_2
        end
        ab = 0.0
    end
    return nothing
end

function so_tile(x, w, b, y, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv = x[(t + ci) - 1]
            wv = w[(ci - 1) * 2 + 1]
            a = a + xv * wv
            xv = x[t + ci]
            wv = w[(ci - 1) * 2 + 2]
            a = a + xv * wv
        end
        y[t] = max(a + b[1], 0.0)
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv = y[(t + ci) - 1]
            wv = w[(ci - 1) * 2 + 2]
            a = a + xv * wv
            xv = y[t + ci]
            wv = w[(ci - 1) * 2 + 1]
            a = a + xv * wv
        end
        z[t] = a
    end
    for k = 1:i_n
        loss[1] = loss[1] + z[k] * z[k]
    end
end
