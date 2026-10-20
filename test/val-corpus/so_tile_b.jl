function initstacks_so_tile_b(i_n)
    xv_stack = Vector{Float64}(undef, (max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1)
    wv_stack = Vector{Float64}(undef, (max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1)
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
            xv = x[(t + ci) - 1]
            wv = w[(ci - 1) * 2 + 1]
            a = a + xv * wv
            xv = x[t + ci]
            wv = w[(ci - 1) * 2 + 2]
            a = a + xv * wv
        end
        y[t] = max(a + b[1], 0.0)
        __icse_0 = (t - 1) + 1
        __idx_a_stack_3 = __icse_0
        a_stack[__idx_a_stack_3] = a
        __idx_wv_stack_5 = __icse_0
        wv_stack[__idx_wv_stack_5] = wv
        __idx_xv_stack_7 = __icse_0
        xv_stack[__idx_xv_stack_7] = xv
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
        __icse_1 = max(0, div(i_n - 1, 1) + 1) + ((t - 1) + 1)
        __idx_a_stack_3 = __icse_1
        a_stack[__idx_a_stack_3] = a
        __idx_wv_stack_5 = __icse_1
        wv_stack[__idx_wv_stack_5] = wv
        __idx_xv_stack_7 = __icse_1
        xv_stack[__idx_xv_stack_7] = xv
    end
    for k = 1:i_n
        __cse_2 = z[k]
        loss[1] = loss[1] + __cse_2 * __cse_2
    end
    __icse_3 = max(0, div(i_n - 1, 1) + 1)
    __icse_4 = (__icse_3 + __icse_3) + 1
    __idx_wv_stack_3 = __icse_4
    wv_stack[__idx_wv_stack_3] = wv
    __idx_xv_stack_5 = __icse_4
    xv_stack[__idx_xv_stack_5] = xv
    __idx_wv_stack_0 = __icse_4
    wv = wv_stack[__idx_wv_stack_0]
    __idx_xv_stack_2 = __icse_4
    xv = xv_stack[__idx_xv_stack_2]
    for k = i_n:-1:1
        __cse_5 = z[k] * lossb[1]
        zb[k] = zb[k] + __cse_5
        zb[k] = zb[k] + __cse_5
    end
    for t = i_n:-1:1
        __icse_6 = max(0, div(i_n - 1, 1) + 1) + ((t - 1) + 1)
        __idx_a_stack_0 = __icse_6
        a = a_stack[__idx_a_stack_0]
        __idx_wv_stack_2 = __icse_6
        wv = wv_stack[__idx_wv_stack_2]
        __idx_xv_stack_4 = __icse_6
        xv = xv_stack[__idx_xv_stack_4]
        __oldb_0 = zb[t]
        zb[t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            wv = w[(ci - 1) * 2 + 1]
            xv = y[t + ci]
            xvb = xvb + wv * ab
            wvb = wvb + xv * ab
            wv = w[(ci - 1) * 2 + 2]
            __oldb_1 = wvb
            wvb = 0.0
            wb[(ci - 1) * 2 + 1] = wb[(ci - 1) * 2 + 1] + __oldb_1
            xv = y[(t + ci) - 1]
            __oldb_1 = xvb
            xvb = 0.0
            yb[t + ci] = yb[t + ci] + __oldb_1
            xvb = xvb + wv * ab
            wvb = wvb + xv * ab
            __oldb_0 = wvb
            wvb = 0.0
            wb[(ci - 1) * 2 + 2] = wb[(ci - 1) * 2 + 2] + __oldb_0
            __oldb_0 = xvb
            xvb = 0.0
            yb[(t + ci) - 1] = yb[(t + ci) - 1] + __oldb_0
        end
        ab = 0.0
    end
    for t = i_n:-1:1
        __icse_7 = (t - 1) + 1
        __idx_a_stack_0 = __icse_7
        a = a_stack[__idx_a_stack_0]
        __idx_wv_stack_2 = __icse_7
        wv = wv_stack[__idx_wv_stack_2]
        __idx_xv_stack_4 = __icse_7
        xv = xv_stack[__idx_xv_stack_4]
        __oldb_0 = yb[t]
        yb[t] = 0.0
        __cse_8 = (0.5 * (1.0 + sign(a + b[1]))) * __oldb_0
        ab = ab + __cse_8
        bb[1] = bb[1] + __cse_8
        for ci = i_nc:-1:1
            wv = w[(ci - 1) * 2 + 2]
            xv = x[t + ci]
            xvb = xvb + wv * ab
            wvb = wvb + xv * ab
            wv = w[(ci - 1) * 2 + 1]
            __oldb_1 = wvb
            wvb = 0.0
            wb[(ci - 1) * 2 + 2] = wb[(ci - 1) * 2 + 2] + __oldb_1
            xv = x[(t + ci) - 1]
            __oldb_1 = xvb
            xvb = 0.0
            xb[t + ci] = xb[t + ci] + __oldb_1
            xvb = xvb + wv * ab
            wvb = wvb + xv * ab
            __oldb_0 = wvb
            wvb = 0.0
            wb[(ci - 1) * 2 + 1] = wb[(ci - 1) * 2 + 1] + __oldb_0
            __oldb_0 = xvb
            xvb = 0.0
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + __oldb_0
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
