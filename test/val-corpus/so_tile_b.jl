function initstacks_so_tile_b(i_n)
    a_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1))
    return a_stack
end

function so_tile_b(x, xb, w, wb, b, bb, y, yb, z, zb, loss, lossb, i_n, i_nc, a_stack)
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
        __idx_a_stack_3 = (t - 1) + 1
        a_stack[__idx_a_stack_3] = a
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
        __idx_a_stack_3 = max(0, div(i_n - 1, 1) + 1) + ((t - 1) + 1)
        a_stack[__idx_a_stack_3] = a
    end
    for k = 1:i_n
        __cse_0 = z[k]
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    for k = i_n:-1:1
        __cse_1 = z[k] * lossb[1]
        zb[k] = zb[k] + __cse_1
        zb[k] = zb[k] + __cse_1
    end
    for t = i_n:-1:1
        __idx_a_stack_0 = max(0, div(i_n - 1, 1) + 1) + ((t - 1) + 1)
        a = a_stack[__idx_a_stack_0]
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
        __idx_a_stack_0 = (t - 1) + 1
        a = a_stack[__idx_a_stack_0]
        __oldb_0 = yb[t]
        yb[t] = 0.0
        __cse_2 = (0.5 * (1.0 + sign(a + b[1]))) * __oldb_0
        ab = ab + __cse_2
        bb[1] = bb[1] + __cse_2
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
