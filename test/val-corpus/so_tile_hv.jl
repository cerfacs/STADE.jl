function initstacks_so_tile_b(i_n)
    a_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1))
    return a_stack
end

function so_tile_hv(x, xb, w, wb, b, bb, y, yb, z, zb, loss, lossb, i_n, i_nc, xd, xbd, wd, wbd, bd, bbd, yd, ybd, zd, zbd, lossd, lossbd, a_stack)
    a_stack_d = Vector{Float64}(undef, length(a_stack))
    a = 0.0
    wv = 0.0
    xv = 0.0
    ab = 0.0
    wvb = 0.0
    xvb = 0.0
    ad = 0.0
    abd = 0.0
    wvd = 0.0
    wvbd = 0.0
    xvd = 0.0
    xvbd = 0.0
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            xvd = xd[(t + ci) - 1]
            xv = x[(t + ci) - 1]
            wvd = wd[(ci - 1) * 2 + 1]
            wv = w[(ci - 1) * 2 + 1]
            ad = ad + (wv * xvd + xv * wvd)
            a = a + xv * wv
            xvd = xd[t + ci]
            xv = x[t + ci]
            wvd = wd[(ci - 1) * 2 + 2]
            wv = w[(ci - 1) * 2 + 2]
            ad = ad + (wv * xvd + xv * wvd)
            a = a + xv * wv
        end
        __hcse_0 = a + b[1]
        yd[t] = (0.5 * (1.0 + sign(__hcse_0))) * (ad + bd[1])
        y[t] = max(__hcse_0, 0.0)
        __idx_a_stack_3 = (t - 1) + 1
        a_stack_d[__idx_a_stack_3] = ad
        a_stack[__idx_a_stack_3] = a
    end
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            xvd = yd[(t + ci) - 1]
            xv = y[(t + ci) - 1]
            wvd = wd[(ci - 1) * 2 + 2]
            wv = w[(ci - 1) * 2 + 2]
            ad = ad + (wv * xvd + xv * wvd)
            a = a + xv * wv
            xvd = yd[t + ci]
            xv = y[t + ci]
            wvd = wd[(ci - 1) * 2 + 1]
            wv = w[(ci - 1) * 2 + 1]
            ad = ad + (wv * xvd + xv * wvd)
            a = a + xv * wv
        end
        zd[t] = ad
        z[t] = a
        __idx_a_stack_3 = max(0, div(i_n - 1, 1) + 1) + ((t - 1) + 1)
        a_stack_d[__idx_a_stack_3] = ad
        a_stack[__idx_a_stack_3] = a
    end
    for k = 1:i_n
        __cse_0d = zd[k]
        __cse_0 = z[k]
        __hcse_1 = __cse_0 * __cse_0d
        lossd[1] = lossd[1] + (__hcse_1 + __hcse_1)
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    for k = i_n:-1:1
        __hcse_2 = lossb[1]
        __hcse_3 = z[k]
        __cse_1d = __hcse_2 * zd[k] + __hcse_3 * lossbd[1]
        __cse_1 = __hcse_3 * __hcse_2
        zbd[k] = zbd[k] + __cse_1d
        zb[k] = zb[k] + __cse_1
        zbd[k] = zbd[k] + __cse_1d
        zb[k] = zb[k] + __cse_1
    end
    for t = i_n:-1:1
        __idx_a_stack_0 = max(0, div(i_n - 1, 1) + 1) + ((t - 1) + 1)
        ad = a_stack_d[__idx_a_stack_0]
        a = a_stack[__idx_a_stack_0]
        __oldb_0d = zbd[t]
        __oldb_0 = zb[t]
        zbd[t] = 0.0
        zb[t] = 0.0
        abd = abd + __oldb_0d
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            wvd = wd[(ci - 1) * 2 + 1]
            wv = w[(ci - 1) * 2 + 1]
            xvd = yd[t + ci]
            xv = y[t + ci]
            xvbd = xvbd + (ab * wvd + wv * abd)
            xvb = xvb + wv * ab
            wvbd = wvbd + (ab * xvd + xv * abd)
            wvb = wvb + xv * ab
            wvd = wd[(ci - 1) * 2 + 2]
            wv = w[(ci - 1) * 2 + 2]
            __oldb_1d = wvbd
            __oldb_1 = wvb
            wvbd = 0.0
            wvb = 0.0
            wbd[(ci - 1) * 2 + 1] = wbd[(ci - 1) * 2 + 1] + __oldb_1d
            wb[(ci - 1) * 2 + 1] = wb[(ci - 1) * 2 + 1] + __oldb_1
            xvd = yd[(t + ci) - 1]
            xv = y[(t + ci) - 1]
            __oldb_1d = xvbd
            __oldb_1 = xvb
            xvbd = 0.0
            xvb = 0.0
            ybd[t + ci] = ybd[t + ci] + __oldb_1d
            yb[t + ci] = yb[t + ci] + __oldb_1
            xvbd = xvbd + (ab * wvd + wv * abd)
            xvb = xvb + wv * ab
            wvbd = wvbd + (ab * xvd + xv * abd)
            wvb = wvb + xv * ab
            __oldb_0d = wvbd
            __oldb_0 = wvb
            wvbd = 0.0
            wvb = 0.0
            wbd[(ci - 1) * 2 + 2] = wbd[(ci - 1) * 2 + 2] + __oldb_0d
            wb[(ci - 1) * 2 + 2] = wb[(ci - 1) * 2 + 2] + __oldb_0
            __oldb_0d = xvbd
            __oldb_0 = xvb
            xvbd = 0.0
            xvb = 0.0
            ybd[(t + ci) - 1] = ybd[(t + ci) - 1] + __oldb_0d
            yb[(t + ci) - 1] = yb[(t + ci) - 1] + __oldb_0
        end
        abd = 0.0
        ab = 0.0
    end
    for t = i_n:-1:1
        __idx_a_stack_0 = (t - 1) + 1
        ad = a_stack_d[__idx_a_stack_0]
        a = a_stack[__idx_a_stack_0]
        __oldb_0d = ybd[t]
        __oldb_0 = yb[t]
        ybd[t] = 0.0
        yb[t] = 0.0
        __hcse_4 = 0.5 * (1.0 + sign(a + b[1]))
        __cse_2d = __hcse_4 * __oldb_0d
        __cse_2 = __hcse_4 * __oldb_0
        abd = abd + __cse_2d
        ab = ab + __cse_2
        bbd[1] = bbd[1] + __cse_2d
        bb[1] = bb[1] + __cse_2
        for ci = i_nc:-1:1
            wvd = wd[(ci - 1) * 2 + 2]
            wv = w[(ci - 1) * 2 + 2]
            xvd = xd[t + ci]
            xv = x[t + ci]
            xvbd = xvbd + (ab * wvd + wv * abd)
            xvb = xvb + wv * ab
            wvbd = wvbd + (ab * xvd + xv * abd)
            wvb = wvb + xv * ab
            wvd = wd[(ci - 1) * 2 + 1]
            wv = w[(ci - 1) * 2 + 1]
            __oldb_1d = wvbd
            __oldb_1 = wvb
            wvbd = 0.0
            wvb = 0.0
            wbd[(ci - 1) * 2 + 2] = wbd[(ci - 1) * 2 + 2] + __oldb_1d
            wb[(ci - 1) * 2 + 2] = wb[(ci - 1) * 2 + 2] + __oldb_1
            xvd = xd[(t + ci) - 1]
            xv = x[(t + ci) - 1]
            __oldb_1d = xvbd
            __oldb_1 = xvb
            xvbd = 0.0
            xvb = 0.0
            xbd[t + ci] = xbd[t + ci] + __oldb_1d
            xb[t + ci] = xb[t + ci] + __oldb_1
            xvbd = xvbd + (ab * wvd + wv * abd)
            xvb = xvb + wv * ab
            wvbd = wvbd + (ab * xvd + xv * abd)
            wvb = wvb + xv * ab
            __oldb_0d = wvbd
            __oldb_0 = wvb
            wvbd = 0.0
            wvb = 0.0
            wbd[(ci - 1) * 2 + 1] = wbd[(ci - 1) * 2 + 1] + __oldb_0d
            wb[(ci - 1) * 2 + 1] = wb[(ci - 1) * 2 + 1] + __oldb_0
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
