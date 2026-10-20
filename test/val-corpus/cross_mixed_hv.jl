function initstacks_cross_mixed_b(i_nc)
    a_stack = Vector{Float64}(undef, 1)
    b_stack = Vector{Float64}(undef, 1)
    return (a_stack, b_stack)
end

function cross_mixed_hv(x, xb, w, wb, loss, lossb, i_nc, xd, xbd, wd, wbd, lossd, lossbd, a_stack, b_stack)
    a_stack_d = Vector{Float64}(undef, length(a_stack))
    b_stack_d = Vector{Float64}(undef, length(b_stack))
    a = 0.0
    b = 0.0
    tmp = 0.0
    xv = 0.0
    ab = 0.0
    bb = 0.0
    tmpb = 0.0
    xvb = 0.0
    ad = 0.0
    abd = 0.0
    bd = 0.0
    bbd = 0.0
    tmpd = 0.0
    tmpbd = 0.0
    xvd = 0.0
    xvbd = 0.0
    ad = 0.0
    a = 0.0
    for ci = 1:i_nc
        __cse_0d = wd[ci]
        __cse_0 = w[ci]
        __hcse_0 = x[ci]
        tmpd = __cse_0 * xd[ci] + __hcse_0 * __cse_0d
        tmp = __hcse_0 * __cse_0
        xvd = 0.5tmpd
        xv = tmp * 0.5
        __hcse_1 = (xv * __cse_0) * xvd
        ad = ad + ((__hcse_1 + __hcse_1) + (xv * xv) * __cse_0d)
        a = a + xv * xv * __cse_0
    end
    bd = 0.0
    b = 0.0
    for ci = 1:i_nc
        xvd = xd[ci]
        xv = x[ci]
        __hcse_2 = w[ci]
        __hcse_3 = (xv * __hcse_2) * xvd
        bd = bd + ((__hcse_3 + __hcse_3) + (xv * xv) * wd[ci])
        b = b + xv * xv * __hcse_2
    end
    __hcse_4 = a * ad
    __hcse_5 = b * bd
    lossd[1] = (lossd[1] + (__hcse_4 + __hcse_4)) + (__hcse_5 + __hcse_5)
    loss[1] = loss[1] + a * a + b * b
    a_stack_d[1] = ad
    a_stack[1] = a
    b_stack_d[1] = bd
    b_stack[1] = b
    ad = a_stack_d[1]
    a = a_stack[1]
    bd = b_stack_d[1]
    b = b_stack[1]
    __cse_1d = lossbd[1]
    __cse_1 = lossb[1]
    __cse_2d = __cse_1 * ad + a * __cse_1d
    __cse_2 = a * __cse_1
    abd = abd + __cse_2d
    ab = ab + __cse_2
    abd = abd + __cse_2d
    ab = ab + __cse_2
    __cse_3d = __cse_1 * bd + b * __cse_1d
    __cse_3 = b * __cse_1
    bbd = bbd + __cse_3d
    bb = bb + __cse_3
    bbd = bbd + __cse_3d
    bb = bb + __cse_3
    for ci = 1:i_nc
        xvd = xd[ci]
        xv = x[ci]
        __cse_4d = wd[ci]
        __cse_4 = w[ci]
        __hcse_6 = xv * __cse_4
        __hcse_7 = __hcse_6 * xvd
        __hcse_8 = xv * xv
        bd = bd + ((__hcse_7 + __hcse_7) + __hcse_8 * __cse_4d)
        b = b + xv * xv * __cse_4
        __cse_5d = bb * (__cse_4 * xvd + xv * __cse_4d) + __hcse_6 * bbd
        __cse_5 = __hcse_6 * bb
        xvbd = xvbd + __cse_5d
        xvb = xvb + __cse_5
        xvbd = xvbd + __cse_5d
        xvb = xvb + __cse_5
        __hcse_9 = xv * xvd
        wbd[ci] = wbd[ci] + (bb * (__hcse_9 + __hcse_9) + __hcse_8 * bbd)
        wb[ci] = wb[ci] + __hcse_8 * bb
        __oldb_0d = xvbd
        __oldb_0 = xvb
        xvbd = 0.0
        xvb = 0.0
        xbd[ci] = xbd[ci] + __oldb_0d
        xb[ci] = xb[ci] + __oldb_0
    end
    bbd = 0.0
    bb = 0.0
    for ci = 1:i_nc
        __cse_6d = xd[ci]
        __cse_6 = x[ci]
        __cse_7d = wd[ci]
        __cse_7 = w[ci]
        __cse_8d = __cse_7 * __cse_6d + __cse_6 * __cse_7d
        __cse_8 = __cse_6 * __cse_7
        tmpd = __cse_8d
        tmp = __cse_8
        xvd = 0.5tmpd
        xv = tmp * 0.5
        __hcse_10 = xv * __cse_7
        __hcse_11 = __hcse_10 * xvd
        __hcse_12 = xv * xv
        ad = ad + ((__hcse_11 + __hcse_11) + __hcse_12 * __cse_7d)
        a = a + xv * xv * __cse_7
        tmpd = __cse_8d
        tmp = __cse_8
        __cse_9d = ab * (__cse_7 * xvd + xv * __cse_7d) + __hcse_10 * abd
        __cse_9 = __hcse_10 * ab
        xvbd = xvbd + __cse_9d
        xvb = xvb + __cse_9
        xvbd = xvbd + __cse_9d
        xvb = xvb + __cse_9
        __hcse_13 = xv * xvd
        wbd[ci] = wbd[ci] + (ab * (__hcse_13 + __hcse_13) + __hcse_12 * abd)
        wb[ci] = wb[ci] + __hcse_12 * ab
        __oldb_0d = xvbd
        __oldb_0 = xvb
        xvbd = 0.0
        xvb = 0.0
        tmpbd = tmpbd + 0.5__oldb_0d
        tmpb = tmpb + 0.5__oldb_0
        __oldb_0d = tmpbd
        __oldb_0 = tmpb
        tmpbd = 0.0
        tmpb = 0.0
        xbd[ci] = xbd[ci] + (__oldb_0 * __cse_7d + __cse_7 * __oldb_0d)
        xb[ci] = xb[ci] + __cse_7 * __oldb_0
        wbd[ci] = wbd[ci] + (__oldb_0 * __cse_6d + __cse_6 * __oldb_0d)
        wb[ci] = wb[ci] + __cse_6 * __oldb_0
    end
    abd = 0.0
    ab = 0.0
    return nothing
end

function cross_mixed(x, w, loss, i_nc)
    a = 0.0
    for ci = 1:i_nc
        tmp = x[ci] * w[ci]
        xv = tmp * 0.5
        a = a + xv * xv * w[ci]
    end
    b = 0.0
    for ci = 1:i_nc
        xv = x[ci]
        b = b + xv * xv * w[ci]
    end
    loss[1] = loss[1] + a * a + b * b
end
