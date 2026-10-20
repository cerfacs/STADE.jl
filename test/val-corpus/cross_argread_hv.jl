function initstacks_cross_argread_b(i_nc)
    s_stack = Vector{Float64}(undef, max(0, div(i_nc - 1, 1) + 1) + 1)
    a_stack = Vector{Float64}(undef, 1)
    return (s_stack, a_stack)
end

function cross_argread_hv(x, xb, w, wb, loss, lossb, s, sb, i_nc, xd, xbd, wd, wbd, lossd, lossbd, sd, sbd, s_stack, a_stack)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    a_stack_d = Vector{Float64}(undef, length(a_stack))
    a = 0.0
    ab = 0.0
    ad = 0.0
    abd = 0.0
    __hcse_0 = s * sd
    ad = __hcse_0 + __hcse_0
    a = s * s
    for ci = 1:i_nc
        __idx_s_stack_0 = (ci - 1) + 1
        s_stack_d[__idx_s_stack_0] = sd
        s_stack[__idx_s_stack_0] = s
        sd = 2.0 * xd[ci]
        s = x[ci] * 2.0
        __hcse_1 = w[ci]
        __hcse_2 = (s * __hcse_1) * sd
        ad = ad + ((__hcse_2 + __hcse_2) + (s * s) * wd[ci])
        a = a + s * s * __hcse_1
    end
    __hcse_3 = a * ad
    lossd[1] = lossd[1] + (__hcse_3 + __hcse_3)
    loss[1] = loss[1] + a * a
    a_stack_d[1] = ad
    a_stack[1] = a
    __ihcse_4 = max(0, div(i_nc - 1, 1) + 1) + 1
    __idx_s_stack_4 = __ihcse_4
    s_stack_d[__idx_s_stack_4] = sd
    s_stack[__idx_s_stack_4] = s
    ad = a_stack_d[1]
    a = a_stack[1]
    __idx_s_stack_1 = __ihcse_4
    sd = s_stack_d[__idx_s_stack_1]
    s = s_stack[__idx_s_stack_1]
    __hcse_5 = lossb[1]
    __cse_0d = __hcse_5 * ad + a * lossbd[1]
    __cse_0 = a * __hcse_5
    abd = abd + __cse_0d
    ab = ab + __cse_0
    abd = abd + __cse_0d
    ab = ab + __cse_0
    for ci = i_nc:-1:1
        __hcse_6 = w[ci]
        __hcse_7 = s * __hcse_6
        __cse_1d = ab * (__hcse_6 * sd + s * wd[ci]) + __hcse_7 * abd
        __cse_1 = __hcse_7 * ab
        sbd = sbd + __cse_1d
        sb = sb + __cse_1
        sbd = sbd + __cse_1d
        sb = sb + __cse_1
        __hcse_8 = s * sd
        __hcse_9 = s * s
        wbd[ci] = wbd[ci] + (ab * (__hcse_8 + __hcse_8) + __hcse_9 * abd)
        wb[ci] = wb[ci] + __hcse_9 * ab
        __idx_s_stack_0 = (ci - 1) + 1
        sd = s_stack_d[__idx_s_stack_0]
        s = s_stack[__idx_s_stack_0]
        __oldb_2d = sbd
        __oldb_2 = sb
        sbd = 0.0
        sb = 0.0
        xbd[ci] = xbd[ci] + 2.0__oldb_2d
        xb[ci] = xb[ci] + 2.0__oldb_2
    end
    __oldb_0d = abd
    __oldb_0 = ab
    abd = 0.0
    ab = 0.0
    __cse_2d = __oldb_0 * sd + s * __oldb_0d
    __cse_2 = s * __oldb_0
    sbd = sbd + __cse_2d
    sb = sb + __cse_2
    sbd = sbd + __cse_2d
    sb = sb + __cse_2
    return (sb, sbd)
end

function cross_argread(x, w, loss, s, i_nc)
    a = s * s
    for ci = 1:i_nc
        s = x[ci] * 2.0
        a = a + s * s * w[ci]
    end
    loss[1] = loss[1] + a * a
end
