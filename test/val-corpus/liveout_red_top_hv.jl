function initstacks_liveout_red_top_b()
    a_stack = Vector{Float64}(undef, 1)
    return a_stack
end

function liveout_red_top_hv(x, xb, w, wb, loss, lossb, i_nc, xd, xbd, wd, wbd, lossd, lossbd, a_stack)
    a_stack_d = Vector{Float64}(undef, length(a_stack))
    a = 0.0
    xv2 = 0.0
    ab = 0.0
    xv2b = 0.0
    ad = 0.0
    abd = 0.0
    xv2d = 0.0
    xv2bd = 0.0
    ad = 0.0
    a = 0.0
    for ci = 1:i_nc
        xv2d = xd[ci]
        xv2 = x[ci]
        __hcse_0 = w[ci]
        __hcse_1 = (xv2 * __hcse_0) * xv2d
        ad = ad + ((__hcse_1 + __hcse_1) + (xv2 * xv2) * wd[ci])
        a = a + xv2 * xv2 * __hcse_0
    end
    __hcse_2 = a * ad
    lossd[1] = lossd[1] + (__hcse_2 + __hcse_2)
    loss[1] = loss[1] + a * a
    a_stack_d[1] = ad
    a_stack[1] = a
    ad = a_stack_d[1]
    a = a_stack[1]
    __hcse_3 = lossb[1]
    __cse_0d = __hcse_3 * ad + a * lossbd[1]
    __cse_0 = a * __hcse_3
    abd = abd + __cse_0d
    ab = ab + __cse_0
    abd = abd + __cse_0d
    ab = ab + __cse_0
    for ci = 1:i_nc
        __cse_1d = xd[ci]
        __cse_1 = x[ci]
        xv2d = __cse_1d
        xv2 = __cse_1
        __cse_2d = wd[ci]
        __cse_2 = w[ci]
        __hcse_4 = (xv2 * __cse_2) * xv2d
        ad = ad + ((__hcse_4 + __hcse_4) + (xv2 * xv2) * __cse_2d)
        a = a + xv2 * xv2 * __cse_2
        xv2d = __cse_1d
        xv2 = __cse_1
        __hcse_5 = xv2 * __cse_2
        __cse_3d = ab * (__cse_2 * xv2d + xv2 * __cse_2d) + __hcse_5 * abd
        __cse_3 = __hcse_5 * ab
        xv2bd = xv2bd + __cse_3d
        xv2b = xv2b + __cse_3
        xv2bd = xv2bd + __cse_3d
        xv2b = xv2b + __cse_3
        __hcse_6 = xv2 * xv2d
        __hcse_7 = xv2 * xv2
        wbd[ci] = wbd[ci] + (ab * (__hcse_6 + __hcse_6) + __hcse_7 * abd)
        wb[ci] = wb[ci] + __hcse_7 * ab
        __oldb_0d = xv2bd
        __oldb_0 = xv2b
        xv2bd = 0.0
        xv2b = 0.0
        xbd[ci] = xbd[ci] + __oldb_0d
        xb[ci] = xb[ci] + __oldb_0
    end
    abd = 0.0
    ab = 0.0
    return nothing
end

function liveout_red_top(x, w, loss, i_nc)
    a = 0.0
    for ci = 1:i_nc
        xv2 = x[ci]
        a = a + xv2 * xv2 * w[ci]
    end
    loss[1] = loss[1] + a * a
end
