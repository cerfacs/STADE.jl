function initstacks_liveout_red_top_b()
    a_stack = Vector{Float64}(undef, 1)
    return a_stack
end

function liveout_red_top_b(x, xb, w, wb, loss, lossb, i_nc, a_stack)
    a = 0.0
    xv2 = 0.0
    ab = 0.0
    xv2b = 0.0
    a = 0.0
    for ci = 1:i_nc
        xv2 = x[ci]
        a = a + xv2 * xv2 * w[ci]
    end
    loss[1] = loss[1] + a * a
    a_stack[1] = a
    a = a_stack[1]
    __cse_0 = a * lossb[1]
    ab = ab + __cse_0
    ab = ab + __cse_0
    for ci = 1:i_nc
        __cse_1 = x[ci]
        xv2 = __cse_1
        __cse_2 = w[ci]
        a = a + xv2 * xv2 * __cse_2
        xv2 = __cse_1
        __cse_3 = (xv2 * __cse_2) * ab
        xv2b = xv2b + __cse_3
        xv2b = xv2b + __cse_3
        wb[ci] = wb[ci] + (xv2 * xv2) * ab
        __oldb_0 = xv2b
        xv2b = 0.0
        xb[ci] = xb[ci] + __oldb_0
    end
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
