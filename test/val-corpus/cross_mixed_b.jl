function initstacks_cross_mixed_b(i_nc)
    a_stack = Vector{Float64}(undef, 1)
    b_stack = Vector{Float64}(undef, 1)
    return (a_stack, b_stack)
end

function cross_mixed_b(x, xb, w, wb, loss, lossb, i_nc, a_stack, b_stack)
    a = 0.0
    b = 0.0
    tmp = 0.0
    xv = 0.0
    ab = 0.0
    bb = 0.0
    tmpb = 0.0
    xvb = 0.0
    a = 0.0
    for ci = 1:i_nc
        __cse_0 = w[ci]
        tmp = x[ci] * __cse_0
        xv = tmp * 0.5
        a = a + xv * xv * __cse_0
    end
    b = 0.0
    for ci = 1:i_nc
        xv = x[ci]
        b = b + xv * xv * w[ci]
    end
    loss[1] = loss[1] + a * a + b * b
    a_stack[1] = a
    b_stack[1] = b
    a = a_stack[1]
    b = b_stack[1]
    __cse_1 = lossb[1]
    __cse_2 = a * __cse_1
    ab = ab + __cse_2
    ab = ab + __cse_2
    __cse_3 = b * __cse_1
    bb = bb + __cse_3
    bb = bb + __cse_3
    for ci = 1:i_nc
        xv = x[ci]
        __cse_4 = w[ci]
        b = b + xv * xv * __cse_4
        __cse_5 = (xv * __cse_4) * bb
        xvb = xvb + __cse_5
        xvb = xvb + __cse_5
        wb[ci] = wb[ci] + (xv * xv) * bb
        __oldb_0 = xvb
        xvb = 0.0
        xb[ci] = xb[ci] + __oldb_0
    end
    bb = 0.0
    for ci = 1:i_nc
        __cse_6 = x[ci]
        __cse_7 = w[ci]
        __cse_8 = __cse_6 * __cse_7
        tmp = __cse_8
        xv = tmp * 0.5
        a = a + xv * xv * __cse_7
        tmp = __cse_8
        __cse_9 = (xv * __cse_7) * ab
        xvb = xvb + __cse_9
        xvb = xvb + __cse_9
        wb[ci] = wb[ci] + (xv * xv) * ab
        __oldb_0 = xvb
        xvb = 0.0
        tmpb = tmpb + 0.5__oldb_0
        __oldb_0 = tmpb
        tmpb = 0.0
        xb[ci] = xb[ci] + __cse_7 * __oldb_0
        wb[ci] = wb[ci] + __cse_6 * __oldb_0
    end
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
