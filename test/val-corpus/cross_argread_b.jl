function initstacks_cross_argread_b(i_nc)
    s_stack = Vector{Float64}(undef, max(0, div(i_nc - 1, 1) + 1) + 1)
    a_stack = Vector{Float64}(undef, 1)
    return (s_stack, a_stack)
end

function cross_argread_b(x, xb, w, wb, loss, lossb, s, sb, i_nc, s_stack, a_stack)
    a = 0.0
    ab = 0.0
    a = s * s
    for ci = 1:i_nc
        __idx_s_stack_0 = (ci - 1) + 1
        s_stack[__idx_s_stack_0] = s
        s = x[ci] * 2.0
        a = a + s * s * w[ci]
    end
    loss[1] = loss[1] + a * a
    a_stack[1] = a
    __icse_0 = max(0, div(i_nc - 1, 1) + 1) + 1
    __idx_s_stack_4 = __icse_0
    s_stack[__idx_s_stack_4] = s
    a = a_stack[1]
    __idx_s_stack_1 = __icse_0
    s = s_stack[__idx_s_stack_1]
    __cse_1 = a * lossb[1]
    ab = ab + __cse_1
    ab = ab + __cse_1
    for ci = i_nc:-1:1
        __cse_2 = (s * w[ci]) * ab
        sb = sb + __cse_2
        sb = sb + __cse_2
        wb[ci] = wb[ci] + (s * s) * ab
        __idx_s_stack_0 = (ci - 1) + 1
        s = s_stack[__idx_s_stack_0]
        __oldb_2 = sb
        sb = 0.0
        xb[ci] = xb[ci] + 2.0__oldb_2
    end
    __oldb_0 = ab
    ab = 0.0
    __cse_3 = s * __oldb_0
    sb = sb + __cse_3
    sb = sb + __cse_3
    return sb
end

function cross_argread(x, w, loss, s, i_nc)
    a = s * s
    for ci = 1:i_nc
        s = x[ci] * 2.0
        a = a + s * s * w[ci]
    end
    loss[1] = loss[1] + a * a
end
