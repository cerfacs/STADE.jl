function initstacks_asnap_direct_b(i_njac)
    m_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1))
    x_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1))
    return (m_stack, x_stack)
end

function asnap_direct_b(x, xb, w, wb, node, m, mb, i_njac, m_stack, x_stack)
    for i_ = 1:i_njac
        n0 = node[1]
        __icse_0 = (i_ - 1) + 1
        __idx_m_stack_1 = __icse_0
        __cse_1 = m[n0]
        m_stack[__idx_m_stack_1] = __cse_1
        __cse_2 = x[1]
        m[n0] = __cse_1 + w[1] * __cse_2 * 0.1
        __idx_x_stack_4 = __icse_0
        x_stack[__idx_x_stack_4] = __cse_2
        __cse_3 = m[n0]
        x[1] = __cse_2 + __cse_3 * __cse_3 * 0.5
    end
    for i_ = i_njac:-1:1
        n0 = node[1]
        __icse_4 = (i_ - 1) + 1
        __idx_x_stack_0 = __icse_4
        x[1] = x_stack[__idx_x_stack_0]
        __cse_5 = xb[1]
        __cse_6 = (m[n0] * 0.5) * __cse_5
        mb[n0] = mb[n0] + __cse_6
        mb[n0] = mb[n0] + __cse_6
        __idx_m_stack_0 = __icse_4
        m[n0] = m_stack[__idx_m_stack_0]
        __cse_7 = mb[n0]
        wb[1] = wb[1] + (x[1] * 0.1) * __cse_7
        xb[1] = __cse_5 + (w[1] * 0.1) * __cse_7
    end
    return nothing
end

function asnap_direct(x, w, node, m, i_njac)
    for i_ = 1:i_njac
        n0 = node[1]
        m[n0] = m[n0] + w[1] * x[1] * 0.1
        x[1] = x[1] + m[n0] * m[n0] * 0.5
    end
end
