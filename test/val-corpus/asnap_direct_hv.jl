function initstacks_asnap_direct_b(i_njac)
    m_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1))
    x_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1))
    return (m_stack, x_stack)
end

function asnap_direct_hv(x, xb, w, wb, node, m, mb, i_njac, xd, xbd, wd, wbd, md, mbd, m_stack, x_stack)
    m_stack_d = Vector{Float64}(undef, length(m_stack))
    x_stack_d = Vector{Float64}(undef, length(x_stack))
    for i_ = 1:i_njac
        n0 = node[1]
        __icse_0 = (i_ - 1) + 1
        __idx_m_stack_1 = __icse_0
        __cse_1d = md[n0]
        __cse_1 = m[n0]
        m_stack_d[__idx_m_stack_1] = __cse_1d
        m_stack[__idx_m_stack_1] = __cse_1
        __cse_2d = xd[1]
        __cse_2 = x[1]
        __hcse_0 = w[1]
        md[n0] = __cse_1d + ((__cse_2 * 0.1) * wd[1] + (__hcse_0 * 0.1) * __cse_2d)
        m[n0] = __cse_1 + __hcse_0 * __cse_2 * 0.1
        __idx_x_stack_4 = __icse_0
        x_stack_d[__idx_x_stack_4] = __cse_2d
        x_stack[__idx_x_stack_4] = __cse_2
        __cse_3d = md[n0]
        __cse_3 = m[n0]
        __hcse_1 = (__cse_3 * 0.5) * __cse_3d
        xd[1] = __cse_2d + (__hcse_1 + __hcse_1)
        x[1] = __cse_2 + __cse_3 * __cse_3 * 0.5
    end
    for i_ = i_njac:-1:1
        n0 = node[1]
        __icse_4 = (i_ - 1) + 1
        __idx_x_stack_0 = __icse_4
        xd[1] = x_stack_d[__idx_x_stack_0]
        x[1] = x_stack[__idx_x_stack_0]
        __cse_5d = xbd[1]
        __cse_5 = xb[1]
        __hcse_2 = m[n0] * 0.5
        __cse_6d = __cse_5 * (0.5 * md[n0]) + __hcse_2 * __cse_5d
        __cse_6 = __hcse_2 * __cse_5
        mbd[n0] = mbd[n0] + __cse_6d
        mb[n0] = mb[n0] + __cse_6
        mbd[n0] = mbd[n0] + __cse_6d
        mb[n0] = mb[n0] + __cse_6
        __idx_m_stack_0 = __icse_4
        md[n0] = m_stack_d[__idx_m_stack_0]
        m[n0] = m_stack[__idx_m_stack_0]
        __cse_7d = mbd[n0]
        __cse_7 = mb[n0]
        __hcse_3 = x[1] * 0.1
        wbd[1] = wbd[1] + (__cse_7 * (0.1 * xd[1]) + __hcse_3 * __cse_7d)
        wb[1] = wb[1] + __hcse_3 * __cse_7
        __hcse_4 = w[1] * 0.1
        xbd[1] = __cse_5d + (__cse_7 * (0.1 * wd[1]) + __hcse_4 * __cse_7d)
        xb[1] = __cse_5 + __hcse_4 * __cse_7
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
