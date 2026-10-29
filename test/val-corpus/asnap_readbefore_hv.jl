function initstacks_asnap_readbefore_b(i_ncell, i_njac, i_nnode)
    m_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) * max(0, div(3 - 1, 1) + 1))
    x_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    return (m_stack, x_stack)
end

function asnap_readbefore_hv(x, xb, w, wb, i_cell_to_node, m, mb, y, yb, i_ncell, i_nnode, i_njac, xd, xbd, wd, wbd, md, mbd, yd, ybd, m_stack, x_stack)
    m_stack_d = Vector{Float64}(undef, length(m_stack))
    x_stack_d = Vector{Float64}(undef, length(x_stack))
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            __cse_0d = md[i_node]
            __cse_0 = m[i_node]
            __hcse_0 = (__cse_0 * 0.3) * __cse_0d
            yd[i_node] = yd[i_node] + (__hcse_0 + __hcse_0)
            y[i_node] = y[i_node] + __cse_0 * __cse_0 * 0.3
        end
        for i_node = 1:i_nnode
            __idx_m_stack_0 = ((i_ - 1) + (i_node - 1) * (div(i_njac - 1, 1) + 1)) + 1
            m_stack_d[__idx_m_stack_0] = md[i_node]
            m_stack[__idx_m_stack_0] = m[i_node]
            md[i_node] = 0.0
            m[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                __icse_1 = div(i_njac - 1, 1) + 1
                __idx_m_stack_1 = max(0, __icse_1) * max(0, div(i_nnode - 1, 1) + 1) + (((i_ - 1) + ((i_cell - 1) * (div(3 - 1, 1) + 1) + (i_loc - 1)) * __icse_1) + 1)
                __cse_2d = md[i_n]
                __cse_2 = m[i_n]
                m_stack_d[__idx_m_stack_1] = __cse_2d
                m_stack[__idx_m_stack_1] = __cse_2
                __hcse_1 = x[i_n]
                __hcse_2 = w[i_cell]
                md[i_n] = __cse_2d + ((__hcse_1 * 0.1) * wd[i_cell] + (__hcse_2 * 0.1) * xd[i_n])
                m[i_n] = __cse_2 + __hcse_2 * __hcse_1 * 0.1
            end
        end
        for i_node = 1:i_nnode
            __idx_x_stack_0 = ((i_ - 1) + (i_node - 1) * (div(i_njac - 1, 1) + 1)) + 1
            __cse_3d = xd[i_node]
            __cse_3 = x[i_node]
            x_stack_d[__idx_x_stack_0] = __cse_3d
            x_stack[__idx_x_stack_0] = __cse_3
            xd[i_node] = __cse_3d + 0.5 * md[i_node]
            x[i_node] = __cse_3 + m[i_node] * 0.5
        end
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __idx_x_stack_0 = ((i_ - 1) + (i_node - 1) * (div(i_njac - 1, 1) + 1)) + 1
            xd[i_node] = x_stack_d[__idx_x_stack_0]
            x[i_node] = x_stack[__idx_x_stack_0]
            mbd[i_node] = mbd[i_node] + 0.5 * xbd[i_node]
            mb[i_node] = mb[i_node] + 0.5 * xb[i_node]
        end
        for i_cell = i_ncell:-1:1
            for i_loc = 3:-1:1
                i_n = i_cell_to_node[i_loc, i_cell]
                __icse_4 = div(i_njac - 1, 1) + 1
                __idx_m_stack_0 = max(0, __icse_4) * max(0, div(i_nnode - 1, 1) + 1) + (((i_ - 1) + ((i_cell - 1) * (div(3 - 1, 1) + 1) + (i_loc - 1)) * __icse_4) + 1)
                md[i_n] = m_stack_d[__idx_m_stack_0]
                m[i_n] = m_stack[__idx_m_stack_0]
                __cse_5d = mbd[i_n]
                __cse_5 = mb[i_n]
                __hcse_3 = x[i_n] * 0.1
                wbd[i_cell] = wbd[i_cell] + (__cse_5 * (0.1 * xd[i_n]) + __hcse_3 * __cse_5d)
                wb[i_cell] = wb[i_cell] + __hcse_3 * __cse_5
                __hcse_4 = w[i_cell] * 0.1
                xbd[i_n] = xbd[i_n] + (__cse_5 * (0.1 * wd[i_cell]) + __hcse_4 * __cse_5d)
                xb[i_n] = xb[i_n] + __hcse_4 * __cse_5
            end
        end
        for i_node = i_nnode:-1:1
            __idx_m_stack_0 = ((i_ - 1) + (i_node - 1) * (div(i_njac - 1, 1) + 1)) + 1
            md[i_node] = m_stack_d[__idx_m_stack_0]
            m[i_node] = m_stack[__idx_m_stack_0]
            mbd[i_node] = 0.0
            mb[i_node] = 0.0
        end
        for i_node = i_nnode:-1:1
            __hcse_5 = yb[i_node]
            __hcse_6 = m[i_node] * 0.3
            __cse_6d = __hcse_5 * (0.3 * md[i_node]) + __hcse_6 * ybd[i_node]
            __cse_6 = __hcse_6 * __hcse_5
            mbd[i_node] = mbd[i_node] + __cse_6d
            mb[i_node] = mb[i_node] + __cse_6
            mbd[i_node] = mbd[i_node] + __cse_6d
            mb[i_node] = mb[i_node] + __cse_6
        end
    end
    return nothing
end

function asnap_readbefore(x, w, i_cell_to_node, m, y, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            y[i_node] = y[i_node] + m[i_node] * m[i_node] * 0.3
        end
        for i_node = 1:i_nnode
            m[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] + w[i_cell] * x[i_n] * 0.1
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * 0.5
        end
    end
end
