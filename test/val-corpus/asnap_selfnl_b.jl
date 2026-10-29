function initstacks_asnap_selfnl_b(i_ncell, i_njac, i_nnode)
    m_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) * max(0, div(3 - 1, 1) + 1))
    x_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    return (m_stack, x_stack)
end

function asnap_selfnl_b(x, xb, w, wb, i_cell_to_node, m, mb, i_ncell, i_nnode, i_njac, m_stack, x_stack)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            __idx_m_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            m_stack[__idx_m_stack_0] = m[i_node]
            m[i_node] = 1.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                __icse_0 = div(3 - 1, 1) + 1
                __idx_m_stack_1 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * __icse_0) + (i_cell - 1) * __icse_0 + (i_loc - 1)) + 1)
                __cse_1 = m[i_n]
                m_stack[__idx_m_stack_1] = __cse_1
                m[i_n] = __cse_1 * (1.0 + w[i_cell] * x[i_n] * 0.05)
            end
        end
        for i_node = 1:i_nnode
            __idx_x_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            __cse_2 = x[i_node]
            x_stack[__idx_x_stack_0] = __cse_2
            __cse_3 = m[i_node]
            x[i_node] = __cse_2 + __cse_3 * __cse_3 * 0.1
        end
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __idx_x_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            x[i_node] = x_stack[__idx_x_stack_0]
            __cse_4 = (m[i_node] * 0.1) * xb[i_node]
            mb[i_node] = mb[i_node] + __cse_4
            mb[i_node] = mb[i_node] + __cse_4
        end
        for i_cell = i_ncell:-1:1
            for i_loc = 3:-1:1
                i_n = i_cell_to_node[i_loc, i_cell]
                __icse_5 = div(3 - 1, 1) + 1
                __idx_m_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * __icse_5) + (i_cell - 1) * __icse_5 + (i_loc - 1)) + 1)
                m[i_n] = m_stack[__idx_m_stack_0]
                __oldb_2 = mb[i_n]
                mb[i_n] = 0.0
                __cse_6 = w[i_cell]
                __cse_7 = x[i_n]
                mb[i_n] = mb[i_n] + (1.0 + __cse_6 * __cse_7 * 0.05) * __oldb_2
                __cse_8 = m[i_n] * __oldb_2
                wb[i_cell] = wb[i_cell] + (__cse_7 * 0.05) * __cse_8
                xb[i_n] = xb[i_n] + (__cse_6 * 0.05) * __cse_8
            end
        end
        for i_node = i_nnode:-1:1
            __idx_m_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            m[i_node] = m_stack[__idx_m_stack_0]
            mb[i_node] = 0.0
        end
    end
    return nothing
end

function asnap_selfnl(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            m[i_node] = 1.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] * (1.0 + w[i_cell] * x[i_n] * 0.05)
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.1
        end
    end
end
