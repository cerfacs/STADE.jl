function initstacks_idm_gather_rmw_b(i_ncell)
    m_stack = Vector{Float64}(undef, max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1))
    return m_stack
end

function idm_gather_rmw_b(m, mb, x, xb, i_cell_to_node, loss, lossb, i_ncell, i_nnode, m_stack)
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            __idx_m_stack_1 = ((i_cell - 1) + (i_loc - 1) * (div(i_ncell - 1, 1) + 1)) + 1
            __cse_0 = m[i_node]
            m_stack[__idx_m_stack_1] = __cse_0
            m[i_node] = __cse_0 * 0.5 + x[i_cell]
        end
    end
    for i_n = 1:i_nnode
        __cse_1 = m[i_n]
        loss[1] = loss[1] + __cse_1 * __cse_1
    end
    for i_n = i_nnode:-1:1
        __cse_2 = m[i_n] * lossb[1]
        mb[i_n] = mb[i_n] + __cse_2
        mb[i_n] = mb[i_n] + __cse_2
    end
    for i_cell = i_ncell:-1:1
        for i_loc = 4:-1:1
            i_node = i_cell_to_node[i_loc, i_cell]
            __idx_m_stack_0 = ((i_cell - 1) + (i_loc - 1) * (div(i_ncell - 1, 1) + 1)) + 1
            m[i_node] = m_stack[__idx_m_stack_0]
            __oldb_2 = mb[i_node]
            mb[i_node] = 0.0
            mb[i_node] = mb[i_node] + 0.5__oldb_2
            xb[i_cell] = xb[i_cell] + __oldb_2
        end
    end
    return nothing
end

function idm_gather_rmw(m, x, i_cell_to_node, loss, i_ncell, i_nnode)
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            m[i_node] = m[i_node] * 0.5 + x[i_cell]
        end
    end
    for i_n = 1:i_nnode
        loss[1] = loss[1] + m[i_n] * m[i_n]
    end
end
