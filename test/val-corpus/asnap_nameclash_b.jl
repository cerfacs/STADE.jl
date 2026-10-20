function initstacks_asnap_nameclash_b(i_ncell, i_njac, i_nnode, m_snap)
    __len_m_snap = length(m_snap)
    m_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) * max(0, div(3 - 1, 1) + 1))
    x_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    m_snap_snap_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * __len_m_snap)
    return (m_stack, x_stack, m_snap_snap_stack, __len_m_snap)
end

function asnap_nameclash_b(x, xb, w, wb, i_cell_to_node, m, mb, m_snap, m_snapb, i_ncell, i_nnode, i_njac, m_stack, x_stack, m_snap_snap_stack, __len_m_snap)
    for i_ = 1:i_njac
        __idx_m_snap_snap_stack_0 = (i_ - 1) + 1
        __sbase_m_snap = (__idx_m_snap_snap_stack_0 - 1) * __len_m_snap
        for __sk_m_snap = 1:__len_m_snap
            m_snap_snap_stack[__sbase_m_snap + __sk_m_snap] = m_snap[__sk_m_snap]
        end
        for i_node = 1:i_nnode
            __idx_m_stack_0 = ((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1
            m_stack[__idx_m_stack_0] = m[i_node]
            m[i_node] = 0.0
            m_snap[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                __icse_0 = div(3 - 1, 1) + 1
                __idx_m_stack_1 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * __icse_0) + (i_cell - 1) * __icse_0 + (i_loc - 1)) + 1)
                __cse_1 = m[i_n]
                m_stack[__idx_m_stack_1] = __cse_1
                __cse_2 = w[i_cell]
                m[i_n] = __cse_1 + __cse_2 * x[i_n] * 0.1
                m_snap[i_n] = m_snap[i_n] + __cse_2 * 0.05
            end
        end
        for i_node = 1:i_nnode
            __idx_x_stack_0 = ((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1
            __cse_3 = x[i_node]
            x_stack[__idx_x_stack_0] = __cse_3
            __cse_4 = m[i_node]
            __cse_5 = m_snap[i_node]
            x[i_node] = __cse_3 + __cse_4 * __cse_4 * 0.5 + __cse_5 * __cse_5 * 0.25
        end
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __idx_x_stack_0 = ((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1
            x[i_node] = x_stack[__idx_x_stack_0]
            __cse_6 = xb[i_node]
            __cse_7 = (m[i_node] * 0.5) * __cse_6
            mb[i_node] = mb[i_node] + __cse_7
            mb[i_node] = mb[i_node] + __cse_7
            __cse_8 = (m_snap[i_node] * 0.25) * __cse_6
            m_snapb[i_node] = m_snapb[i_node] + __cse_8
            m_snapb[i_node] = m_snapb[i_node] + __cse_8
        end
        for i_cell = i_ncell:-1:1
            for i_loc = 3:-1:1
                i_n = i_cell_to_node[i_loc, i_cell]
                wb[i_cell] = wb[i_cell] + 0.05 * m_snapb[i_n]
                __icse_9 = div(3 - 1, 1) + 1
                __idx_m_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * __icse_9) + (i_cell - 1) * __icse_9 + (i_loc - 1)) + 1)
                m[i_n] = m_stack[__idx_m_stack_0]
                __cse_10 = mb[i_n]
                wb[i_cell] = wb[i_cell] + (x[i_n] * 0.1) * __cse_10
                xb[i_n] = xb[i_n] + (w[i_cell] * 0.1) * __cse_10
            end
        end
        for i_node = i_nnode:-1:1
            m_snapb[i_node] = 0.0
            __idx_m_stack_0 = ((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1
            m[i_node] = m_stack[__idx_m_stack_0]
            mb[i_node] = 0.0
        end
        __idx_m_snap_snap_stack_0 = (i_ - 1) + 1
        __sbase_m_snap = (__idx_m_snap_snap_stack_0 - 1) * __len_m_snap
        for __sk_m_snap = 1:__len_m_snap
            m_snap[__sk_m_snap] = m_snap_snap_stack[__sbase_m_snap + __sk_m_snap]
        end
    end
    return nothing
end

function asnap_nameclash(x, w, i_cell_to_node, m, m_snap, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            m[i_node] = 0.0
            m_snap[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] + w[i_cell] * x[i_n] * 0.1
                m_snap[i_n] = m_snap[i_n] + w[i_cell] * 0.05
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.5 + m_snap[i_node] * m_snap[i_node] * 0.25
        end
    end
end
