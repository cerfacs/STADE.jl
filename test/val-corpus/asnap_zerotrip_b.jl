function initstacks_asnap_zerotrip_b(i_njac, i_nnode, m)
    __len_m = length(m)
    x_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    m_snap_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * __len_m)
    return (x_stack, m_snap_stack, __len_m)
end

function asnap_zerotrip_b(x, xb, w, wb, i_cell_to_node, m, mb, i_ncell, i_nnode, i_njac, x_stack, m_snap_stack, __len_m)
    i_nc2 = i_ncell - 8
    for i_ = 1:i_njac
        __idx_m_snap_stack_0 = (i_ - 1) + 1
        __sbase_m = (__idx_m_snap_stack_0 - 1) * __len_m
        for __sk_m = 1:__len_m
            m_snap_stack[__sbase_m + __sk_m] = m[__sk_m]
        end
        for i_cell = 1:i_nc2
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] + w[i_cell] * x[i_n] * 0.1
            end
        end
        for i_node = 1:i_nnode
            __idx_x_stack_0 = ((i_ - 1) + (i_node - 1) * (div(i_njac - 1, 1) + 1)) + 1
            __cse_0 = x[i_node]
            x_stack[__idx_x_stack_0] = __cse_0
            __cse_1 = m[i_node]
            x[i_node] = __cse_0 + __cse_1 * __cse_1 * 0.5
        end
    end
    i_nc2 = i_ncell - 8
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __idx_x_stack_0 = ((i_ - 1) + (i_node - 1) * (div(i_njac - 1, 1) + 1)) + 1
            x[i_node] = x_stack[__idx_x_stack_0]
            __cse_2 = (m[i_node] * 0.5) * xb[i_node]
            mb[i_node] = mb[i_node] + __cse_2
            mb[i_node] = mb[i_node] + __cse_2
        end
        for i_cell = i_nc2:-1:1
            for i_loc = 3:-1:1
                i_n = i_cell_to_node[i_loc, i_cell]
                __cse_3 = mb[i_n]
                wb[i_cell] = wb[i_cell] + (x[i_n] * 0.1) * __cse_3
                xb[i_n] = xb[i_n] + (w[i_cell] * 0.1) * __cse_3
            end
        end
        __idx_m_snap_stack_0 = (i_ - 1) + 1
        __sbase_m = (__idx_m_snap_stack_0 - 1) * __len_m
        for __sk_m = 1:__len_m
            m[__sk_m] = m_snap_stack[__sbase_m + __sk_m]
        end
    end
    return nothing
end

function asnap_zerotrip(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
    i_nc2 = i_ncell - 8
    for i_ = 1:i_njac
        for i_cell = 1:i_nc2
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] + w[i_cell] * x[i_n] * 0.1
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.5
        end
    end
end
