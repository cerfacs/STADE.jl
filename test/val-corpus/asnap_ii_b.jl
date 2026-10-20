function initstacks_asnap_ii_b(i_ncell, i_njac, m)
    __len_m = length(m)
    auxu_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1))
    m_snap_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * __len_m)
    return (auxu_stack, m_snap_stack, __len_m)
end

function asnap_ii_b(x, xb, w, wb, i_cell_to_node, m, mb, i_ncell, i_nnode, i_njac, auxu_stack, m_snap_stack, __len_m)
    auxu = 0.0
    auxub = 0.0
    for i_ = 1:i_njac
        __idx_m_snap_stack_0 = (i_ - 1) + 1
        __sbase_m = (__idx_m_snap_stack_0 - 1) * __len_m
        for __sk_m = 1:__len_m
            m_snap_stack[__sbase_m + __sk_m] = m[__sk_m]
        end
        for i_node = 1:i_nnode
            m[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            auxu = 0.0
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                auxu = auxu + w[i_n]
            end
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] + auxu * auxu * 0.1
            end
            __idx_auxu_stack_3 = ((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1
            auxu_stack[__idx_auxu_stack_3] = auxu
        end
        for i_node = 1:i_nnode
            __cse_0 = m[i_node]
            x[i_node] = x[i_node] + __cse_0 * __cse_0 * 0.5
        end
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __cse_1 = (m[i_node] * 0.5) * xb[i_node]
            mb[i_node] = mb[i_node] + __cse_1
            mb[i_node] = mb[i_node] + __cse_1
        end
        for i_cell = i_ncell:-1:1
            __idx_auxu_stack_0 = ((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1
            auxu = auxu_stack[__idx_auxu_stack_0]
            for i_loc = 3:-1:1
                i_n = i_cell_to_node[i_loc, i_cell]
                __cse_2 = (auxu * 0.1) * mb[i_n]
                auxub = auxub + __cse_2
                auxub = auxub + __cse_2
            end
            for i_loc = 1:3
                __icse_3 = i_cell_to_node[i_loc, i_cell]
                i_n = __icse_3
                auxu = auxu + w[i_n]
                i_n = __icse_3
                wb[i_n] = wb[i_n] + auxub
            end
            auxub = 0.0
        end
        for i_node = i_nnode:-1:1
            mb[i_node] = 0.0
        end
        __idx_m_snap_stack_0 = (i_ - 1) + 1
        __sbase_m = (__idx_m_snap_stack_0 - 1) * __len_m
        for __sk_m = 1:__len_m
            m[__sk_m] = m_snap_stack[__sbase_m + __sk_m]
        end
    end
    return nothing
end

function asnap_ii(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            m[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            auxu = 0.0
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                auxu = auxu + w[i_n]
            end
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] + auxu * auxu * 0.1
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.5
        end
    end
end
