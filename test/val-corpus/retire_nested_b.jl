function initstacks_retire_nested_b(i_ncell, i_njac, i_nlev, i_nnode)
    nc = i_ncell
    prefix_m_stack_1 = Vector{Int}(undef, max(0, div(i_nlev - 1, 1) + 1))
    __tot_m_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div(i_nlev - 1, 1) + 1))
    __tot_tripcount_stack_1 = 0
    prefix_x_stack_1 = Vector{Int}(undef, max(0, div(i_nlev - 1, 1) + 1))
    __tot_x_stack_1 = 0
    val_nc_1 = Vector{Int64}(undef, max(0, div(i_nlev - 1, 1) + 1))
    for lev = 1:i_nlev
        prefix_m_stack_1[(lev - 1) + 1] = __tot_m_stack_1
        prefix_tripcount_stack_1[(lev - 1) + 1] = __tot_tripcount_stack_1
        prefix_x_stack_1[(lev - 1) + 1] = __tot_x_stack_1
        val_nc_1[(lev - 1) + 1] = nc
        __tot_m_stack_1 = __tot_m_stack_1 + (max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1) * max(0, div(3 - 1, 1) + 1))
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + max(0, div(i_njac - 1, 1) + 1)
        __tot_x_stack_1 = __tot_x_stack_1 + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1)
        nc = nc - 1
    end
    m_stack = Vector{Float64}(undef, __tot_m_stack_1)
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    x_stack = Vector{Float64}(undef, __tot_x_stack_1)
    return (m_stack, tripcount_stack, x_stack, prefix_m_stack_1, prefix_tripcount_stack_1, prefix_x_stack_1, __tot_m_stack_1, __tot_tripcount_stack_1, __tot_x_stack_1, val_nc_1)
end

function retire_nested_b(x, xb, w, wb, i_cell_to_node, m, mb, i_ncell, i_nnode, i_nlev, i_njac, m_stack, tripcount_stack, x_stack, prefix_m_stack_1, prefix_tripcount_stack_1, prefix_x_stack_1, __tot_m_stack_1, __tot_tripcount_stack_1, __tot_x_stack_1, val_nc_1)
    nc = i_ncell
    for lev = 1:i_nlev
        for i_ = 1:i_njac
            for i_node = 1:i_nnode
                __idx_m_stack_1_0 = prefix_m_stack_1[(lev - 1) + 1] + (((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)
                m_stack[__idx_m_stack_1_0] = m[i_node]
                m[i_node] = 0.0
            end
            __idx_tripcount_stack_1_1 = prefix_tripcount_stack_1[(lev - 1) + 1] + ((i_ - 1) + 1)
            tripcount_stack[__idx_tripcount_stack_1_1] = nc
            for i_cell = 1:nc
                for i_loc = 1:3
                    i_n = i_cell_to_node[i_loc, i_cell]
                    __icse_0 = div(3 - 1, 1) + 1
                    __idx_m_stack_1_1 = (prefix_m_stack_1[(lev - 1) + 1] + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1)) + (((i_ - 1) * ((div(val_nc_1[(lev - 1) + 1] - 1, 1) + 1) * __icse_0) + (i_cell - 1) * __icse_0 + (i_loc - 1)) + 1)
                    __cse_1 = m[i_n]
                    m_stack[__idx_m_stack_1_1] = __cse_1
                    m[i_n] = __cse_1 + w[i_cell] * x[i_n] * 0.1
                end
            end
            for i_node = 1:i_nnode
                __idx_x_stack_1_0 = prefix_x_stack_1[(lev - 1) + 1] + (((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)
                __cse_2 = x[i_node]
                x_stack[__idx_x_stack_1_0] = __cse_2
                __cse_3 = m[i_node]
                x[i_node] = __cse_2 + __cse_3 * __cse_3 * 0.5
            end
        end
        nc = nc - 1
    end
    for lev = i_nlev:-1:1
        for i_ = i_njac:-1:1
            for i_node = i_nnode:-1:1
                __idx_x_stack_1_0 = prefix_x_stack_1[(lev - 1) + 1] + (((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)
                x[i_node] = x_stack[__idx_x_stack_1_0]
                __cse_4 = (m[i_node] * 0.5) * xb[i_node]
                mb[i_node] = mb[i_node] + __cse_4
                mb[i_node] = mb[i_node] + __cse_4
            end
            __idx_tripcount_stack_1_1 = prefix_tripcount_stack_1[(lev - 1) + 1] + ((i_ - 1) + 1)
            nc = tripcount_stack[__idx_tripcount_stack_1_1]
            for i_cell = nc:-1:1
                for i_loc = 3:-1:1
                    i_n = i_cell_to_node[i_loc, i_cell]
                    __icse_5 = div(3 - 1, 1) + 1
                    __idx_m_stack_1_0 = (prefix_m_stack_1[(lev - 1) + 1] + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1)) + (((i_ - 1) * ((div(val_nc_1[(lev - 1) + 1] - 1, 1) + 1) * __icse_5) + (i_cell - 1) * __icse_5 + (i_loc - 1)) + 1)
                    m[i_n] = m_stack[__idx_m_stack_1_0]
                    __cse_6 = mb[i_n]
                    wb[i_cell] = wb[i_cell] + (x[i_n] * 0.1) * __cse_6
                    xb[i_n] = xb[i_n] + (w[i_cell] * 0.1) * __cse_6
                end
            end
            for i_node = i_nnode:-1:1
                __idx_m_stack_1_0 = prefix_m_stack_1[(lev - 1) + 1] + (((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)
                m[i_node] = m_stack[__idx_m_stack_1_0]
                mb[i_node] = 0.0
            end
        end
    end
    return nothing
end

function retire_nested(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_nlev, i_njac)
    nc = i_ncell
    for lev = 1:i_nlev
        for i_ = 1:i_njac
            for i_node = 1:i_nnode
                m[i_node] = 0.0
            end
            for i_cell = 1:nc
                for i_loc = 1:3
                    i_n = i_cell_to_node[i_loc, i_cell]
                    m[i_n] = m[i_n] + w[i_cell] * x[i_n] * 0.1
                end
            end
            for i_node = 1:i_nnode
                x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.5
            end
        end
        nc = nc - 1
    end
end
