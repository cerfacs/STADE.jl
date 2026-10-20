function initstacks_asnap_readbetween_b(i_ncell, i_njac, i_nnode)
    m_stack = Vector{Float64}(undef, (max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) * max(0, div(3 - 1, 1) + 1)) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) * max(0, div(3 - 1, 1) + 1))
    x_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    return (m_stack, x_stack)
end

function asnap_readbetween_hv(x, xb, w, wb, i_cell_to_node, m, mb, i_ncell, i_nnode, i_njac, xd, xbd, wd, wbd, md, mbd, m_stack, x_stack)
    m_stack_d = Vector{Float64}(undef, length(m_stack))
    x_stack_d = Vector{Float64}(undef, length(x_stack))
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            __idx_m_stack_0 = ((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1
            m_stack_d[__idx_m_stack_0] = md[i_node]
            m_stack[__idx_m_stack_0] = m[i_node]
            md[i_node] = 0.0
            m[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                __icse_0 = div(3 - 1, 1) + 1
                __idx_m_stack_1 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * __icse_0) + (i_cell - 1) * __icse_0 + (i_loc - 1)) + 1)
                __cse_1d = md[i_n]
                __cse_1 = m[i_n]
                m_stack_d[__idx_m_stack_1] = __cse_1d
                m_stack[__idx_m_stack_1] = __cse_1
                md[i_n] = __cse_1d + 0.1 * wd[i_cell]
                m[i_n] = __cse_1 + w[i_cell] * 0.1
            end
        end
        for i_node = 1:i_nnode
            __idx_x_stack_0 = ((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1
            __cse_2d = xd[i_node]
            __cse_2 = x[i_node]
            x_stack_d[__idx_x_stack_0] = __cse_2d
            x_stack[__idx_x_stack_0] = __cse_2
            __cse_3d = md[i_node]
            __cse_3 = m[i_node]
            __hcse_0 = (__cse_3 * 0.5) * __cse_3d
            xd[i_node] = __cse_2d + (__hcse_0 + __hcse_0)
            x[i_node] = __cse_2 + __cse_3 * __cse_3 * 0.5
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                __icse_4 = max(0, div(i_njac - 1, 1) + 1)
                __icse_5 = div(i_ncell - 1, 1) + 1
                __icse_6 = div(3 - 1, 1) + 1
                __idx_m_stack_1 = (__icse_4 * max(0, div(i_nnode - 1, 1) + 1) + __icse_4 * max(0, __icse_5) * max(0, __icse_6)) + (((i_ - 1) * (__icse_5 * __icse_6) + (i_cell - 1) * __icse_6 + (i_loc - 1)) + 1)
                __cse_7d = md[i_n]
                __cse_7 = m[i_n]
                m_stack_d[__idx_m_stack_1] = __cse_7d
                m_stack[__idx_m_stack_1] = __cse_7
                md[i_n] = __cse_7d + 0.05 * xd[i_n]
                m[i_n] = __cse_7 + x[i_n] * 0.05
            end
        end
    end
    for i_ = i_njac:-1:1
        for i_cell = i_ncell:-1:1
            for i_loc = 3:-1:1
                i_n = i_cell_to_node[i_loc, i_cell]
                __icse_8 = max(0, div(i_njac - 1, 1) + 1)
                __icse_9 = div(i_ncell - 1, 1) + 1
                __icse_10 = div(3 - 1, 1) + 1
                __idx_m_stack_0 = (__icse_8 * max(0, div(i_nnode - 1, 1) + 1) + __icse_8 * max(0, __icse_9) * max(0, __icse_10)) + (((i_ - 1) * (__icse_9 * __icse_10) + (i_cell - 1) * __icse_10 + (i_loc - 1)) + 1)
                md[i_n] = m_stack_d[__idx_m_stack_0]
                m[i_n] = m_stack[__idx_m_stack_0]
                xbd[i_n] = xbd[i_n] + 0.05 * mbd[i_n]
                xb[i_n] = xb[i_n] + 0.05 * mb[i_n]
            end
        end
        for i_node = i_nnode:-1:1
            __idx_x_stack_0 = ((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1
            xd[i_node] = x_stack_d[__idx_x_stack_0]
            x[i_node] = x_stack[__idx_x_stack_0]
            __hcse_1 = xb[i_node]
            __hcse_2 = m[i_node] * 0.5
            __cse_11d = __hcse_1 * (0.5 * md[i_node]) + __hcse_2 * xbd[i_node]
            __cse_11 = __hcse_2 * __hcse_1
            mbd[i_node] = mbd[i_node] + __cse_11d
            mb[i_node] = mb[i_node] + __cse_11
            mbd[i_node] = mbd[i_node] + __cse_11d
            mb[i_node] = mb[i_node] + __cse_11
        end
        for i_cell = i_ncell:-1:1
            for i_loc = 3:-1:1
                i_n = i_cell_to_node[i_loc, i_cell]
                __icse_12 = div(3 - 1, 1) + 1
                __idx_m_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * __icse_12) + (i_cell - 1) * __icse_12 + (i_loc - 1)) + 1)
                md[i_n] = m_stack_d[__idx_m_stack_0]
                m[i_n] = m_stack[__idx_m_stack_0]
                wbd[i_cell] = wbd[i_cell] + 0.1 * mbd[i_n]
                wb[i_cell] = wb[i_cell] + 0.1 * mb[i_n]
            end
        end
        for i_node = i_nnode:-1:1
            __idx_m_stack_0 = ((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1
            md[i_node] = m_stack_d[__idx_m_stack_0]
            m[i_node] = m_stack[__idx_m_stack_0]
            mbd[i_node] = 0.0
            mb[i_node] = 0.0
        end
    end
    return nothing
end

function asnap_readbetween(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            m[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] + w[i_cell] * 0.1
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.5
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] + x[i_n] * 0.05
            end
        end
    end
end
