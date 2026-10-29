function initstacks_asnap_selfnl_b(i_ncell, i_njac, i_nnode)
    m_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) * max(0, div(3 - 1, 1) + 1))
    x_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    return (m_stack, x_stack)
end

function asnap_selfnl_hv(x, xb, w, wb, i_cell_to_node, m, mb, i_ncell, i_nnode, i_njac, xd, xbd, wd, wbd, md, mbd, m_stack, x_stack)
    m_stack_d = Vector{Float64}(undef, length(m_stack))
    x_stack_d = Vector{Float64}(undef, length(x_stack))
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            __idx_m_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            m_stack_d[__idx_m_stack_0] = md[i_node]
            m_stack[__idx_m_stack_0] = m[i_node]
            md[i_node] = 0.0
            m[i_node] = 1.0
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
                __hcse_0 = w[i_cell]
                __hcse_1 = x[i_n]
                __hcse_2 = 1.0 + __hcse_0 * __hcse_1 * 0.05
                md[i_n] = __hcse_2 * __cse_1d + __cse_1 * ((__hcse_1 * 0.05) * wd[i_cell] + (__hcse_0 * 0.05) * xd[i_n])
                m[i_n] = __cse_1 * __hcse_2
            end
        end
        for i_node = 1:i_nnode
            __idx_x_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            __cse_2d = xd[i_node]
            __cse_2 = x[i_node]
            x_stack_d[__idx_x_stack_0] = __cse_2d
            x_stack[__idx_x_stack_0] = __cse_2
            __cse_3d = md[i_node]
            __cse_3 = m[i_node]
            __hcse_3 = (__cse_3 * 0.1) * __cse_3d
            xd[i_node] = __cse_2d + (__hcse_3 + __hcse_3)
            x[i_node] = __cse_2 + __cse_3 * __cse_3 * 0.1
        end
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __idx_x_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            xd[i_node] = x_stack_d[__idx_x_stack_0]
            x[i_node] = x_stack[__idx_x_stack_0]
            __hcse_4 = xb[i_node]
            __hcse_5 = m[i_node] * 0.1
            __cse_4d = __hcse_4 * (0.1 * md[i_node]) + __hcse_5 * xbd[i_node]
            __cse_4 = __hcse_5 * __hcse_4
            mbd[i_node] = mbd[i_node] + __cse_4d
            mb[i_node] = mb[i_node] + __cse_4
            mbd[i_node] = mbd[i_node] + __cse_4d
            mb[i_node] = mb[i_node] + __cse_4
        end
        for i_cell = i_ncell:-1:1
            for i_loc = 3:-1:1
                i_n = i_cell_to_node[i_loc, i_cell]
                __icse_5 = div(3 - 1, 1) + 1
                __idx_m_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * __icse_5) + (i_cell - 1) * __icse_5 + (i_loc - 1)) + 1)
                md[i_n] = m_stack_d[__idx_m_stack_0]
                m[i_n] = m_stack[__idx_m_stack_0]
                __oldb_2d = mbd[i_n]
                __oldb_2 = mb[i_n]
                mbd[i_n] = 0.0
                mb[i_n] = 0.0
                __cse_6d = wd[i_cell]
                __cse_6 = w[i_cell]
                __cse_7d = xd[i_n]
                __cse_7 = x[i_n]
                __hcse_6 = __cse_7 * 0.05
                __hcse_7 = __cse_6 * 0.05
                __hcse_8 = 1.0 + __cse_6 * __cse_7 * 0.05
                mbd[i_n] = mbd[i_n] + (__oldb_2 * (__hcse_6 * __cse_6d + __hcse_7 * __cse_7d) + __hcse_8 * __oldb_2d)
                mb[i_n] = mb[i_n] + __hcse_8 * __oldb_2
                __hcse_9 = m[i_n]
                __cse_8d = __oldb_2 * md[i_n] + __hcse_9 * __oldb_2d
                __cse_8 = __hcse_9 * __oldb_2
                wbd[i_cell] = wbd[i_cell] + (__cse_8 * (0.05__cse_7d) + __hcse_6 * __cse_8d)
                wb[i_cell] = wb[i_cell] + __hcse_6 * __cse_8
                xbd[i_n] = xbd[i_n] + (__cse_8 * (0.05__cse_6d) + __hcse_7 * __cse_8d)
                xb[i_n] = xb[i_n] + __hcse_7 * __cse_8
            end
        end
        for i_node = i_nnode:-1:1
            __idx_m_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            md[i_node] = m_stack_d[__idx_m_stack_0]
            m[i_node] = m_stack[__idx_m_stack_0]
            mbd[i_node] = 0.0
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
