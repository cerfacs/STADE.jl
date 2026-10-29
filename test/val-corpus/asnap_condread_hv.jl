function initstacks_asnap_condread_b(i_ncell, i_njac, i_nnode)
    m_stack = Vector{Float64}(undef, (max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) * max(0, div(3 - 1, 1) + 1)) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) * max(0, div(3 - 1, 1) + 1))
    branch_stack = Vector{Int64}(undef, max(0, div(i_njac - 1, 1) + 1))
    x_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    return (m_stack, branch_stack, x_stack)
end

function asnap_condread_hv(x, xb, w, wb, i_cell_to_node, m, mb, i_ncell, i_nnode, i_njac, xd, xbd, wd, wbd, md, mbd, m_stack, branch_stack, x_stack)
    m_stack_d = Vector{Float64}(undef, length(m_stack))
    x_stack_d = Vector{Float64}(undef, length(x_stack))
    for i_ = 1:i_njac
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
                __icse_0 = div(i_njac - 1, 1) + 1
                __idx_m_stack_1 = max(0, __icse_0) * max(0, div(i_nnode - 1, 1) + 1) + (((i_ - 1) + ((i_cell - 1) * (div(3 - 1, 1) + 1) + (i_loc - 1)) * __icse_0) + 1)
                __cse_1d = md[i_n]
                __cse_1 = m[i_n]
                m_stack_d[__idx_m_stack_1] = __cse_1d
                m_stack[__idx_m_stack_1] = __cse_1
                __hcse_0 = x[i_n]
                __hcse_1 = w[i_cell]
                md[i_n] = __cse_1d + ((__hcse_0 * 0.1) * wd[i_cell] + (__hcse_1 * 0.1) * xd[i_n])
                m[i_n] = __cse_1 + __hcse_1 * __hcse_0 * 0.1
            end
        end
        if m[1] > 0.3
            __icse_2 = (i_ - 1) + 1
            __idx_branch_stack_0 = __icse_2
            branch_stack[__idx_branch_stack_0] = 1
            __idx_x_stack_0 = __icse_2
            __cse_3d = xd[1]
            __cse_3 = x[1]
            x_stack_d[__idx_x_stack_0] = __cse_3d
            x_stack[__idx_x_stack_0] = __cse_3
            xd[1] = 1.1__cse_3d
            x[1] = __cse_3 * 1.1
        else
            __idx_branch_stack_0 = (i_ - 1) + 1
            branch_stack[__idx_branch_stack_0] = 0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                __icse_4 = div(i_njac - 1, 1) + 1
                __icse_5 = max(0, __icse_4)
                __icse_6 = div(3 - 1, 1) + 1
                __idx_m_stack_1 = (__icse_5 * max(0, div(i_nnode - 1, 1) + 1) + __icse_5 * max(0, div(i_ncell - 1, 1) + 1) * max(0, __icse_6)) + (((i_ - 1) + ((i_cell - 1) * __icse_6 + (i_loc - 1)) * __icse_4) + 1)
                __cse_7d = md[i_n]
                __cse_7 = m[i_n]
                m_stack_d[__idx_m_stack_1] = __cse_7d
                m_stack[__idx_m_stack_1] = __cse_7
                md[i_n] = __cse_7d + 0.05 * wd[i_cell]
                m[i_n] = __cse_7 + w[i_cell] * 0.05
            end
        end
        for i_node = 1:i_nnode
            __icse_8 = div(i_njac - 1, 1) + 1
            __idx_x_stack_0 = max(0, __icse_8) + (((i_ - 1) + (i_node - 1) * __icse_8) + 1)
            __cse_9d = xd[i_node]
            __cse_9 = x[i_node]
            x_stack_d[__idx_x_stack_0] = __cse_9d
            x_stack[__idx_x_stack_0] = __cse_9
            __cse_10d = md[i_node]
            __cse_10 = m[i_node]
            __hcse_2 = (__cse_10 * 0.5) * __cse_10d
            xd[i_node] = __cse_9d + (__hcse_2 + __hcse_2)
            x[i_node] = __cse_9 + __cse_10 * __cse_10 * 0.5
        end
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __icse_11 = div(i_njac - 1, 1) + 1
            __idx_x_stack_0 = max(0, __icse_11) + (((i_ - 1) + (i_node - 1) * __icse_11) + 1)
            xd[i_node] = x_stack_d[__idx_x_stack_0]
            x[i_node] = x_stack[__idx_x_stack_0]
            __hcse_3 = xb[i_node]
            __hcse_4 = m[i_node] * 0.5
            __cse_12d = __hcse_3 * (0.5 * md[i_node]) + __hcse_4 * xbd[i_node]
            __cse_12 = __hcse_4 * __hcse_3
            mbd[i_node] = mbd[i_node] + __cse_12d
            mb[i_node] = mb[i_node] + __cse_12
            mbd[i_node] = mbd[i_node] + __cse_12d
            mb[i_node] = mb[i_node] + __cse_12
        end
        for i_cell = i_ncell:-1:1
            for i_loc = 3:-1:1
                i_n = i_cell_to_node[i_loc, i_cell]
                __icse_13 = div(i_njac - 1, 1) + 1
                __icse_14 = max(0, __icse_13)
                __icse_15 = div(3 - 1, 1) + 1
                __idx_m_stack_0 = (__icse_14 * max(0, div(i_nnode - 1, 1) + 1) + __icse_14 * max(0, div(i_ncell - 1, 1) + 1) * max(0, __icse_15)) + (((i_ - 1) + ((i_cell - 1) * __icse_15 + (i_loc - 1)) * __icse_13) + 1)
                md[i_n] = m_stack_d[__idx_m_stack_0]
                m[i_n] = m_stack[__idx_m_stack_0]
                wbd[i_cell] = wbd[i_cell] + 0.05 * mbd[i_n]
                wb[i_cell] = wb[i_cell] + 0.05 * mb[i_n]
            end
        end
        __idx_branch_stack_2 = (i_ - 1) + 1
        __branch = branch_stack[__idx_branch_stack_2]
        if __branch == 1
            __idx_x_stack_0 = (i_ - 1) + 1
            xd[1] = x_stack_d[__idx_x_stack_0]
            x[1] = x_stack[__idx_x_stack_0]
            __oldb_2d = xbd[1]
            __oldb_2 = xb[1]
            xbd[1] = 0.0
            xb[1] = 0.0
            xbd[1] = xbd[1] + 1.1__oldb_2d
            xb[1] = xb[1] + 1.1__oldb_2
        end
        for i_cell = i_ncell:-1:1
            for i_loc = 3:-1:1
                i_n = i_cell_to_node[i_loc, i_cell]
                __icse_16 = div(i_njac - 1, 1) + 1
                __idx_m_stack_0 = max(0, __icse_16) * max(0, div(i_nnode - 1, 1) + 1) + (((i_ - 1) + ((i_cell - 1) * (div(3 - 1, 1) + 1) + (i_loc - 1)) * __icse_16) + 1)
                md[i_n] = m_stack_d[__idx_m_stack_0]
                m[i_n] = m_stack[__idx_m_stack_0]
                __cse_17d = mbd[i_n]
                __cse_17 = mb[i_n]
                __hcse_5 = x[i_n] * 0.1
                wbd[i_cell] = wbd[i_cell] + (__cse_17 * (0.1 * xd[i_n]) + __hcse_5 * __cse_17d)
                wb[i_cell] = wb[i_cell] + __hcse_5 * __cse_17
                __hcse_6 = w[i_cell] * 0.1
                xbd[i_n] = xbd[i_n] + (__cse_17 * (0.1 * wd[i_cell]) + __hcse_6 * __cse_17d)
                xb[i_n] = xb[i_n] + __hcse_6 * __cse_17
            end
        end
        for i_node = i_nnode:-1:1
            __idx_m_stack_0 = ((i_ - 1) + (i_node - 1) * (div(i_njac - 1, 1) + 1)) + 1
            md[i_node] = m_stack_d[__idx_m_stack_0]
            m[i_node] = m_stack[__idx_m_stack_0]
            mbd[i_node] = 0.0
            mb[i_node] = 0.0
        end
    end
    return nothing
end

function asnap_condread(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            m[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] + w[i_cell] * x[i_n] * 0.1
            end
        end
        if m[1] > 0.3
            x[1] = x[1] * 1.1
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] + w[i_cell] * 0.05
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.5
        end
    end
end
