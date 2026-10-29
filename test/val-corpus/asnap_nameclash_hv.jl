function initstacks_asnap_nameclash_b(i_ncell, i_njac, i_nnode, m_snap)
    __len_m_snap = length(m_snap)
    m_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) * max(0, div(3 - 1, 1) + 1))
    x_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    m_snap_snap_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * __len_m_snap)
    return (m_stack, x_stack, m_snap_snap_stack, __len_m_snap)
end

function asnap_nameclash_hv(x, xb, w, wb, i_cell_to_node, m, mb, m_snap, m_snapb, i_ncell, i_nnode, i_njac, xd, xbd, wd, wbd, md, mbd, m_snapd, m_snapbd, m_stack, x_stack, m_snap_snap_stack, __len_m_snap)
    m_stack_d = Vector{Float64}(undef, length(m_stack))
    x_stack_d = Vector{Float64}(undef, length(x_stack))
    m_snap_snap_stack_d = Vector{Float64}(undef, length(m_snap_snap_stack))
    for i_ = 1:i_njac
        __idx_m_snap_snap_stack_0 = (i_ - 1) + 1
        __sbase_m_snap = (__idx_m_snap_snap_stack_0 - 1) * __len_m_snap
        for __sk_m_snap = 1:__len_m_snap
            m_snap_snap_stack_d[__sbase_m_snap + __sk_m_snap] = m_snapd[__sk_m_snap]
            m_snap_snap_stack[__sbase_m_snap + __sk_m_snap] = m_snap[__sk_m_snap]
        end
        for i_node = 1:i_nnode
            __idx_m_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            m_stack_d[__idx_m_stack_0] = md[i_node]
            m_stack[__idx_m_stack_0] = m[i_node]
            md[i_node] = 0.0
            m[i_node] = 0.0
            m_snapd[i_node] = 0.0
            m_snap[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                __idx_m_stack_1 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + (((i_cell - 1) + ((i_ - 1) * (div(3 - 1, 1) + 1) + (i_loc - 1)) * (div(i_ncell - 1, 1) + 1)) + 1)
                __cse_0d = md[i_n]
                __cse_0 = m[i_n]
                m_stack_d[__idx_m_stack_1] = __cse_0d
                m_stack[__idx_m_stack_1] = __cse_0
                __cse_1d = wd[i_cell]
                __cse_1 = w[i_cell]
                __hcse_0 = x[i_n]
                md[i_n] = __cse_0d + ((__hcse_0 * 0.1) * __cse_1d + (__cse_1 * 0.1) * xd[i_n])
                m[i_n] = __cse_0 + __cse_1 * __hcse_0 * 0.1
                m_snapd[i_n] = m_snapd[i_n] + 0.05__cse_1d
                m_snap[i_n] = m_snap[i_n] + __cse_1 * 0.05
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
            __cse_4d = m_snapd[i_node]
            __cse_4 = m_snap[i_node]
            __hcse_1 = (__cse_3 * 0.5) * __cse_3d
            __hcse_2 = (__cse_4 * 0.25) * __cse_4d
            xd[i_node] = (__cse_2d + (__hcse_1 + __hcse_1)) + (__hcse_2 + __hcse_2)
            x[i_node] = __cse_2 + __cse_3 * __cse_3 * 0.5 + __cse_4 * __cse_4 * 0.25
        end
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __idx_x_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            xd[i_node] = x_stack_d[__idx_x_stack_0]
            x[i_node] = x_stack[__idx_x_stack_0]
            __cse_5d = xbd[i_node]
            __cse_5 = xb[i_node]
            __hcse_3 = m[i_node] * 0.5
            __cse_6d = __cse_5 * (0.5 * md[i_node]) + __hcse_3 * __cse_5d
            __cse_6 = __hcse_3 * __cse_5
            mbd[i_node] = mbd[i_node] + __cse_6d
            mb[i_node] = mb[i_node] + __cse_6
            mbd[i_node] = mbd[i_node] + __cse_6d
            mb[i_node] = mb[i_node] + __cse_6
            __hcse_4 = m_snap[i_node] * 0.25
            __cse_7d = __cse_5 * (0.25 * m_snapd[i_node]) + __hcse_4 * __cse_5d
            __cse_7 = __hcse_4 * __cse_5
            m_snapbd[i_node] = m_snapbd[i_node] + __cse_7d
            m_snapb[i_node] = m_snapb[i_node] + __cse_7
            m_snapbd[i_node] = m_snapbd[i_node] + __cse_7d
            m_snapb[i_node] = m_snapb[i_node] + __cse_7
        end
        for i_cell = i_ncell:-1:1
            for i_loc = 3:-1:1
                i_n = i_cell_to_node[i_loc, i_cell]
                wbd[i_cell] = wbd[i_cell] + 0.05 * m_snapbd[i_n]
                wb[i_cell] = wb[i_cell] + 0.05 * m_snapb[i_n]
                __idx_m_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + (((i_cell - 1) + ((i_ - 1) * (div(3 - 1, 1) + 1) + (i_loc - 1)) * (div(i_ncell - 1, 1) + 1)) + 1)
                md[i_n] = m_stack_d[__idx_m_stack_0]
                m[i_n] = m_stack[__idx_m_stack_0]
                __cse_8d = mbd[i_n]
                __cse_8 = mb[i_n]
                __hcse_5 = x[i_n] * 0.1
                wbd[i_cell] = wbd[i_cell] + (__cse_8 * (0.1 * xd[i_n]) + __hcse_5 * __cse_8d)
                wb[i_cell] = wb[i_cell] + __hcse_5 * __cse_8
                __hcse_6 = w[i_cell] * 0.1
                xbd[i_n] = xbd[i_n] + (__cse_8 * (0.1 * wd[i_cell]) + __hcse_6 * __cse_8d)
                xb[i_n] = xb[i_n] + __hcse_6 * __cse_8
            end
        end
        for i_node = i_nnode:-1:1
            m_snapbd[i_node] = 0.0
            m_snapb[i_node] = 0.0
            __idx_m_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            md[i_node] = m_stack_d[__idx_m_stack_0]
            m[i_node] = m_stack[__idx_m_stack_0]
            mbd[i_node] = 0.0
            mb[i_node] = 0.0
        end
        __idx_m_snap_snap_stack_0 = (i_ - 1) + 1
        __sbase_m_snap = (__idx_m_snap_snap_stack_0 - 1) * __len_m_snap
        for __sk_m_snap = 1:__len_m_snap
            m_snapd[__sk_m_snap] = m_snap_snap_stack_d[__sbase_m_snap + __sk_m_snap]
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
