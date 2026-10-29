function initstacks_asnap_outside_b(i_njac, i_nnode, m)
    __len_m = length(m)
    x_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    m_snap_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * __len_m)
    return (x_stack, m_snap_stack, __len_m)
end

function asnap_outside_hv(x, xb, y, yb, w, wb, i_cell_to_node, m, mb, i_ncell, i_nnode, i_njac, xd, xbd, yd, ybd, wd, wbd, md, mbd, x_stack, m_snap_stack, __len_m)
    x_stack_d = Vector{Float64}(undef, length(x_stack))
    m_snap_stack_d = Vector{Float64}(undef, length(m_snap_stack))
    for i_node = 1:i_nnode
        __cse_0d = md[i_node]
        __cse_0 = m[i_node]
        __hcse_0 = (__cse_0 * 0.4) * __cse_0d
        yd[i_node] = yd[i_node] + (__hcse_0 + __hcse_0)
        y[i_node] = y[i_node] + __cse_0 * __cse_0 * 0.4
    end
    for i_ = 1:i_njac
        __idx_m_snap_stack_0 = (i_ - 1) + 1
        __sbase_m = (__idx_m_snap_stack_0 - 1) * __len_m
        for __sk_m = 1:__len_m
            m_snap_stack_d[__sbase_m + __sk_m] = md[__sk_m]
            m_snap_stack[__sbase_m + __sk_m] = m[__sk_m]
        end
        for i_node = 1:i_nnode
            md[i_node] = 0.0
            m[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                __hcse_1 = x[i_n]
                __hcse_2 = w[i_cell]
                md[i_n] = md[i_n] + ((__hcse_1 * 0.1) * wd[i_cell] + (__hcse_2 * 0.1) * xd[i_n])
                m[i_n] = m[i_n] + __hcse_2 * __hcse_1 * 0.1
            end
        end
        for i_node = 1:i_nnode
            __idx_x_stack_0 = ((i_ - 1) + (i_node - 1) * (div(i_njac - 1, 1) + 1)) + 1
            __cse_1d = xd[i_node]
            __cse_1 = x[i_node]
            x_stack_d[__idx_x_stack_0] = __cse_1d
            x_stack[__idx_x_stack_0] = __cse_1
            __cse_2d = md[i_node]
            __cse_2 = m[i_node]
            __hcse_3 = (__cse_2 * 0.5) * __cse_2d
            xd[i_node] = __cse_1d + (__hcse_3 + __hcse_3)
            x[i_node] = __cse_1 + __cse_2 * __cse_2 * 0.5
        end
    end
    for i_node = 1:i_nnode
        __cse_3d = md[i_node]
        __cse_3 = m[i_node]
        __hcse_4 = (__cse_3 * 0.6) * __cse_3d
        yd[i_node] = yd[i_node] + (__hcse_4 + __hcse_4)
        y[i_node] = y[i_node] + __cse_3 * __cse_3 * 0.6
    end
    for i_node = i_nnode:-1:1
        __hcse_5 = yb[i_node]
        __hcse_6 = m[i_node] * 0.6
        __cse_4d = __hcse_5 * (0.6 * md[i_node]) + __hcse_6 * ybd[i_node]
        __cse_4 = __hcse_6 * __hcse_5
        mbd[i_node] = mbd[i_node] + __cse_4d
        mb[i_node] = mb[i_node] + __cse_4
        mbd[i_node] = mbd[i_node] + __cse_4d
        mb[i_node] = mb[i_node] + __cse_4
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __idx_x_stack_0 = ((i_ - 1) + (i_node - 1) * (div(i_njac - 1, 1) + 1)) + 1
            xd[i_node] = x_stack_d[__idx_x_stack_0]
            x[i_node] = x_stack[__idx_x_stack_0]
            __hcse_7 = xb[i_node]
            __hcse_8 = m[i_node] * 0.5
            __cse_5d = __hcse_7 * (0.5 * md[i_node]) + __hcse_8 * xbd[i_node]
            __cse_5 = __hcse_8 * __hcse_7
            mbd[i_node] = mbd[i_node] + __cse_5d
            mb[i_node] = mb[i_node] + __cse_5
            mbd[i_node] = mbd[i_node] + __cse_5d
            mb[i_node] = mb[i_node] + __cse_5
        end
        for i_cell = i_ncell:-1:1
            for i_loc = 3:-1:1
                i_n = i_cell_to_node[i_loc, i_cell]
                __cse_6d = mbd[i_n]
                __cse_6 = mb[i_n]
                __hcse_9 = x[i_n] * 0.1
                wbd[i_cell] = wbd[i_cell] + (__cse_6 * (0.1 * xd[i_n]) + __hcse_9 * __cse_6d)
                wb[i_cell] = wb[i_cell] + __hcse_9 * __cse_6
                __hcse_10 = w[i_cell] * 0.1
                xbd[i_n] = xbd[i_n] + (__cse_6 * (0.1 * wd[i_cell]) + __hcse_10 * __cse_6d)
                xb[i_n] = xb[i_n] + __hcse_10 * __cse_6
            end
        end
        for i_node = i_nnode:-1:1
            mbd[i_node] = 0.0
            mb[i_node] = 0.0
        end
        __idx_m_snap_stack_0 = (i_ - 1) + 1
        __sbase_m = (__idx_m_snap_stack_0 - 1) * __len_m
        for __sk_m = 1:__len_m
            md[__sk_m] = m_snap_stack_d[__sbase_m + __sk_m]
            m[__sk_m] = m_snap_stack[__sbase_m + __sk_m]
        end
    end
    for i_node = i_nnode:-1:1
        __hcse_11 = yb[i_node]
        __hcse_12 = m[i_node] * 0.4
        __cse_7d = __hcse_11 * (0.4 * md[i_node]) + __hcse_12 * ybd[i_node]
        __cse_7 = __hcse_12 * __hcse_11
        mbd[i_node] = mbd[i_node] + __cse_7d
        mb[i_node] = mb[i_node] + __cse_7
        mbd[i_node] = mbd[i_node] + __cse_7d
        mb[i_node] = mb[i_node] + __cse_7
    end
    return nothing
end

function asnap_outside(x, y, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
    for i_node = 1:i_nnode
        y[i_node] = y[i_node] + m[i_node] * m[i_node] * 0.4
    end
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
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.5
        end
    end
    for i_node = 1:i_nnode
        y[i_node] = y[i_node] + m[i_node] * m[i_node] * 0.6
    end
end
