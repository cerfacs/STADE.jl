function initstacks_asnap_nested_b(i_njac, i_nnode, i_nout, m)
    __len_m = length(m)
    x_stack = Vector{Float64}(undef, max(0, div(i_nout - 1, 1) + 1) * max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_nout - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    m_snap_stack = Vector{Float64}(undef, (max(0, div(i_nout - 1, 1) + 1) * max(0, div(i_njac - 1, 1) + 1)) * __len_m)
    return (x_stack, m_snap_stack, __len_m)
end

function asnap_nested_hv(x, xb, w, wb, i_cell_to_node, m, mb, i_ncell, i_nnode, i_njac, i_nout, xd, xbd, wd, wbd, md, mbd, x_stack, m_snap_stack, __len_m)
    x_stack_d = Vector{Float64}(undef, length(x_stack))
    m_snap_stack_d = Vector{Float64}(undef, length(m_snap_stack))
    for i_o = 1:i_nout
        for i_ = 1:i_njac
            __idx_m_snap_stack_0 = ((i_o - 1) * (div(i_njac - 1, 1) + 1) + (i_ - 1)) + 1
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
                    __hcse_0 = x[i_n]
                    __hcse_1 = w[i_cell]
                    md[i_n] = md[i_n] + ((__hcse_0 * 0.1) * wd[i_cell] + (__hcse_1 * 0.1) * xd[i_n])
                    m[i_n] = m[i_n] + __hcse_1 * __hcse_0 * 0.1
                end
            end
            for i_node = 1:i_nnode
                __idx_x_stack_0 = ((i_node - 1) + ((i_o - 1) * (div(i_njac - 1, 1) + 1) + (i_ - 1)) * (div(i_nnode - 1, 1) + 1)) + 1
                __cse_0d = xd[i_node]
                __cse_0 = x[i_node]
                x_stack_d[__idx_x_stack_0] = __cse_0d
                x_stack[__idx_x_stack_0] = __cse_0
                xd[i_node] = __cse_0d + 0.5 * md[i_node]
                x[i_node] = __cse_0 + m[i_node] * 0.5
            end
        end
        for i_node = 1:i_nnode
            __icse_1 = div(i_nnode - 1, 1) + 1
            __idx_x_stack_0 = max(0, div(i_nout - 1, 1) + 1) * max(0, div(i_njac - 1, 1) + 1) * max(0, __icse_1) + (((i_node - 1) + (i_o - 1) * __icse_1) + 1)
            __cse_2d = xd[i_node]
            __cse_2 = x[i_node]
            x_stack_d[__idx_x_stack_0] = __cse_2d
            x_stack[__idx_x_stack_0] = __cse_2
            __cse_3d = md[i_node]
            __cse_3 = m[i_node]
            __hcse_2 = (__cse_3 * 0.2) * __cse_3d
            xd[i_node] = __cse_2d + (__hcse_2 + __hcse_2)
            x[i_node] = __cse_2 + __cse_3 * __cse_3 * 0.2
        end
    end
    for i_o = i_nout:-1:1
        for i_node = i_nnode:-1:1
            __icse_4 = div(i_nnode - 1, 1) + 1
            __idx_x_stack_0 = max(0, div(i_nout - 1, 1) + 1) * max(0, div(i_njac - 1, 1) + 1) * max(0, __icse_4) + (((i_node - 1) + (i_o - 1) * __icse_4) + 1)
            xd[i_node] = x_stack_d[__idx_x_stack_0]
            x[i_node] = x_stack[__idx_x_stack_0]
            __hcse_3 = xb[i_node]
            __hcse_4 = m[i_node] * 0.2
            __cse_5d = __hcse_3 * (0.2 * md[i_node]) + __hcse_4 * xbd[i_node]
            __cse_5 = __hcse_4 * __hcse_3
            mbd[i_node] = mbd[i_node] + __cse_5d
            mb[i_node] = mb[i_node] + __cse_5
            mbd[i_node] = mbd[i_node] + __cse_5d
            mb[i_node] = mb[i_node] + __cse_5
        end
        for i_ = i_njac:-1:1
            for i_node = i_nnode:-1:1
                __idx_x_stack_0 = ((i_node - 1) + ((i_o - 1) * (div(i_njac - 1, 1) + 1) + (i_ - 1)) * (div(i_nnode - 1, 1) + 1)) + 1
                xd[i_node] = x_stack_d[__idx_x_stack_0]
                x[i_node] = x_stack[__idx_x_stack_0]
                mbd[i_node] = mbd[i_node] + 0.5 * xbd[i_node]
                mb[i_node] = mb[i_node] + 0.5 * xb[i_node]
            end
            for i_cell = i_ncell:-1:1
                for i_loc = 3:-1:1
                    i_n = i_cell_to_node[i_loc, i_cell]
                    __cse_6d = mbd[i_n]
                    __cse_6 = mb[i_n]
                    __hcse_5 = x[i_n] * 0.1
                    wbd[i_cell] = wbd[i_cell] + (__cse_6 * (0.1 * xd[i_n]) + __hcse_5 * __cse_6d)
                    wb[i_cell] = wb[i_cell] + __hcse_5 * __cse_6
                    __hcse_6 = w[i_cell] * 0.1
                    xbd[i_n] = xbd[i_n] + (__cse_6 * (0.1 * wd[i_cell]) + __hcse_6 * __cse_6d)
                    xb[i_n] = xb[i_n] + __hcse_6 * __cse_6
                end
            end
            for i_node = i_nnode:-1:1
                mbd[i_node] = 0.0
                mb[i_node] = 0.0
            end
            __idx_m_snap_stack_0 = ((i_o - 1) * (div(i_njac - 1, 1) + 1) + (i_ - 1)) + 1
            __sbase_m = (__idx_m_snap_stack_0 - 1) * __len_m
            for __sk_m = 1:__len_m
                md[__sk_m] = m_snap_stack_d[__sbase_m + __sk_m]
                m[__sk_m] = m_snap_stack[__sbase_m + __sk_m]
            end
        end
    end
    return nothing
end

function asnap_nested(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac, i_nout)
    for i_o = 1:i_nout
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
                x[i_node] = x[i_node] + m[i_node] * 0.5
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.2
        end
    end
end
