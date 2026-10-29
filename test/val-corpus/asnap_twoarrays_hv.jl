function initstacks_asnap_twoarrays_b(i_njac, i_nnode, m1, m2)
    __len_m1 = length(m1)
    __len_m2 = length(m2)
    x_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    m1_snap_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * __len_m1)
    m2_snap_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * __len_m2)
    return (x_stack, m1_snap_stack, m2_snap_stack, __len_m1, __len_m2)
end

function asnap_twoarrays_hv(x, xb, w, wb, i_cell_to_node, m1, m1b, m2, m2b, i_ncell, i_nnode, i_njac, xd, xbd, wd, wbd, m1d, m1bd, m2d, m2bd, x_stack, m1_snap_stack, m2_snap_stack, __len_m1, __len_m2)
    x_stack_d = Vector{Float64}(undef, length(x_stack))
    m1_snap_stack_d = Vector{Float64}(undef, length(m1_snap_stack))
    m2_snap_stack_d = Vector{Float64}(undef, length(m2_snap_stack))
    for i_ = 1:i_njac
        __idx_m1_snap_stack_0 = (i_ - 1) + 1
        __sbase_m1 = (__idx_m1_snap_stack_0 - 1) * __len_m1
        for __sk_m1 = 1:__len_m1
            m1_snap_stack_d[__sbase_m1 + __sk_m1] = m1d[__sk_m1]
            m1_snap_stack[__sbase_m1 + __sk_m1] = m1[__sk_m1]
        end
        __idx_m2_snap_stack_3 = (i_ - 1) + 1
        __sbase_m2 = (__idx_m2_snap_stack_3 - 1) * __len_m2
        for __sk_m2 = 1:__len_m2
            m2_snap_stack_d[__sbase_m2 + __sk_m2] = m2d[__sk_m2]
            m2_snap_stack[__sbase_m2 + __sk_m2] = m2[__sk_m2]
        end
        for i_node = 1:i_nnode
            m1d[i_node] = 0.0
            m1[i_node] = 0.0
            m2d[i_node] = 0.0
            m2[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                __cse_0d = wd[i_cell]
                __cse_0 = w[i_cell]
                __hcse_0 = x[i_n]
                m1d[i_n] = m1d[i_n] + ((__hcse_0 * 0.1) * __cse_0d + (__cse_0 * 0.1) * xd[i_n])
                m1[i_n] = m1[i_n] + __cse_0 * __hcse_0 * 0.1
                m2d[i_n] = m2d[i_n] + 0.2__cse_0d
                m2[i_n] = m2[i_n] + __cse_0 * 0.2
            end
        end
        for i_node = 1:i_nnode
            __idx_x_stack_0 = ((i_ - 1) + (i_node - 1) * (div(i_njac - 1, 1) + 1)) + 1
            __cse_1d = xd[i_node]
            __cse_1 = x[i_node]
            x_stack_d[__idx_x_stack_0] = __cse_1d
            x_stack[__idx_x_stack_0] = __cse_1
            __hcse_1 = m2[i_node]
            __hcse_2 = m1[i_node]
            xd[i_node] = __cse_1d + ((__hcse_1 * 0.5) * m1d[i_node] + (__hcse_2 * 0.5) * m2d[i_node])
            x[i_node] = __cse_1 + __hcse_2 * __hcse_1 * 0.5
        end
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __idx_x_stack_0 = ((i_ - 1) + (i_node - 1) * (div(i_njac - 1, 1) + 1)) + 1
            xd[i_node] = x_stack_d[__idx_x_stack_0]
            x[i_node] = x_stack[__idx_x_stack_0]
            __cse_2d = xbd[i_node]
            __cse_2 = xb[i_node]
            __hcse_3 = m2[i_node] * 0.5
            m1bd[i_node] = m1bd[i_node] + (__cse_2 * (0.5 * m2d[i_node]) + __hcse_3 * __cse_2d)
            m1b[i_node] = m1b[i_node] + __hcse_3 * __cse_2
            __hcse_4 = m1[i_node] * 0.5
            m2bd[i_node] = m2bd[i_node] + (__cse_2 * (0.5 * m1d[i_node]) + __hcse_4 * __cse_2d)
            m2b[i_node] = m2b[i_node] + __hcse_4 * __cse_2
        end
        for i_cell = i_ncell:-1:1
            for i_loc = 3:-1:1
                i_n = i_cell_to_node[i_loc, i_cell]
                wbd[i_cell] = wbd[i_cell] + 0.2 * m2bd[i_n]
                wb[i_cell] = wb[i_cell] + 0.2 * m2b[i_n]
                __cse_3d = m1bd[i_n]
                __cse_3 = m1b[i_n]
                __hcse_5 = x[i_n] * 0.1
                wbd[i_cell] = wbd[i_cell] + (__cse_3 * (0.1 * xd[i_n]) + __hcse_5 * __cse_3d)
                wb[i_cell] = wb[i_cell] + __hcse_5 * __cse_3
                __hcse_6 = w[i_cell] * 0.1
                xbd[i_n] = xbd[i_n] + (__cse_3 * (0.1 * wd[i_cell]) + __hcse_6 * __cse_3d)
                xb[i_n] = xb[i_n] + __hcse_6 * __cse_3
            end
        end
        for i_node = i_nnode:-1:1
            m2bd[i_node] = 0.0
            m2b[i_node] = 0.0
            m1bd[i_node] = 0.0
            m1b[i_node] = 0.0
        end
        __idx_m1_snap_stack_0 = (i_ - 1) + 1
        __sbase_m1 = (__idx_m1_snap_stack_0 - 1) * __len_m1
        for __sk_m1 = 1:__len_m1
            m1d[__sk_m1] = m1_snap_stack_d[__sbase_m1 + __sk_m1]
            m1[__sk_m1] = m1_snap_stack[__sbase_m1 + __sk_m1]
        end
        __idx_m2_snap_stack_3 = (i_ - 1) + 1
        __sbase_m2 = (__idx_m2_snap_stack_3 - 1) * __len_m2
        for __sk_m2 = 1:__len_m2
            m2d[__sk_m2] = m2_snap_stack_d[__sbase_m2 + __sk_m2]
            m2[__sk_m2] = m2_snap_stack[__sbase_m2 + __sk_m2]
        end
    end
    return nothing
end

function asnap_twoarrays(x, w, i_cell_to_node, m1, m2, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            m1[i_node] = 0.0
            m2[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m1[i_n] = m1[i_n] + w[i_cell] * x[i_n] * 0.1
                m2[i_n] = m2[i_n] + w[i_cell] * 0.2
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m1[i_node] * m2[i_node] * 0.5
        end
    end
end
