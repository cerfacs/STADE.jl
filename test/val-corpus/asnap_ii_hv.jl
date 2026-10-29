function initstacks_asnap_ii_b(i_ncell, i_njac, m)
    __len_m = length(m)
    auxu_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1))
    m_snap_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * __len_m)
    return (auxu_stack, m_snap_stack, __len_m)
end

function asnap_ii_hv(x, xb, w, wb, i_cell_to_node, m, mb, i_ncell, i_nnode, i_njac, xd, xbd, wd, wbd, md, mbd, auxu_stack, m_snap_stack, __len_m)
    auxu_stack_d = Vector{Float64}(undef, length(auxu_stack))
    m_snap_stack_d = Vector{Float64}(undef, length(m_snap_stack))
    auxu = 0.0
    auxub = 0.0
    auxud = 0.0
    auxubd = 0.0
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
            auxud = 0.0
            auxu = 0.0
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                auxud = auxud + wd[i_n]
                auxu = auxu + w[i_n]
            end
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                __hcse_0 = (auxu * 0.1) * auxud
                md[i_n] = md[i_n] + (__hcse_0 + __hcse_0)
                m[i_n] = m[i_n] + auxu * auxu * 0.1
            end
            __idx_auxu_stack_3 = ((i_ - 1) + (i_cell - 1) * (div(i_njac - 1, 1) + 1)) + 1
            auxu_stack_d[__idx_auxu_stack_3] = auxud
            auxu_stack[__idx_auxu_stack_3] = auxu
        end
        for i_node = 1:i_nnode
            __cse_0d = md[i_node]
            __cse_0 = m[i_node]
            __hcse_1 = (__cse_0 * 0.5) * __cse_0d
            xd[i_node] = xd[i_node] + (__hcse_1 + __hcse_1)
            x[i_node] = x[i_node] + __cse_0 * __cse_0 * 0.5
        end
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __hcse_2 = xb[i_node]
            __hcse_3 = m[i_node] * 0.5
            __cse_1d = __hcse_2 * (0.5 * md[i_node]) + __hcse_3 * xbd[i_node]
            __cse_1 = __hcse_3 * __hcse_2
            mbd[i_node] = mbd[i_node] + __cse_1d
            mb[i_node] = mb[i_node] + __cse_1
            mbd[i_node] = mbd[i_node] + __cse_1d
            mb[i_node] = mb[i_node] + __cse_1
        end
        for i_cell = i_ncell:-1:1
            __idx_auxu_stack_0 = ((i_ - 1) + (i_cell - 1) * (div(i_njac - 1, 1) + 1)) + 1
            auxud = auxu_stack_d[__idx_auxu_stack_0]
            auxu = auxu_stack[__idx_auxu_stack_0]
            for i_loc = 3:-1:1
                i_n = i_cell_to_node[i_loc, i_cell]
                __hcse_4 = mb[i_n]
                __hcse_5 = auxu * 0.1
                __cse_2d = __hcse_4 * (0.1auxud) + __hcse_5 * mbd[i_n]
                __cse_2 = __hcse_5 * __hcse_4
                auxubd = auxubd + __cse_2d
                auxub = auxub + __cse_2
                auxubd = auxubd + __cse_2d
                auxub = auxub + __cse_2
            end
            for i_loc = 1:3
                __icse_3 = i_cell_to_node[i_loc, i_cell]
                i_n = __icse_3
                auxud = auxud + wd[i_n]
                auxu = auxu + w[i_n]
                i_n = __icse_3
                wbd[i_n] = wbd[i_n] + auxubd
                wb[i_n] = wb[i_n] + auxub
            end
            auxubd = 0.0
            auxub = 0.0
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
