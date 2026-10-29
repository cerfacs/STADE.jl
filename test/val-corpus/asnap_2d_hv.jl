function initstacks_asnap_2d_b(M, i_njac, i_nnode)
    __len_M = length(M)
    x_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    M_snap_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * __len_M)
    return (x_stack, M_snap_stack, __len_M)
end

function asnap_2d_hv(x, xb, w, wb, i_cell_to_node, M, Mb, i_ncell, i_nnode, i_njac, xd, xbd, wd, wbd, Md, Mbd, x_stack, M_snap_stack, __len_M)
    x_stack_d = Vector{Float64}(undef, length(x_stack))
    M_snap_stack_d = Vector{Float64}(undef, length(M_snap_stack))
    for i_ = 1:i_njac
        __idx_M_snap_stack_0 = (i_ - 1) + 1
        __sbase_M = (__idx_M_snap_stack_0 - 1) * __len_M
        for __sk_M = 1:__len_M
            M_snap_stack_d[__sbase_M + __sk_M] = Md[__sk_M]
            M_snap_stack[__sbase_M + __sk_M] = M[__sk_M]
        end
        for i_node = 1:i_nnode
            for q = 1:2
                Md[q, i_node] = 0.0
                M[q, i_node] = 0.0
            end
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                for q = 1:2
                    __hcse_0 = x[i_n]
                    __hcse_1 = w[i_cell]
                    Md[q, i_n] = Md[q, i_n] + ((__hcse_0 * 0.1) * wd[i_cell] + (__hcse_1 * 0.1) * xd[i_n])
                    M[q, i_n] = M[q, i_n] + __hcse_1 * __hcse_0 * 0.1
                end
            end
        end
        for i_node = 1:i_nnode
            __idx_x_stack_0 = ((i_ - 1) + (i_node - 1) * (div(i_njac - 1, 1) + 1)) + 1
            __cse_0d = xd[i_node]
            __cse_0 = x[i_node]
            x_stack_d[__idx_x_stack_0] = __cse_0d
            x_stack[__idx_x_stack_0] = __cse_0
            __hcse_2 = M[2, i_node]
            __hcse_3 = M[1, i_node]
            xd[i_node] = __cse_0d + ((__hcse_2 * 0.5) * Md[1, i_node] + (__hcse_3 * 0.5) * Md[2, i_node])
            x[i_node] = __cse_0 + __hcse_3 * __hcse_2 * 0.5
        end
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __idx_x_stack_0 = ((i_ - 1) + (i_node - 1) * (div(i_njac - 1, 1) + 1)) + 1
            xd[i_node] = x_stack_d[__idx_x_stack_0]
            x[i_node] = x_stack[__idx_x_stack_0]
            __cse_1d = xbd[i_node]
            __cse_1 = xb[i_node]
            __hcse_4 = M[2, i_node] * 0.5
            Mbd[1, i_node] = Mbd[1, i_node] + (__cse_1 * (0.5 * Md[2, i_node]) + __hcse_4 * __cse_1d)
            Mb[1, i_node] = Mb[1, i_node] + __hcse_4 * __cse_1
            __hcse_5 = M[1, i_node] * 0.5
            Mbd[2, i_node] = Mbd[2, i_node] + (__cse_1 * (0.5 * Md[1, i_node]) + __hcse_5 * __cse_1d)
            Mb[2, i_node] = Mb[2, i_node] + __hcse_5 * __cse_1
        end
        for i_cell = i_ncell:-1:1
            for i_loc = 3:-1:1
                i_n = i_cell_to_node[i_loc, i_cell]
                for q = 2:-1:1
                    __cse_2d = Mbd[q, i_n]
                    __cse_2 = Mb[q, i_n]
                    __hcse_6 = x[i_n] * 0.1
                    wbd[i_cell] = wbd[i_cell] + (__cse_2 * (0.1 * xd[i_n]) + __hcse_6 * __cse_2d)
                    wb[i_cell] = wb[i_cell] + __hcse_6 * __cse_2
                    __hcse_7 = w[i_cell] * 0.1
                    xbd[i_n] = xbd[i_n] + (__cse_2 * (0.1 * wd[i_cell]) + __hcse_7 * __cse_2d)
                    xb[i_n] = xb[i_n] + __hcse_7 * __cse_2
                end
            end
        end
        for i_node = i_nnode:-1:1
            for q = 2:-1:1
                Mbd[q, i_node] = 0.0
                Mb[q, i_node] = 0.0
            end
        end
        __idx_M_snap_stack_0 = (i_ - 1) + 1
        __sbase_M = (__idx_M_snap_stack_0 - 1) * __len_M
        for __sk_M = 1:__len_M
            Md[__sk_M] = M_snap_stack_d[__sbase_M + __sk_M]
            M[__sk_M] = M_snap_stack[__sbase_M + __sk_M]
        end
    end
    return nothing
end

function asnap_2d(x, w, i_cell_to_node, M, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            for q = 1:2
                M[q, i_node] = 0.0
            end
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                for q = 1:2
                    M[q, i_n] = M[q, i_n] + w[i_cell] * x[i_n] * 0.1
                end
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + M[1, i_node] * M[2, i_node] * 0.5
        end
    end
end
