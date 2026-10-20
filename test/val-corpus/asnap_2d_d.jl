function asnap_2d_d(x, xd, w, wd, i_cell_to_node, M, Md, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            for q = 1:2
                Md[q, i_node] = 0.0
                M[q, i_node] = 0.0
            end
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_nd = 0.0
                i_n = i_cell_to_node[i_loc, i_cell]
                for q = 1:2
                    __cse_0 = x[i_n]
                    __cse_1 = w[i_cell]
                    Md[q, i_n] = Md[q, i_n] + ((__cse_0 * 0.1) * wd[i_cell] + (__cse_1 * 0.1) * xd[i_n])
                    M[q, i_n] = M[q, i_n] + __cse_1 * __cse_0 * 0.1
                end
            end
        end
        for i_node = 1:i_nnode
            __cse_2 = M[2, i_node]
            __cse_3 = M[1, i_node]
            xd[i_node] = xd[i_node] + ((__cse_2 * 0.5) * Md[1, i_node] + (__cse_3 * 0.5) * Md[2, i_node])
            x[i_node] = x[i_node] + __cse_3 * __cse_2 * 0.5
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
