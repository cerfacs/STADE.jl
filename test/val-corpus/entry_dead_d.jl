function entry_dead_d(x, xd, y, yd, i_cell_to_node, i_ncell, i_nnode, res, resd, out, outd)
    vd = 0.0
    v = 0.0
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_noded = 0.0
            i_node = i_cell_to_node[i_loc, i_cell]
            __cse_0 = y[i_cell]
            __cse_1 = x[i_node]
            vd = __cse_0 * xd[i_node] + __cse_1 * yd[i_cell]
            v = __cse_1 * __cse_0
            for i_k = 1:4
                i_k_noded = 0.0
                i_k_node = i_cell_to_node[i_k, i_cell]
                __cse_2 = v * vd
                resd[i_k_node] = resd[i_k_node] + (__cse_2 + __cse_2)
                res[i_k_node] = res[i_k_node] + v * v
            end
        end
    end
    for i_x = 1:i_nnode
        __cse_3 = res[i_x]
        __cse_4 = __cse_3 * resd[i_x]
        outd[i_x] = outd[i_x] + (__cse_4 + __cse_4)
        out[i_x] = out[i_x] + __cse_3 * __cse_3
    end
    return nothing
end

function entry_dead(x, y, i_cell_to_node, i_ncell, i_nnode, res, out)
    v = 0.0
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            v = x[i_node] * y[i_cell]
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
                res[i_k_node] = res[i_k_node] + v * v
            end
        end
    end
    for i_x = 1:i_nnode
        out[i_x] = out[i_x] + res[i_x] * res[i_x]
    end
end
