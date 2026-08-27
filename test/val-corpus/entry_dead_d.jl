function entry_dead_d(x, xd, y, yd, i_cell_to_node, i_ncell, i_nnode, res, resd, out, outd)
    vd = 0.0
    v = 0.0
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_noded = 0.0
            i_node = i_cell_to_node[i_loc, i_cell]
            vd = y[i_cell] * xd[i_node] + x[i_node] * yd[i_cell]
            v = x[i_node] * y[i_cell]
            for i_k = 1:4
                i_k_noded = 0.0
                i_k_node = i_cell_to_node[i_k, i_cell]
                resd[i_k_node] = resd[i_k_node] + (v * vd + v * vd)
                res[i_k_node] = res[i_k_node] + v * v
            end
        end
    end
    for i_x = 1:i_nnode
        outd[i_x] = outd[i_x] + (res[i_x] * resd[i_x] + res[i_x] * resd[i_x])
        out[i_x] = out[i_x] + res[i_x] * res[i_x]
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
