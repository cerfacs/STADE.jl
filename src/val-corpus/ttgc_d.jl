function ttgc_d(u, ud, i_cell_to_node, cell_vol, cell_vold, i_ncell, res, resd)
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_noded = 0.0
            i_node = i_cell_to_node[i_loc, i_cell]
            red = (1.0 / cell_vol[i_cell]) * ud[i_node] + -(u[i_node] / cell_vol[i_cell] ^ 2) * cell_vold[i_cell]
            re = u[i_node] / cell_vol[i_cell]
            for i_k = 1:4
                i_k_noded = 0.0
                i_k_node = i_cell_to_node[i_k, i_cell]
                resd[i_k_node] = resd[i_k_node] + (u[i_node] * red + re * ud[i_node])
                res[i_k_node] = res[i_k_node] + re * u[i_node]
            end
        end
    end
    return nothing
end

function ttgc(u, i_cell_to_node, cell_vol, i_ncell, res)
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            re = u[i_node] / cell_vol[i_cell]
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
                res[i_k_node] = res[i_k_node] + re * u[i_node]
            end
        end
    end
end
