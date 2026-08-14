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