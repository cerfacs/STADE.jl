function ttgc(u, i_cell_to_node, cell_vol, skx, i_ncell, i_nnode, c, dt, beta, gamma, res, up, loss)
    for i_cell = 1:i_ncell
        cavgx = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            cavgx = cavgx + c[1, i_node] / 4
        end
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            vere = -(1.0 / 3.0) * u[i_node] * c[1, i_node] * skx[i_loc, i_cell]
            re = vere / cell_vol[i_cell]
            aerex = cavgx * re
            resi = -(dt / 4) * (0.5 - gamma[i_cell]) * vere
            for i_k = 1:4
                aeresk = aerex * skx[i_k, i_cell]
                factor = aeresk * dt ^ 2
                auxres = resi + factor * beta[i_cell]
                i_k_node = i_cell_to_node[i_k, i_cell]
                res[i_k_node] = res[i_k_node] + auxres
            end
        end
    end
    for i_seq_node = 1:i_nnode
        loss[1] = loss[1] + ((u[i_seq_node] + up[i_seq_node])) ^ 2
    end
end