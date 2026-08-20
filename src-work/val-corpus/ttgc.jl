function ttgc(u, uref, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup, loss)
    for i_cell = 1:i_ncell
        cavgx = 0.0
        cavgy = 0.0
        cavgz = 0.0
        for i_seq_loc = 1:4
            i_node = i_cell_to_node[i_seq_loc, i_cell]
            cavgx = cavgx + c[1, i_node] / 4
            cavgy = cavgy + c[2, i_node] / 4
            cavgz = cavgz + c[3, i_node] / 4
        end
        for i_seq_loc = 1:4
            i_node = i_cell_to_node[i_seq_loc, i_cell]
            vere = -(1.0 / 3.0) * u[i_node] * (c[1, i_node] * skx[i_seq_loc, i_cell] + c[2, i_node] * sky[i_seq_loc, i_cell] + c[3, i_node] * skz[i_seq_loc, i_cell])
            re = vere / cell_vol[i_cell]
            aerex = cavgx * re
            aerey = cavgy * re
            aerez = cavgz * re
            resi = -(dt / 4) * (0.5 - gamma[i_cell]) * vere
            for i_seq_k = 1:4
                aeresk = aerex * skx[i_seq_k, i_cell] + aerey * sky[i_seq_k, i_cell] + aerez * skz[i_seq_k, i_cell]
                factor = (dt ^ 2 / 3) * aeresk
                auxres = resi + factor * beta[i_cell]
                auxres2 = factor * gamma[i_cell]
                i_seq_k_node = i_cell_to_node[i_seq_k, i_cell]
                res[i_seq_k_node] = res[i_seq_k_node] + auxres
                res2[i_seq_k_node] = res2[i_seq_k_node] + auxres2
            end
        end
    end
    for i_node = 1:i_nnode
        up[i_node] = res[i_node] / node_vol[i_node]
    end
    for i_seq_ = 1:i_njac
        for i_node = 1:i_nnode
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            i_node1 = i_cell_to_node[1, i_cell]
            i_node2 = i_cell_to_node[2, i_cell]
            i_node3 = i_cell_to_node[3, i_cell]
            i_node4 = i_cell_to_node[4, i_cell]
            auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
            mup[i_node1] = mup[i_node1] + ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20.0
            mup[i_node2] = mup[i_node2] + ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20.0
            mup[i_node3] = mup[i_node3] + ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20.0
            mup[i_node4] = mup[i_node4] + ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20.0
        end
        for i_node = 1:i_nnode
            up[i_node] = up[i_node] + (res[i_node] - mup[i_node]) / node_vol[i_node]
        end
    end
    for i_cell = 1:i_ncell
        for i_seq_loc = 1:4
            i_node = i_cell_to_node[i_seq_loc, i_cell]
            vere = -(1.0 / 3.0) * (u[i_node] + up[i_node]) * (c[1, i_node] * skx[i_seq_loc, i_cell] + c[2, i_node] * sky[i_seq_loc, i_cell])
            resi = -(dt / 4) * vere
            for i_seq_k = 1:4
                i_seq_k_node = i_cell_to_node[i_seq_k, i_cell]
                res2[i_seq_k_node] = res2[i_seq_k_node] + resi
            end
        end
    end
    for i_node = 1:i_nnode
        up[i_node] = res2[i_node] / node_vol[i_node]
    end
    for i_seq_ = 1:i_njac
        for i_node = 1:i_nnode
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            i_node1 = i_cell_to_node[1, i_cell]
            i_node2 = i_cell_to_node[2, i_cell]
            i_node3 = i_cell_to_node[3, i_cell]
            i_node4 = i_cell_to_node[4, i_cell]
            auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
            mup[i_node1] = mup[i_node1] + ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20.0
            mup[i_node2] = mup[i_node2] + ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20.0
            mup[i_node3] = mup[i_node3] + ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20.0
            mup[i_node4] = mup[i_node4] + ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20.0
        end
        for i_node = 1:i_nnode
            up[i_node] = up[i_node] + (res2[i_node] - mup[i_node]) / node_vol[i_node]
        end
    end
    for i_seq_node = 1:i_nnode
        loss[1] = loss[1] + ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node]) ^ 2
    end
end