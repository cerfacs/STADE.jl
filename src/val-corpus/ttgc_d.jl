function ttgc_d(i_cell_to_node, node_vol, node_vold, i_ncell, i_nnode, i_njac, up, upd, mup, mupd)
    for i_seq_ = 1:i_njac
        for i_node = 1:i_nnode
            mupd[i_node] = 0.0
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            i_node1d = 0.0
            i_node1 = i_cell_to_node[1, i_cell]
            mupd[i_node1] = mupd[i_node1] + upd[i_node1]
            mup[i_node1] = mup[i_node1] + up[i_node1]
        end
        for i_node = 1:i_nnode
            upd[i_node] = upd[i_node] + -(((1.0 / node_vol[i_node]) * mupd[i_node] + -(mup[i_node] / node_vol[i_node] ^ 2) * node_vold[i_node]))
            up[i_node] = up[i_node] - mup[i_node] / node_vol[i_node]
        end
    end
    return nothing
end

function ttgc(i_cell_to_node, node_vol, i_ncell, i_nnode, i_njac, up, mup)
    for i_seq_ = 1:i_njac
        for i_node = 1:i_nnode
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            i_node1 = i_cell_to_node[1, i_cell]
            mup[i_node1] = mup[i_node1] + up[i_node1]
        end
        for i_node = 1:i_nnode
            up[i_node] = up[i_node] - mup[i_node] / node_vol[i_node]
        end
    end
end
