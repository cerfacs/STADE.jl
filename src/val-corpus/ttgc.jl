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