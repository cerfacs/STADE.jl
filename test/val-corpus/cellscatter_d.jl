function cellscatter_d(i_cell_to_node, cell_vol, cell_vold, i_ncell, i_nnode, i_njac, up, upd, mup, mupd)
    for i_ = 1:i_njac
        for i_cell = 1:i_ncell
            auxud = 0.0
            auxu = 0.0
            for i_loc = 1:4
                i_lnoded = 0.0
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxud = auxud + upd[i_lnode]
                auxu = auxu + up[i_lnode]
            end
            for i_loc = 1:4
                i_lnoded = 0.0
                i_lnode = i_cell_to_node[i_loc, i_cell]
                mupd[i_lnode] = mupd[i_lnode] + (cell_vol[i_cell] * (auxud + upd[i_lnode]) + (auxu + up[i_lnode]) * cell_vold[i_cell])
                mup[i_lnode] = mup[i_lnode] + (auxu + up[i_lnode]) * cell_vol[i_cell]
            end
        end
        for i_node = 1:i_nnode
            upd[i_node] = upd[i_node] + -(mupd[i_node])
            up[i_node] = up[i_node] - mup[i_node]
        end
    end
    return nothing
end

function cellscatter(i_cell_to_node, cell_vol, i_ncell, i_nnode, i_njac, up, mup)
    for i_ = 1:i_njac
        for i_cell = 1:i_ncell
            auxu = 0.0
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxu = auxu + up[i_lnode]
            end
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                mup[i_lnode] = mup[i_lnode] + (auxu + up[i_lnode]) * cell_vol[i_cell]
            end
        end
        for i_node = 1:i_nnode
            up[i_node] = up[i_node] - mup[i_node]
        end
    end
end
