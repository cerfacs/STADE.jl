# asnap_readbefore(x, w, i_cell_to_node, m, y, i_ncell, i_nnode, i_njac)
#
# A nonlinear read of m at the top of the loop body, before any write to m in the same body. The read sees the value the previous
# iteration left. The array snapshot must be refused (guard G4) and the per-write saves must stay.
#
# y: nodal output, updated in place
function asnap_readbefore(x, w, i_cell_to_node, m, y, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            y[i_node] = y[i_node] + m[i_node] * m[i_node] * 0.3
        end
        for i_node = 1:i_nnode
            m[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] + w[i_cell] * x[i_n] * 0.1
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * 0.5
        end
    end
    return nothing
end
