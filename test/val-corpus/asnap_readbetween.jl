# asnap_readbetween(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
#
# A nonlinear read of m between two statements that write m. The read must see the value after the first fill only. The snapshot must
# be refused (guard G4).
function asnap_readbetween(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            m[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] + w[i_cell] * 0.1
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.5
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] + x[i_n] * 0.05
            end
        end
    end
    return nothing
end
