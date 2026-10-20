# asnap_nested(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac, i_nout)
#
# A loop nest. The outer loop reads m inside its first statement, the inner Jacobi loop, after the writes of that statement, so the
# outer loop is refused (guard G4 reads the nest as one statement). The inner loop meets every guard and owns the one snapshot of m.
function asnap_nested(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac, i_nout)
    for i_o = 1:i_nout
        for i_ = 1:i_njac
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
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.2
        end
    end
    return nothing
end
