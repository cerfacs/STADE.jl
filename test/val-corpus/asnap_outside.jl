# asnap_outside(x, y, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
#
# m is read nonlinearly before the Jacobi loop and after it. Both reads sit outside the loop. The snapshot must leave them correct:
# the restore at the end of the first reverse iteration gives back the value m had on entry.
function asnap_outside(x, y, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
    for i_node = 1:i_nnode
        y[i_node] = y[i_node] + m[i_node] * m[i_node] * 0.4
    end
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
            x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.5
        end
    end
    for i_node = 1:i_nnode
        y[i_node] = y[i_node] + m[i_node] * m[i_node] * 0.6
    end
    return nothing
end
