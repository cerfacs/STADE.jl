# asnap_twoarrays(x, w, i_cell_to_node, m1, m2, i_ncell, i_nnode, i_njac)
#
# Two accumulators with the same shape in one loop body. Each array gets its own snapshot.
function asnap_twoarrays(x, w, i_cell_to_node, m1, m2, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            m1[i_node] = 0.0
            m2[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m1[i_n] = m1[i_n] + w[i_cell] * x[i_n] * 0.1
                m2[i_n] = m2[i_n] + w[i_cell] * 0.2
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m1[i_node] * m2[i_node] * 0.5
        end
    end
    return nothing
end
