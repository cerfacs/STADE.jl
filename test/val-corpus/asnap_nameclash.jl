# asnap_nameclash(x, w, i_cell_to_node, m, m_snap, i_ncell, i_nnode, i_njac)
#
# The shape of asnap_ok with a second array called m_snap, the name an array snapshot of m would give its stack
# (m_snap_stack). The snapshot of m must be refused, or two stacks share one name. m_snap itself may be snapshotted.
#
# x: nodal state, updated in place
# w: per-cell weight
# i_cell_to_node: 3 x i_ncell node numbers
# m: nodal accumulator, reset and updated in place
# m_snap: second nodal accumulator, updated in place
function asnap_nameclash(x, w, i_cell_to_node, m, m_snap, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            m[i_node] = 0.0
            m_snap[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] + w[i_cell] * x[i_n] * 0.1
                m_snap[i_n] = m_snap[i_n] + w[i_cell] * 0.05
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.5 + m_snap[i_node] * m_snap[i_node] * 0.25
        end
    end
    return nothing
end
