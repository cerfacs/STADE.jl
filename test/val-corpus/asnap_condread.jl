# asnap_condread(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
#
# The shape of asnap_ok with an `if` between the two fills of m, and the condition of the `if` reads m. Version 1 of the array snapshot
# refuses it (guard G5). The refusal is cautious: the reverse sweep replays the branch from a saved flag and never reads the condition
# again, so the snapshot would still be right. No oracle can tell the two apart; the test reports it and does not require it.
#
# x: nodal state, updated in place
# w: per-cell weight
# i_cell_to_node: 3 x i_ncell node numbers
# m: nodal accumulator, reset and updated in place
function asnap_condread(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
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
        if m[1] > 0.3
            x[1] = x[1] * 1.1
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] + w[i_cell] * 0.05
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.5
        end
    end
    return nothing
end
