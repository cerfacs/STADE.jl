# asnap_ragged(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_nlev, i_njac)
#
# The Jacobi loop of asnap_ok inside a level loop that shrinks the cell count by one at the top of each level. The cell loop's bound is reassigned
# in an ancestor loop, so the level loop is a Tier B ragged-block ancestor and its layout uses block-local offsets. Version 1 of the array
# snapshot refuses a loop under such an ancestor (guard G9).
#
# x: nodal state, updated in place
# w: per-cell weight
# i_cell_to_node: 3 x i_ncell node numbers
# m: nodal accumulator, reset and updated in place
function asnap_ragged(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_nlev, i_njac)
    nc = i_ncell
    for lev = 1:i_nlev
        nc = nc - 1
        for i_ = 1:i_njac
            for i_node = 1:i_nnode
                m[i_node] = 0.0
            end
            for i_cell = 1:nc
                for i_loc = 1:3
                    i_n = i_cell_to_node[i_loc, i_cell]
                    m[i_n] = m[i_n] + w[i_cell] * x[i_n] * 0.1
                end
            end
            for i_node = 1:i_nnode
                x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.5
            end
        end
    end
    return nothing
end
