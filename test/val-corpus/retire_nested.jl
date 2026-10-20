# retire_nested(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_nlev, i_njac)
#
# A ragged (Tier B) kernel whose cell count is reassigned AFTER the loops that use it, where the loop that uses it is NOT a
# top-level statement of the level body: it sits inside the Jacobi loop. coarsen_retire has the consumer at the top level.
# Here the sizing pass must still see that the level's inner loops have consumed `nc`, so the reassignment at the end of the
# level body belongs to the next level. Before 0.4.9 the pass looked at top-level loop headers only, recorded each level's
# size and value table from the value AFTER the reassignment, and every block overran its successor (BoundsError).
#
# x: nodal state, updated in place
# w: per-cell weight
# i_cell_to_node: 3 x i_ncell node numbers
# m: nodal accumulator, reset and updated in place
function retire_nested(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_nlev, i_njac)
    nc = i_ncell
    for lev = 1:i_nlev
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
        nc = nc - 1
    end
    return nothing
end
