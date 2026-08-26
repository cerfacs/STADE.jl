# cellscatter(i_cell_to_node, cell_vol, i_ncell, i_nnode, i_njac, up, mup)
#
# Unstructured cell-to-node scatter driven by a sequential Jacobi loop --
# the canonical FEM/finite-volume assembly shape. Each cell gathers its
# four nodes into a scalar accumulator, scatters a volume-weighted
# contribution back to those same nodes, and a node loop then applies the
# update. `auxu` is reset at the top of the cell body and read after the
# gather, which is exactly the shape whose block-boundary snapshot makes
# the reset's own site push redundant (see snap_boundary_kill_vars). With
# that push present the cell loop reads the previous iteration's `auxu`
# and cgen_ refuses to split it, so this kernel is also the offload
# regression case for the whole class.
#
# i_cell_to_node: 4 x i_ncell connectivity, node index per cell corner
# cell_vol: per-cell volume weight
# i_ncell: number of cells
# i_nnode: number of nodes
# i_njac: number of Jacobi passes
# up: nodal state, updated in place
# mup: nodal accumulator, updated in place
function cellscatter(i_cell_to_node, cell_vol, i_ncell, i_nnode, i_njac, up, mup)
    for i_ = 1:i_njac
        for i_cell = 1:i_ncell
            # gather this cell's four nodal values
            auxu = 0.0
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxu = auxu + up[i_lnode]
            end
            # scatter the volume-weighted contribution back to the same nodes
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                mup[i_lnode] = mup[i_lnode] + (auxu + up[i_lnode]) * cell_vol[i_cell]
            end
        end
        # apply one Jacobi update; sequential in i_
        for i_node = 1:i_nnode
            up[i_node] = up[i_node] - mup[i_node]
        end
    end
    return nothing
end
