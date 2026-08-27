# entry_dead(x, y, i_cell_to_node, i_ncell, i_nnode, res, out)
#
# The positive case for norm_insert_dead_entry_resets, in the shape that
# motivated it: a cell-to-node scatter whose scratch scalar is assigned
# in every i_loc iteration before use, so the value carried in from the
# previous cell is dead. Nothing in the source says so, and the adjoint's
# pre-write snapshot reads that carried value, which makes the cell loop
# look loop-carried to cgen_use_before_def. The inserted reset states the
# kill and the loop splits. ttgc's second cell nest is the real instance.
#
# x: nodal input of length i_nnode
# y: per-cell input of length i_ncell
# i_cell_to_node: 4 x i_ncell connectivity
# i_ncell: number of cells
# i_nnode: number of nodes
# res: nodal accumulator of length i_nnode, updated in place
# out: nodal output of length i_nnode, updated in place
function entry_dead(x, y, i_cell_to_node, i_ncell, i_nnode, res, out)
    v = 0.0
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            v = x[i_node] * y[i_cell]
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
                res[i_k_node] = res[i_k_node] + v * v
            end
        end
    end
    for i_x = 1:i_nnode
        out[i_x] = out[i_x] + res[i_x] * res[i_x]
    end
end
