# idm_cse_acc(m, x, i_cell_to_node, loss, i_ncell, i_nnode)
#
# Must offload, with an atomic add. The cell loop scatters into m through a gathered index, written the way CSE leaves a
# hoisted accumulation: the old value is read into `old`, used once, and the element is written back as `old + x`. The gate must
# see through the temporary, or the write stays a plain read-modify-write and two cells sharing a node lose an update.
#
# m: node values, accumulated in place
# x: per-cell input
# i_cell_to_node: 4 x i_ncell node numbers, values in 1:i_nnode
# loss: length-1 output array, accumulated in place
function idm_cse_acc(m, x, i_cell_to_node, loss, i_ncell, i_nnode)
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            old = m[i_node]
            m[i_node] = old + x[i_cell] * 0.5
        end
    end
    for i_n = 1:i_nnode
        loss[1] = loss[1] + m[i_n] * m[i_n]
    end
    return nothing
end
