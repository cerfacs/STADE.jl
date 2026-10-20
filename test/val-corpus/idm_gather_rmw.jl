# idm_gather_rmw(m, x, i_cell_to_node, loss, i_ncell, i_nnode)
#
# Must stay on the host. Each cell scales and increments the node values it touches, `m[n] = m[n] * 0.5 + x[c]`. The read and the
# write use the same gathered index, so the strict array rule accepts the loop, but two cells that share a node run in sequence and
# the result depends on their order. No atomic add repairs a scaled update. The offload report must show it refused for `m`.
#
# m: node values, updated in place
# x: per-cell input
# i_cell_to_node: 4 x i_ncell node numbers, values in 1:i_nnode
# loss: length-1 output array, accumulated in place
function idm_gather_rmw(m, x, i_cell_to_node, loss, i_ncell, i_nnode)
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            m[i_node] = m[i_node] * 0.5 + x[i_cell]
        end
    end
    for i_n = 1:i_nnode
        loss[1] = loss[1] + m[i_n] * m[i_n]
    end
    return nothing
end
