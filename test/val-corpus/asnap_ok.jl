# asnap_ok(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
#
# The reduced ttgc shape. Inside the Jacobi loop, m is zeroed, filled by an indirect accumulation, and read in a nonlinear term after
# every write. m is a final-read array of the loop, so the array snapshot applies to m. x is read nonlinearly in the cell loop before
# the node loop writes it, so the snapshot must NOT apply to x.
#
# x: nodal state, updated in place
# w: per-cell weight
# i_cell_to_node: 3 x i_ncell node numbers
# m: nodal accumulator, reset and updated in place
function asnap_ok(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
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
    return nothing
end
