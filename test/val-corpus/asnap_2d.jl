# asnap_2d(x, w, i_cell_to_node, M, i_ncell, i_nnode, i_njac)
#
# The ttgc shape with a two-dimensional accumulator. Zero M, scatter an indirect accumulation into M[q, node], then read M in a
# nonlinear term after every write. M is a final-read array of the Jacobi loop, so the array snapshot applies. The copy loop must
# use one linear index on a 2-D array while the kernel loops use M[q, node].
#
# x: nodal state, updated in place
# w: per-cell weight
# i_cell_to_node: 3 x i_ncell node numbers
# M: 2 x nodes accumulator, reset and updated in place
function asnap_2d(x, w, i_cell_to_node, M, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            for q = 1:2
                M[q, i_node] = 0.0
            end
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                for q = 1:2
                    M[q, i_n] = M[q, i_n] + w[i_cell] * x[i_n] * 0.1
                end
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + M[1, i_node] * M[2, i_node] * 0.5
        end
    end
    return nothing
end
