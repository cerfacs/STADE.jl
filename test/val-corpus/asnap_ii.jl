# asnap_ii(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
#
# The cell loop gathers a local scalar from w, which the loop never writes, and scatters a nonlinear term of it into m. With
# fuse_ii_loops the cell loop is classified. The write to m then sits in a classified loop, and version 1 of the array snapshot
# refuses the body (guard G7).
function asnap_ii(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            m[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            auxu = 0.0
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                auxu = auxu + w[i_n]
            end
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] + auxu * auxu * 0.1
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.5
        end
    end
    return nothing
end
