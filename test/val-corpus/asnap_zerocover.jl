# asnap_zerocover(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
#
# The reset loop bound is computed in the kernel and is zero or negative for every draw, so the reset never runs. The accumulation
# runs and starts from the stale values of m. A rule that treats the reset loop as a total cover of m is wrong here. The snapshot
# copies the whole array and does not depend on any cover.
function asnap_zerocover(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
    i_nz = i_nnode - 8
    for i_ = 1:i_njac
        for i_node = 1:i_nz
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
