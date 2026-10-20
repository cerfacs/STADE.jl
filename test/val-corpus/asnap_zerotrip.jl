# asnap_zerotrip(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
#
# The cell loop bound is computed in the kernel and is zero or negative for every draw, so the accumulation into m never runs. m is
# not reset, so its input values reach the final read. The snapshot, its restore and the final read must agree with the old code
# when no write to m executes. A loop that may not run has not run.
function asnap_zerotrip(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
    i_nc2 = i_ncell - 8
    for i_ = 1:i_njac
        for i_cell = 1:i_nc2
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
