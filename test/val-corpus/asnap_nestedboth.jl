# asnap_nestedboth(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac, i_nout)
#
# A loop nest where both loops meet every guard for m: the inner sweeps only write m, and the read of m comes after them in the outer
# body. The outer loop owns the snapshot (guard G6). One snapshot of m only, never two nested ones.
function asnap_nestedboth(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac, i_nout)
    for i_o = 1:i_nout
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
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.2
        end
    end
    return nothing
end
