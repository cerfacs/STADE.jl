# asnap_selfnl(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
#
# The write reads its own array nonlinearly: m[n] = m[n] * (1 + ...). The derivative needs the old value, so the snapshot must be
# refused (guard G4). The primal loop is order-dependent when two cells share a node, so this kernel is also a host-only loop on the GPU.
function asnap_selfnl(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            m[i_node] = 1.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] * (1.0 + w[i_cell] * x[i_n] * 0.05)
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.1
        end
    end
    return nothing
end
