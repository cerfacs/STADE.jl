# asnap_noscatter(x, w, m, i_nnode, i_njac)
#
# The shape of asnap_ok without its indirect accumulation: m is zeroed, accumulated at the loop's own index, and read in a nonlinear term after
# every write. Every other guard accepts the snapshot. The accumulation is not a scatter (no integer array supplies its index), so no thread
# can share an element and the snapshot buys nothing. It must be refused (guard G14).
#
# x: nodal state, updated in place
# w: per-node weight
# m: nodal accumulator, reset and updated in place
function asnap_noscatter(x, w, m, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            m[i_node] = 0.0
        end
        for i_node = 1:i_nnode
            m[i_node] = m[i_node] + w[i_node] * x[i_node] * 0.1
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.5
        end
    end
    return nothing
end
