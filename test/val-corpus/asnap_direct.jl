# asnap_direct(x, w, node, m, i_njac)
#
# The loop body writes m directly, at one place, with no inner loop. The array snapshot would be correct here, but it copies the whole
# of m to save one slot (guard G10).
#
# x: state, length 1
# w: weight, length 1
# node: node number of the accumulator slot, node[1], 1-based
# m: accumulator, updated in place
function asnap_direct(x, w, node, m, i_njac)
    for i_ = 1:i_njac
        n0 = node[1]
        m[n0] = m[n0] + w[1] * x[1] * 0.1
        x[1] = x[1] + m[n0] * m[n0] * 0.5
    end
    return nothing
end
