# scalar_two_sites(x, idx, n, out)
#
# An integer scalar assigned twice in a loop body, with reads between the two assignments. The reverse sweep re-runs
# the assignments up front, so it would give every read the second value. The scalar must be restored.
#
# x: input array of length 6
# idx: integer array of length 6, never written
# n: number of iterations
# out: length-1 output array; out[1] receives the sum
function scalar_two_sites(x, idx, n, out)
    s = 0.0
    for i = 1:n
        jj = idx[i]
        j = idx[jj]
        s += x[jj] * x[jj]
        jj = idx[j]
        s += x[jj] * x[i]
    end
    out[1] = s
    return nothing
end
