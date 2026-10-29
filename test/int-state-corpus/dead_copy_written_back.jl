# dead_copy_written_back(x, w, idx, n, out)
#
# An integer array element copied to a scalar and written back to the same element, in a loop whose other statements
# are float. The copy is dead, and the write must still be replayable in the reverse sweep.
#
# x: input array of length n
# w: input array of length n
# idx: integer array of length n
# n: number of entries
# out: length-1 output array; out[1] receives the sum
function dead_copy_written_back(x, w, idx, n, out)
    s = 0.0
    for i = 1:n
        s += w[i] * x[i]
        jj = idx[i]
        idx[i] = jj
    end
    jt = idx[1]
    s += x[jt] * x[jt]
    out[1] = s
    return nothing
end
