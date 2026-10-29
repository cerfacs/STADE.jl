# gat_const_idx(x, y, n)
#
# A loop that also scatters at a fixed index: the sum of its inputs goes to y[1] as well. The receiving array is accumulated at an index that is
# not `i` plus a literal, so the rewrite must refuse the loop.
#
# x: input, length n
# y: output, length n
# n: number of points
function gat_const_idx(x, y, n)
    for i = 2:n - 1
        y[i] = x[i - 1] + x[i + 1]
        y[1] = y[1] + x[i]
    end
    return nothing
end
