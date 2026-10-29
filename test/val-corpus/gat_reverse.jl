# gat_reverse(x, y, n)
#
# A descending primal loop. Its adjoint runs ascending, so the rewrite sees both orientations across the corpus.
#
# x: input, length n
# y: output, length n, interior filled in place
# n: number of points
function gat_reverse(x, y, n)
    y[1] = x[1]
    y[n] = x[n]
    # interior, written from the right
    for i = n - 1:-1:2
        y[i] = x[i - 1] * x[i + 1] + x[i]
    end
    return nothing
end
