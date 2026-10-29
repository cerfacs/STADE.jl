# gat_alias(x, y, n)
#
# The neighbour index goes through a scalar (ip = i + 1), so the adjoint scatters at a symbol that is not `i` plus a literal. The rewrite
# leaves it as an atomic scatter.
#
# x: input, length n
# y: output, length n, interior filled in place
# n: number of points
function gat_alias(x, y, n)
    y[1] = x[1]
    y[n] = x[n]
    # interior, neighbours by index scalars
    for i = 2:n - 1
        ip = i + 1
        im = i - 1
        y[i] = x[im] + x[ip] - 2.0 * x[i]
    end
    return nothing
end
