# gat_inplace_seq(u, n)
#
# Gauss-Seidel style in-place sweep: iteration i reads u[i - 1], which iteration i - 1 has just written. The loop is order-dependent, so the
# adjoint must stay sequential and the gather rewrite must refuse it (the receiving array is also read).
#
# u: field, length n, updated in place
# n: number of points
function gat_inplace_seq(u, n)
    for i = 2:n - 1
        u[i] = 0.5 * u[i] + 0.25 * (u[i - 1] + u[i + 1])
    end
    return nothing
end
