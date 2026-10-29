# gat_stride(x, y, m)
#
# Restriction by two: the adjoint scatters at 2 * i - 1 and 2 * i, not at i plus a constant. The rewrite must refuse it.
#
# x: input, length 2 * m
# y: output, length m
# m: number of coarse points
function gat_stride(x, y, m)
    for i = 1:m
        y[i] = 0.5 * x[2 * i - 1] + 0.5 * x[2 * i]
    end
    return nothing
end
