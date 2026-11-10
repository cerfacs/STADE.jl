# f1_stencil
# Three-point stencil, writes at i1 only, reads two neighbours.
# LOOP i1 I K
function f1_stencil(x, y, n)
    for i1 = 2:n - 1
        y[i1] = x[i1 - 1] + x[i1] + x[i1 + 1]
    end
    return nothing
end
