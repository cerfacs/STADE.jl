# gat_rows(x, y, n, m)
#
# Rows of a two-dimensional array stored flat: the outer loop scatters into the previous and next row, the inner loop is a nest. The rewrite
# handles single loops only, so it must refuse both and leave the atomic scatter.
#
# x: input, length n * m
# y: output, length n * m, interior rows filled in place
# n: number of rows
# m: number of columns
function gat_rows(x, y, n, m)
    for j = 1:m
        y[j] = x[j]
        y[(n - 1) * m + j] = x[(n - 1) * m + j]
    end
    # interior rows from the rows above and below
    for i = 2:n - 1
        for j = 1:m
            y[(i - 1) * m + j] = x[(i - 2) * m + j] + x[i * m + j]
        end
    end
    return nothing
end
