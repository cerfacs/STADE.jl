# prefixscan(loss, y, x, n)
#
# Inclusive prefix scan of x into y, then a plain linear loss over the scan.
# The scan's cross-iteration coupling runs through an array and is purely
# additive, so no snapshot of y is ever pushed. The backward sweep's loop
# direction is the only thing that makes this gradient right.
#
# loss: length-1 output array; loss[1] receives the result
# y: scan output array of length n
# x: input array of length n
# n: number of points in x and y
function prefixscan(loss, y, x, n)
    y[1] = x[1]
    # running scan -- iteration k reads what iteration k-1 wrote
    for i_k = 2:n
        y[i_k] = y[i_k - 1] + x[i_k]
    end
    # running sum of the scan
    for i_j = 1:n
        loss[1] = loss[1] + y[i_j]
    end
    return nothing
end
