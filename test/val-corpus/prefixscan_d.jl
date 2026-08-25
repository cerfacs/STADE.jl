function prefixscan_d(loss, lossd, y, yd, x, xd, n)
    yd[1] = xd[1]
    y[1] = x[1]
    for i_k = 2:n
        yd[i_k] = yd[i_k - 1] + xd[i_k]
        y[i_k] = y[i_k - 1] + x[i_k]
    end
    for i_j = 1:n
        lossd[1] = lossd[1] + yd[i_j]
        loss[1] = loss[1] + y[i_j]
    end
    return nothing
end

function prefixscan(loss, y, x, n)
    y[1] = x[1]
    for i_k = 2:n
        y[i_k] = y[i_k - 1] + x[i_k]
    end
    for i_j = 1:n
        loss[1] = loss[1] + y[i_j]
    end
end
