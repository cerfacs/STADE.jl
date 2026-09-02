function initstacks_prefixscan_b()
    return nothing
end

function prefixscan_b(loss, lossb, y, yb, x, xb, n)
    y[1] = x[1]
    for i_k = 2:n
        y[i_k] = y[i_k - 1] + x[i_k]
    end
    for i_j = 1:n
        loss[1] = loss[1] + y[i_j]
    end
    for i_j = n:-1:1
        yb[i_j] = yb[i_j] + lossb[1]
    end
    for i_k = n:-1:2
        __oldb_0 = yb[i_k]
        yb[i_k] = 0.0
        yb[i_k - 1] = yb[i_k - 1] + __oldb_0
        xb[i_k] = xb[i_k] + __oldb_0
    end
    __oldb_0 = yb[1]
    yb[1] = 0.0
    xb[1] = xb[1] + __oldb_0
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
