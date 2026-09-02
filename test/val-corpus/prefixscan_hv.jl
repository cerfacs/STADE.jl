function initstacks_prefixscan_b()
    return nothing
end

function prefixscan_hv(loss, lossb, y, yb, x, xb, n, lossd, lossbd, yd, ybd, xd, xbd)
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
    for i_j = n:-1:1
        ybd[i_j] = ybd[i_j] + lossbd[1]
        yb[i_j] = yb[i_j] + lossb[1]
    end
    for i_k = n:-1:2
        __oldb_0d = ybd[i_k]
        __oldb_0 = yb[i_k]
        ybd[i_k] = 0.0
        yb[i_k] = 0.0
        ybd[i_k - 1] = ybd[i_k - 1] + __oldb_0d
        yb[i_k - 1] = yb[i_k - 1] + __oldb_0
        xbd[i_k] = xbd[i_k] + __oldb_0d
        xb[i_k] = xb[i_k] + __oldb_0
    end
    __oldb_0d = ybd[1]
    __oldb_0 = yb[1]
    ybd[1] = 0.0
    yb[1] = 0.0
    xbd[1] = xbd[1] + __oldb_0d
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
