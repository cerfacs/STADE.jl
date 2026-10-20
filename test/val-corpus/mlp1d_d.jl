function mlp1d_d(loss, lossd, x, xd, y, yd, w1, w1d, b1, b1d, w2, w2d, b2, b2d, w3, w3d, b3, b3d, h1, h1d, h2, h2d, o, od, n_h)
    for i_j = 1:n_h
        __cse_0 = x[1]
        __cse_1 = w1[i_j]
        s1d = (__cse_0 * w1d[i_j] + __cse_1 * xd[1]) + b1d[i_j]
        s1 = __cse_1 * __cse_0 + b1[i_j]
        __cse_2 = tanh(s1)
        h1d[i_j] = (1.0 - __cse_2 ^ 2) * s1d
        h1[i_j] = __cse_2
    end
    for i_j2 = 1:n_h
        s2d = b2d[i_j2]
        s2 = b2[i_j2]
        for i_k = 1:n_h
            __cse_3 = h1[i_k]
            __cse_4 = w2[(i_j2 - 1) * n_h + i_k]
            s2d = s2d + (__cse_3 * w2d[(i_j2 - 1) * n_h + i_k] + __cse_4 * h1d[i_k])
            s2 = s2 + __cse_4 * __cse_3
        end
        __cse_5 = tanh(s2)
        h2d[i_j2] = (1.0 - __cse_5 ^ 2) * s2d
        h2[i_j2] = __cse_5
    end
    s3d = b3d[1]
    s3 = b3[1]
    for i_k2 = 1:n_h
        __cse_6 = h2[i_k2]
        __cse_7 = w3[i_k2]
        s3d = s3d + (__cse_6 * w3d[i_k2] + __cse_7 * h2d[i_k2])
        s3 = s3 + __cse_7 * __cse_6
    end
    od[1] = s3d
    o[1] = s3
    __cse_8 = o[1] - y[1]
    lossd[1] = lossd[1] + (2__cse_8) * (od[1] + -(yd[1]))
    loss[1] = loss[1] + __cse_8 ^ 2
    return nothing
end

function mlp1d(loss, x, y, w1, b1, w2, b2, w3, b3, h1, h2, o, n_h)
    for i_j = 1:n_h
        s1 = w1[i_j] * x[1] + b1[i_j]
        h1[i_j] = tanh(s1)
    end
    for i_j2 = 1:n_h
        s2 = b2[i_j2]
        for i_k = 1:n_h
            s2 = s2 + w2[(i_j2 - 1) * n_h + i_k] * h1[i_k]
        end
        h2[i_j2] = tanh(s2)
    end
    s3 = b3[1]
    for i_k2 = 1:n_h
        s3 = s3 + w3[i_k2] * h2[i_k2]
    end
    o[1] = s3
    loss[1] = loss[1] + (o[1] - y[1]) ^ 2
end
