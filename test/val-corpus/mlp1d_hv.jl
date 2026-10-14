function initstacks_mlp1d_b(n_h)
    return nothing
end

function mlp1d_hv(loss, lossb, x, xb, y, yb, w1, w1b, b1, b1b, w2, w2b, b2, b2b, w3, w3b, b3, b3b, h1, h1b, h2, h2b, o, ob, n_h, lossd, lossbd, xd, xbd, yd, ybd, w1d, w1bd, b1d, b1bd, w2d, w2bd, b2d, b2bd, w3d, w3bd, b3d, b3bd, h1d, h1bd, h2d, h2bd, od, obd)
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s1b = 0.0
    s2b = 0.0
    s3b = 0.0
    s1d = 0.0
    s1bd = 0.0
    s2d = 0.0
    s2bd = 0.0
    s3d = 0.0
    s3bd = 0.0
    for i_j = 1:n_h
        __hcse_0 = x[1]
        __hcse_1 = w1[i_j]
        s1d = (__hcse_0 * w1d[i_j] + __hcse_1 * xd[1]) + b1d[i_j]
        s1 = __hcse_1 * __hcse_0 + b1[i_j]
        __hcse_2 = tanh(s1)
        h1d[i_j] = (1.0 - __hcse_2 ^ 2) * s1d
        h1[i_j] = __hcse_2
    end
    for i_j2 = 1:n_h
        s2d = b2d[i_j2]
        s2 = b2[i_j2]
        for i_k = 1:n_h
            __hcse_3 = h1[i_k]
            __hcse_4 = w2[(i_j2 - 1) * n_h + i_k]
            s2d = s2d + (__hcse_3 * w2d[(i_j2 - 1) * n_h + i_k] + __hcse_4 * h1d[i_k])
            s2 = s2 + __hcse_4 * __hcse_3
        end
        __hcse_5 = tanh(s2)
        h2d[i_j2] = (1.0 - __hcse_5 ^ 2) * s2d
        h2[i_j2] = __hcse_5
    end
    s3d = b3d[1]
    s3 = b3[1]
    for i_k2 = 1:n_h
        __hcse_6 = h2[i_k2]
        __hcse_7 = w3[i_k2]
        s3d = s3d + (__hcse_6 * w3d[i_k2] + __hcse_7 * h2d[i_k2])
        s3 = s3 + __hcse_7 * __hcse_6
    end
    od[1] = s3d
    o[1] = s3
    __hcse_8 = o[1] - y[1]
    __hcse_9 = 2__hcse_8
    __hcse_10 = od[1] + -(yd[1])
    lossd[1] = lossd[1] + __hcse_9 * __hcse_10
    loss[1] = loss[1] + __hcse_8 ^ 2
    __hcse_11 = lossb[1]
    __cse_0d = __hcse_11 * (2__hcse_10) + __hcse_9 * lossbd[1]
    __cse_0 = __hcse_9 * __hcse_11
    obd[1] = obd[1] + __cse_0d
    ob[1] = ob[1] + __cse_0
    ybd[1] = ybd[1] + -__cse_0d
    yb[1] = yb[1] + -__cse_0
    __oldb_0d = obd[1]
    __oldb_0 = ob[1]
    obd[1] = 0.0
    ob[1] = 0.0
    s3bd = s3bd + __oldb_0d
    s3b = s3b + __oldb_0
    for i_k2 = n_h:-1:1
        __hcse_12 = h2[i_k2]
        w3bd[i_k2] = w3bd[i_k2] + (s3b * h2d[i_k2] + __hcse_12 * s3bd)
        w3b[i_k2] = w3b[i_k2] + __hcse_12 * s3b
        __hcse_13 = w3[i_k2]
        h2bd[i_k2] = h2bd[i_k2] + (s3b * w3d[i_k2] + __hcse_13 * s3bd)
        h2b[i_k2] = h2b[i_k2] + __hcse_13 * s3b
    end
    __oldb_0d = s3bd
    __oldb_0 = s3b
    s3bd = 0.0
    s3b = 0.0
    b3bd[1] = b3bd[1] + __oldb_0d
    b3b[1] = b3b[1] + __oldb_0
    for i_j2 = 1:n_h
        s2d = b2d[i_j2]
        s2 = b2[i_j2]
        for i_k = 1:n_h
            __hcse_14 = h1[i_k]
            __hcse_15 = w2[(i_j2 - 1) * n_h + i_k]
            s2d = s2d + (__hcse_14 * w2d[(i_j2 - 1) * n_h + i_k] + __hcse_15 * h1d[i_k])
            s2 = s2 + __hcse_15 * __hcse_14
        end
        __oldb_0d = h2bd[i_j2]
        __oldb_0 = h2b[i_j2]
        h2bd[i_j2] = 0.0
        h2b[i_j2] = 0.0
        __hcse_16 = tanh(s2)
        __hcse_17 = 1.0 - __hcse_16 ^ 2
        s2bd = s2bd + (__oldb_0 * -((2__hcse_16) * (__hcse_17 * s2d)) + __hcse_17 * __oldb_0d)
        s2b = s2b + __hcse_17 * __oldb_0
        for i_k = 1:n_h
            __cse_1d = w2d[(i_j2 - 1) * n_h + i_k]
            __cse_1 = w2[(i_j2 - 1) * n_h + i_k]
            __cse_2d = h1d[i_k]
            __cse_2 = h1[i_k]
            s2d = s2d + (__cse_2 * __cse_1d + __cse_1 * __cse_2d)
            s2 = s2 + __cse_1 * __cse_2
            w2bd[(i_j2 - 1) * n_h + i_k] = w2bd[(i_j2 - 1) * n_h + i_k] + (s2b * __cse_2d + __cse_2 * s2bd)
            w2b[(i_j2 - 1) * n_h + i_k] = w2b[(i_j2 - 1) * n_h + i_k] + __cse_2 * s2b
            h1bd[i_k] = h1bd[i_k] + (s2b * __cse_1d + __cse_1 * s2bd)
            h1b[i_k] = h1b[i_k] + __cse_1 * s2b
        end
        __oldb_0d = s2bd
        __oldb_0 = s2b
        s2bd = 0.0
        s2b = 0.0
        b2bd[i_j2] = b2bd[i_j2] + __oldb_0d
        b2b[i_j2] = b2b[i_j2] + __oldb_0
    end
    for i_j = 1:n_h
        __cse_3d = w1d[i_j]
        __cse_3 = w1[i_j]
        __cse_4d = xd[1]
        __cse_4 = x[1]
        s1d = (__cse_4 * __cse_3d + __cse_3 * __cse_4d) + b1d[i_j]
        s1 = __cse_3 * __cse_4 + b1[i_j]
        __oldb_0d = h1bd[i_j]
        __oldb_0 = h1b[i_j]
        h1bd[i_j] = 0.0
        h1b[i_j] = 0.0
        __hcse_18 = tanh(s1)
        __hcse_19 = 1.0 - __hcse_18 ^ 2
        s1bd = s1bd + (__oldb_0 * -((2__hcse_18) * (__hcse_19 * s1d)) + __hcse_19 * __oldb_0d)
        s1b = s1b + __hcse_19 * __oldb_0
        __oldb_0d = s1bd
        __oldb_0 = s1b
        s1bd = 0.0
        s1b = 0.0
        w1bd[i_j] = w1bd[i_j] + (__oldb_0 * __cse_4d + __cse_4 * __oldb_0d)
        w1b[i_j] = w1b[i_j] + __cse_4 * __oldb_0
        xbd[1] = xbd[1] + (__oldb_0 * __cse_3d + __cse_3 * __oldb_0d)
        xb[1] = xb[1] + __cse_3 * __oldb_0
        b1bd[i_j] = b1bd[i_j] + __oldb_0d
        b1b[i_j] = b1b[i_j] + __oldb_0
    end
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
