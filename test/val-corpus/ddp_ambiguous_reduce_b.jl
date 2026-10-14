function initstacks_ddp_ambiguous_reduce_b()
    return nothing
end

function ddp_ambiguous_reduce_b(loss, lossb, u, ub, a, ab, v, vb, i_n)
    for i_x = 1:i_n
        v[i_x] = v[i_x] + a[i_x] * u[i_x]
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
    for i_x2 = i_n:-1:1
        vb[i_x2] = vb[i_x2] + (2 * v[i_x2]) * lossb[1]
    end
    for i_x = i_n:-1:1
        __cse_0 = vb[i_x]
        ab[i_x] = ab[i_x] + u[i_x] * __cse_0
        ub[i_x] = ub[i_x] + a[i_x] * __cse_0
    end
    return nothing
end

function ddp_ambiguous_reduce(loss, u, a, v, i_n)
    for i_x = 1:i_n
        v[i_x] = v[i_x] + a[i_x] * u[i_x]
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
end
