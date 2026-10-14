function initstacks_ddp_tangent_input_b()
    return nothing
end

function ddp_tangent_input_b(loss, lossb, u, ub, a, ab, v, vb, i_n)
    for i_x = 1:i_n
        __cse_0 = u[i_x]
        v[i_x] = a[i_x] * __cse_0 * __cse_0
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] * u[i_x2]
    end
    for i_x2 = i_n:-1:1
        __cse_1 = lossb[1]
        vb[i_x2] = vb[i_x2] + u[i_x2] * __cse_1
        ub[i_x2] = ub[i_x2] + v[i_x2] * __cse_1
    end
    for i_x = i_n:-1:1
        __oldb_0 = vb[i_x]
        vb[i_x] = 0.0
        __cse_2 = u[i_x]
        ab[i_x] = ab[i_x] + (__cse_2 * __cse_2) * __oldb_0
        __cse_3 = (a[i_x] * __cse_2) * __oldb_0
        ub[i_x] = ub[i_x] + __cse_3
        ub[i_x] = ub[i_x] + __cse_3
    end
    return nothing
end

function ddp_tangent_input(loss, u, a, v, i_n)
    for i_x = 1:i_n
        v[i_x] = a[i_x] * u[i_x] * u[i_x]
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] * u[i_x2]
    end
end
