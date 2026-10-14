function initstacks_ddp_tangent_input_b()
    return nothing
end

function ddp_tangent_input_hv(loss, lossb, u, ub, a, ab, v, vb, i_n, lossd, lossbd, ud, ubd, ad, abd, vd, vbd)
    for i_x = 1:i_n
        __cse_0d = ud[i_x]
        __cse_0 = u[i_x]
        __hcse_0 = a[i_x]
        __hcse_1 = (__hcse_0 * __cse_0) * __cse_0d
        vd[i_x] = ((__cse_0 * __cse_0) * ad[i_x] + __hcse_1) + __hcse_1
        v[i_x] = __hcse_0 * __cse_0 * __cse_0
    end
    for i_x2 = 1:i_n
        __hcse_2 = u[i_x2]
        __hcse_3 = v[i_x2]
        lossd[1] = lossd[1] + (__hcse_2 * vd[i_x2] + __hcse_3 * ud[i_x2])
        loss[1] = loss[1] + __hcse_3 * __hcse_2
    end
    for i_x2 = i_n:-1:1
        __cse_1d = lossbd[1]
        __cse_1 = lossb[1]
        __hcse_4 = u[i_x2]
        vbd[i_x2] = vbd[i_x2] + (__cse_1 * ud[i_x2] + __hcse_4 * __cse_1d)
        vb[i_x2] = vb[i_x2] + __hcse_4 * __cse_1
        __hcse_5 = v[i_x2]
        ubd[i_x2] = ubd[i_x2] + (__cse_1 * vd[i_x2] + __hcse_5 * __cse_1d)
        ub[i_x2] = ub[i_x2] + __hcse_5 * __cse_1
    end
    for i_x = i_n:-1:1
        __oldb_0d = vbd[i_x]
        __oldb_0 = vb[i_x]
        vbd[i_x] = 0.0
        vb[i_x] = 0.0
        __cse_2d = ud[i_x]
        __cse_2 = u[i_x]
        __hcse_6 = __cse_2 * __cse_2d
        __hcse_7 = __cse_2 * __cse_2
        abd[i_x] = abd[i_x] + (__oldb_0 * (__hcse_6 + __hcse_6) + __hcse_7 * __oldb_0d)
        ab[i_x] = ab[i_x] + __hcse_7 * __oldb_0
        __hcse_8 = a[i_x]
        __hcse_9 = __hcse_8 * __cse_2
        __cse_3d = __oldb_0 * (__cse_2 * ad[i_x] + __hcse_8 * __cse_2d) + __hcse_9 * __oldb_0d
        __cse_3 = __hcse_9 * __oldb_0
        ubd[i_x] = ubd[i_x] + __cse_3d
        ub[i_x] = ub[i_x] + __cse_3
        ubd[i_x] = ubd[i_x] + __cse_3d
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
