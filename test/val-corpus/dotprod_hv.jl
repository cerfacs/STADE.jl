function initstacks_dotprod_b()
    return nothing
end

function dotprod_hv(loss, lossb, u, ub, v, vb, i_n, lossd, lossbd, ud, ubd, vd, vbd)
    for i_x = 1:i_n
        __hcse_0 = v[i_x]
        __hcse_1 = u[i_x]
        lossd[1] = lossd[1] + (__hcse_0 * ud[i_x] + __hcse_1 * vd[i_x])
        loss[1] = loss[1] + __hcse_1 * __hcse_0
    end
    for i_x = i_n:-1:1
        __cse_0d = lossbd[1]
        __cse_0 = lossb[1]
        __hcse_2 = v[i_x]
        ubd[i_x] = ubd[i_x] + (__cse_0 * vd[i_x] + __hcse_2 * __cse_0d)
        ub[i_x] = ub[i_x] + __hcse_2 * __cse_0
        __hcse_3 = u[i_x]
        vbd[i_x] = vbd[i_x] + (__cse_0 * ud[i_x] + __hcse_3 * __cse_0d)
        vb[i_x] = vb[i_x] + __hcse_3 * __cse_0
    end
    return nothing
end

function dotprod(loss, u, v, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] * v[i_x]
    end
end
