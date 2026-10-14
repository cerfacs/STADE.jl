function initstacks_red_three_arrays_b()
    return nothing
end

function red_three_arrays_b(loss, lossb, u, ub, v, vb, w, wb, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + (u[i_x] * v[i_x]) * w[i_x]
    end
    for i_x = i_n:-1:1
        __cse_0 = v[i_x]
        __cse_1 = lossb[1]
        __cse_2 = w[i_x] * __cse_1
        ub[i_x] = ub[i_x] + __cse_0 * __cse_2
        __cse_3 = u[i_x]
        vb[i_x] = vb[i_x] + __cse_3 * __cse_2
        wb[i_x] = wb[i_x] + (__cse_3 * __cse_0) * __cse_1
    end
    return nothing
end

function red_three_arrays(loss, u, v, w, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + (u[i_x] * v[i_x]) * w[i_x]
    end
end
