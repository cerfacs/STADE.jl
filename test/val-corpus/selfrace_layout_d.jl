function selfrace_layout_d(x, xd, w, wd, z, zd, loss, lossd, i_hw, i_c)
    for t = 1:i_hw
        for ci = 1:i_c
            __cse_0 = w[ci]
            __cse_1 = x[(ci - 1) * i_hw + t]
            zd[(ci - 1) * i_hw + t] = zd[(ci - 1) * i_hw + t] + (__cse_0 * xd[(ci - 1) * i_hw + t] + __cse_1 * wd[ci])
            z[(ci - 1) * i_hw + t] = z[(ci - 1) * i_hw + t] + __cse_1 * __cse_0
        end
    end
    for m = 1:i_hw * i_c
        __cse_2 = z[m]
        __cse_3 = __cse_2 * zd[m]
        lossd[1] = lossd[1] + (__cse_3 + __cse_3)
        loss[1] = loss[1] + __cse_2 * __cse_2
    end
    return nothing
end

function selfrace_layout(x, w, z, loss, i_hw, i_c)
    for t = 1:i_hw
        for ci = 1:i_c
            z[(ci - 1) * i_hw + t] = z[(ci - 1) * i_hw + t] + x[(ci - 1) * i_hw + t] * w[ci]
        end
    end
    for m = 1:i_hw * i_c
        loss[1] = loss[1] + z[m] * z[m]
    end
end
