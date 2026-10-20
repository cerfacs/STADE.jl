function initstacks_selfrace_layout_b()
    return nothing
end

function selfrace_layout_hv(x, xb, w, wb, z, zb, loss, lossb, i_hw, i_c, xd, xbd, wd, wbd, zd, zbd, lossd, lossbd)
    for t = 1:i_hw
        for ci = 1:i_c
            __hcse_0 = w[ci]
            __hcse_1 = x[(ci - 1) * i_hw + t]
            zd[(ci - 1) * i_hw + t] = zd[(ci - 1) * i_hw + t] + (__hcse_0 * xd[(ci - 1) * i_hw + t] + __hcse_1 * wd[ci])
            z[(ci - 1) * i_hw + t] = z[(ci - 1) * i_hw + t] + __hcse_1 * __hcse_0
        end
    end
    for m = 1:i_hw * i_c
        __cse_0d = zd[m]
        __cse_0 = z[m]
        __hcse_2 = __cse_0 * __cse_0d
        lossd[1] = lossd[1] + (__hcse_2 + __hcse_2)
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    for m = i_hw * i_c:-1:1
        __hcse_3 = lossb[1]
        __hcse_4 = z[m]
        __cse_1d = __hcse_3 * zd[m] + __hcse_4 * lossbd[1]
        __cse_1 = __hcse_4 * __hcse_3
        zbd[m] = zbd[m] + __cse_1d
        zb[m] = zb[m] + __cse_1
        zbd[m] = zbd[m] + __cse_1d
        zb[m] = zb[m] + __cse_1
    end
    for t = i_hw:-1:1
        for ci = i_c:-1:1
            __cse_2d = zbd[(ci - 1) * i_hw + t]
            __cse_2 = zb[(ci - 1) * i_hw + t]
            __hcse_5 = w[ci]
            xbd[(ci - 1) * i_hw + t] = xbd[(ci - 1) * i_hw + t] + (__cse_2 * wd[ci] + __hcse_5 * __cse_2d)
            xb[(ci - 1) * i_hw + t] = xb[(ci - 1) * i_hw + t] + __hcse_5 * __cse_2
            __hcse_6 = x[(ci - 1) * i_hw + t]
            wbd[ci] = wbd[ci] + (__cse_2 * xd[(ci - 1) * i_hw + t] + __hcse_6 * __cse_2d)
            wb[ci] = wb[ci] + __hcse_6 * __cse_2
        end
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
