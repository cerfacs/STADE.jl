function initstacks_selfrace_layout_b()
    return nothing
end

function selfrace_layout_b(x, xb, w, wb, z, zb, loss, lossb, i_hw, i_c)
    for t = 1:i_hw
        for ci = 1:i_c
            z[(ci - 1) * i_hw + t] = z[(ci - 1) * i_hw + t] + x[(ci - 1) * i_hw + t] * w[ci]
        end
    end
    for m = 1:i_hw * i_c
        __cse_0 = z[m]
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    for m = i_hw * i_c:-1:1
        __cse_1 = z[m] * lossb[1]
        zb[m] = zb[m] + __cse_1
        zb[m] = zb[m] + __cse_1
    end
    for t = i_hw:-1:1
        for ci = i_c:-1:1
            __cse_2 = zb[(ci - 1) * i_hw + t]
            xb[(ci - 1) * i_hw + t] = xb[(ci - 1) * i_hw + t] + w[ci] * __cse_2
            wb[ci] = wb[ci] + x[(ci - 1) * i_hw + t] * __cse_2
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
