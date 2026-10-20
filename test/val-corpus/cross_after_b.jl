function initstacks_cross_after_b(i_n, i_nc)
    xv_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1))
    return xv_stack
end

function cross_after_b(x, xb, w, wb, z, zb, loss, lossb, i_n, i_nc, xv_stack)
    a = 0.0
    xv = 0.0
    ab = 0.0
    xvb = 0.0
    for t = 1:i_n
        a = 0.0
        xv = 0.0
        for ci = 1:i_nc
            __idx_xv_stack_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
            xv_stack[__idx_xv_stack_0] = xv
            xv = x[(t + ci) - 1]
            a = a + xv * xv * w[ci]
        end
        a = a + xv * xv
        z[t] = a
        __idx_xv_stack_5 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + ((t - 1) + 1)
        xv_stack[__idx_xv_stack_5] = xv
    end
    for m = 1:i_n
        __cse_0 = z[m]
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    for m = i_n:-1:1
        __cse_1 = z[m] * lossb[1]
        zb[m] = zb[m] + __cse_1
        zb[m] = zb[m] + __cse_1
    end
    for t = i_n:-1:1
        __idx_xv_stack_0 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + ((t - 1) + 1)
        xv = xv_stack[__idx_xv_stack_0]
        __oldb_0 = zb[t]
        zb[t] = 0.0
        ab = ab + __oldb_0
        __cse_2 = xv * ab
        xvb = xvb + __cse_2
        xvb = xvb + __cse_2
        for ci = i_nc:-1:1
            __cse_3 = (xv * w[ci]) * ab
            xvb = xvb + __cse_3
            xvb = xvb + __cse_3
            wb[ci] = wb[ci] + (xv * xv) * ab
            __idx_xv_stack_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
            xv = xv_stack[__idx_xv_stack_0]
            __oldb_2 = xvb
            xvb = 0.0
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + __oldb_2
        end
        xvb = 0.0
        ab = 0.0
    end
    return nothing
end

function cross_after(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        xv = 0.0
        for ci = 1:i_nc
            xv = x[(t + ci) - 1]
            a = a + xv * xv * w[ci]
        end
        a = a + xv * xv
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
