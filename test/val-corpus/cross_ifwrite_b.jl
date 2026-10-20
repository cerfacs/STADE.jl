function initstacks_cross_ifwrite_b(i_n, i_nc)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1))
    xv_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1))
    return (branch_stack, xv_stack)
end

function cross_ifwrite_b(x, xb, w, wb, z, zb, loss, lossb, i_n, i_nc, branch_stack, xv_stack)
    a = 0.0
    xv = 0.0
    ab = 0.0
    xvb = 0.0
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv = x[(t + ci) - 1]
            a = a + xv * xv * w[ci]
            if x[t + ci] > 0.5
                __icse_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
                __idx_branch_stack_0 = __icse_0
                branch_stack[__idx_branch_stack_0] = 1
                __idx_xv_stack_0 = __icse_0
                xv_stack[__idx_xv_stack_0] = xv
                xv = xv * 2.0
            else
                __idx_branch_stack_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
                branch_stack[__idx_branch_stack_0] = 0
            end
            a = a + xv * xv
            __icse_1 = div(i_nc - 1, 1) + 1
            __idx_xv_stack_4 = max(0, div(i_n - 1, 1) + 1) * max(0, __icse_1) + (((t - 1) * __icse_1 + (ci - 1)) + 1)
            xv_stack[__idx_xv_stack_4] = xv
        end
        z[t] = a
    end
    for m = 1:i_n
        __cse_2 = z[m]
        loss[1] = loss[1] + __cse_2 * __cse_2
    end
    for m = i_n:-1:1
        __cse_3 = z[m] * lossb[1]
        zb[m] = zb[m] + __cse_3
        zb[m] = zb[m] + __cse_3
    end
    for t = i_n:-1:1
        __oldb_0 = zb[t]
        zb[t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            __icse_4 = div(i_nc - 1, 1) + 1
            __icse_5 = ((t - 1) * __icse_4 + (ci - 1)) + 1
            __idx_xv_stack_0 = max(0, div(i_n - 1, 1) + 1) * max(0, __icse_4) + __icse_5
            xv = xv_stack[__idx_xv_stack_0]
            __cse_6 = xv * ab
            xvb = xvb + __cse_6
            xvb = xvb + __cse_6
            __idx_branch_stack_4 = __icse_5
            __branch = branch_stack[__idx_branch_stack_4]
            if __branch == 1
                __idx_xv_stack_0 = ((t - 1) * (div(i_nc - 1, 1) + 1) + (ci - 1)) + 1
                xv = xv_stack[__idx_xv_stack_0]
                __oldb_2 = xvb
                xvb = 0.0
                xvb = xvb + 2.0__oldb_2
            end
            __cse_7 = (xv * w[ci]) * ab
            xvb = xvb + __cse_7
            xvb = xvb + __cse_7
            wb[ci] = wb[ci] + (xv * xv) * ab
            __oldb_0 = xvb
            xvb = 0.0
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + __oldb_0
        end
        ab = 0.0
    end
    return nothing
end

function cross_ifwrite(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv = x[(t + ci) - 1]
            a = a + xv * xv * w[ci]
            if x[t + ci] > 0.5
                xv = xv * 2.0
            end
            a = a + xv * xv
        end
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
