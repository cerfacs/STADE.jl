function initstacks_cse_branch_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1))
    return branch_stack
end

function cse_branch_b(out, outb, u, ub, v, vb, i_flag, i_n, branch_stack)
    s = 0.0
    sb = 0.0
    for i_r = 1:i_n
        s = 0.0
        if i_flag[i_r] > 0
            branch_stack[1] = 1
            __cse_0 = u[i_r]
            __cse_1 = v[i_r]
            s = __cse_0 * __cse_1 + __cse_0 * __cse_1 * __cse_1
        else
            branch_stack[1] = 0
            __cse_2 = u[i_r]
            __cse_3 = v[i_r]
            s = __cse_2 * __cse_3 - __cse_2 * __cse_3 * __cse_2
        end
        __cse_4 = u[i_r]
        __cse_5 = v[i_r]
        __cse_6 = __cse_4 * __cse_5
        out[i_r] = s * __cse_6
        __oldb_0 = outb[i_r]
        outb[i_r] = 0.0
        sb = sb + __cse_6 * __oldb_0
        __cse_7 = s * __oldb_0
        ub[i_r] = ub[i_r] + __cse_5 * __cse_7
        vb[i_r] = vb[i_r] + __cse_4 * __cse_7
        __branch = branch_stack[1]
        if __branch == 1
            __oldb_0 = sb
            sb = 0.0
            __cse_8 = v[i_r]
            ub[i_r] = ub[i_r] + __cse_8 * __oldb_0
            __cse_9 = u[i_r]
            vb[i_r] = vb[i_r] + __cse_9 * __oldb_0
            ub[i_r] = ub[i_r] + (__cse_8 * __cse_8) * __oldb_0
            __cse_10 = (__cse_9 * __cse_8) * __oldb_0
            vb[i_r] = vb[i_r] + __cse_10
            vb[i_r] = vb[i_r] + __cse_10
        else
            __oldb_0 = sb
            sb = 0.0
            __cse_11 = v[i_r]
            ub[i_r] = ub[i_r] + __cse_11 * __oldb_0
            __cse_12 = u[i_r]
            vb[i_r] = vb[i_r] + __cse_12 * __oldb_0
            __cse_13 = -__oldb_0
            ub[i_r] = ub[i_r] + (__cse_11 * __cse_12) * __cse_13
            vb[i_r] = vb[i_r] + (__cse_12 * __cse_12) * __cse_13
            ub[i_r] = ub[i_r] + (__cse_12 * __cse_11) * __cse_13
        end
        sb = 0.0
    end
    return nothing
end

function cse_branch(out, u, v, i_flag, i_n)
    for i_r = 1:i_n
        s = 0.0
        if i_flag[i_r] > 0
            s = u[i_r] * v[i_r] + u[i_r] * v[i_r] * v[i_r]
        else
            s = u[i_r] * v[i_r] - u[i_r] * v[i_r] * u[i_r]
        end
        out[i_r] = s * (u[i_r] * v[i_r])
    end
end
