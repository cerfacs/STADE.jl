function initstacks_cse_branch_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1))
    return branch_stack
end

function cse_branch_hv(out, outb, u, ub, v, vb, i_flag, i_n, outd, outbd, ud, ubd, vd, vbd, branch_stack)
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    for i_r = 1:i_n
        sd = 0.0
        s = 0.0
        if i_flag[i_r] > 0
            branch_stack[1] = 1
            __cse_0d = ud[i_r]
            __cse_0 = u[i_r]
            __cse_1d = vd[i_r]
            __cse_1 = v[i_r]
            __hcse_0 = __cse_0 * __cse_1
            __hcse_1 = __hcse_0 * __cse_1d
            sd = (__cse_1 * __cse_0d + __cse_0 * __cse_1d) + (((__cse_1 * __cse_1) * __cse_0d + __hcse_1) + __hcse_1)
            s = __hcse_0 + __cse_0 * __cse_1 * __cse_1
        else
            branch_stack[1] = 0
            __cse_2d = ud[i_r]
            __cse_2 = u[i_r]
            __cse_3d = vd[i_r]
            __cse_3 = v[i_r]
            __hcse_2 = __cse_2 * __cse_3
            sd = (__cse_3 * __cse_2d + __cse_2 * __cse_3d) + -((((__cse_3 * __cse_2) * __cse_2d + (__cse_2 * __cse_2) * __cse_3d) + __hcse_2 * __cse_2d))
            s = __hcse_2 - __cse_2 * __cse_3 * __cse_2
        end
        __cse_4d = ud[i_r]
        __cse_4 = u[i_r]
        __cse_5d = vd[i_r]
        __cse_5 = v[i_r]
        __cse_6d = __cse_5 * __cse_4d + __cse_4 * __cse_5d
        __cse_6 = __cse_4 * __cse_5
        outd[i_r] = __cse_6 * sd + s * __cse_6d
        out[i_r] = s * __cse_6
        __oldb_0d = outbd[i_r]
        __oldb_0 = outb[i_r]
        outbd[i_r] = 0.0
        outb[i_r] = 0.0
        sbd = sbd + (__oldb_0 * __cse_6d + __cse_6 * __oldb_0d)
        sb = sb + __cse_6 * __oldb_0
        __cse_7d = __oldb_0 * sd + s * __oldb_0d
        __cse_7 = s * __oldb_0
        ubd[i_r] = ubd[i_r] + (__cse_7 * __cse_5d + __cse_5 * __cse_7d)
        ub[i_r] = ub[i_r] + __cse_5 * __cse_7
        vbd[i_r] = vbd[i_r] + (__cse_7 * __cse_4d + __cse_4 * __cse_7d)
        vb[i_r] = vb[i_r] + __cse_4 * __cse_7
        __branch = branch_stack[1]
        if __branch == 1
            __oldb_0d = sbd
            __oldb_0 = sb
            sbd = 0.0
            sb = 0.0
            __cse_8d = vd[i_r]
            __cse_8 = v[i_r]
            ubd[i_r] = ubd[i_r] + (__oldb_0 * __cse_8d + __cse_8 * __oldb_0d)
            ub[i_r] = ub[i_r] + __cse_8 * __oldb_0
            __cse_9d = ud[i_r]
            __cse_9 = u[i_r]
            vbd[i_r] = vbd[i_r] + (__oldb_0 * __cse_9d + __cse_9 * __oldb_0d)
            vb[i_r] = vb[i_r] + __cse_9 * __oldb_0
            __hcse_3 = __cse_8 * __cse_8d
            __hcse_4 = __cse_8 * __cse_8
            ubd[i_r] = ubd[i_r] + (__oldb_0 * (__hcse_3 + __hcse_3) + __hcse_4 * __oldb_0d)
            ub[i_r] = ub[i_r] + __hcse_4 * __oldb_0
            __hcse_5 = __cse_9 * __cse_8
            __cse_10d = __oldb_0 * (__cse_8 * __cse_9d + __cse_9 * __cse_8d) + __hcse_5 * __oldb_0d
            __cse_10 = __hcse_5 * __oldb_0
            vbd[i_r] = vbd[i_r] + __cse_10d
            vb[i_r] = vb[i_r] + __cse_10
            vbd[i_r] = vbd[i_r] + __cse_10d
            vb[i_r] = vb[i_r] + __cse_10
        else
            __oldb_0d = sbd
            __oldb_0 = sb
            sbd = 0.0
            sb = 0.0
            __cse_11d = vd[i_r]
            __cse_11 = v[i_r]
            ubd[i_r] = ubd[i_r] + (__oldb_0 * __cse_11d + __cse_11 * __oldb_0d)
            ub[i_r] = ub[i_r] + __cse_11 * __oldb_0
            __cse_12d = ud[i_r]
            __cse_12 = u[i_r]
            vbd[i_r] = vbd[i_r] + (__oldb_0 * __cse_12d + __cse_12 * __oldb_0d)
            vb[i_r] = vb[i_r] + __cse_12 * __oldb_0
            __cse_13d = -__oldb_0d
            __cse_13 = -__oldb_0
            __hcse_6 = __cse_12 * __cse_11d
            __hcse_7 = __cse_11 * __cse_12d
            __hcse_8 = __cse_11 * __cse_12
            ubd[i_r] = ubd[i_r] + (__cse_13 * (__hcse_6 + __hcse_7) + __hcse_8 * __cse_13d)
            ub[i_r] = ub[i_r] + __hcse_8 * __cse_13
            __hcse_9 = __cse_12 * __cse_12d
            __hcse_10 = __cse_12 * __cse_12
            vbd[i_r] = vbd[i_r] + (__cse_13 * (__hcse_9 + __hcse_9) + __hcse_10 * __cse_13d)
            vb[i_r] = vb[i_r] + __hcse_10 * __cse_13
            __hcse_11 = __cse_12 * __cse_11
            ubd[i_r] = ubd[i_r] + (__cse_13 * (__hcse_7 + __hcse_6) + __hcse_11 * __cse_13d)
            ub[i_r] = ub[i_r] + __hcse_11 * __cse_13
        end
        sbd = 0.0
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
