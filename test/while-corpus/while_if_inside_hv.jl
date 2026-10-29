function initstacks_while_if_inside_b()
    whilecount_stack = Vector{Int64}()
    __wcarry_i_k_stack = Vector{Int64}()
    branch_stack = Vector{Int64}()
    return (whilecount_stack, __wcarry_i_k_stack, branch_stack)
end

function while_if_inside_hv(loss, lossb, u, ub, a, ab, v, vb, i_n, lossd, lossbd, ud, ubd, ad, abd, vd, vbd, whilecount_stack, __wcarry_i_k_stack, branch_stack)
    i_k = 1
    __wcount_d661fa = 0
    while i_k <= i_n
        push!(__wcarry_i_k_stack, i_k)
        if u[i_k] > 0.0
            __hcse_0 = u[i_k]
            __hcse_1 = a[i_k]
            vd[i_k] = __hcse_0 * ad[i_k] + __hcse_1 * ud[i_k]
            v[i_k] = __hcse_1 * __hcse_0
            push!(branch_stack, 1)
        else
            __cse_0d = ud[i_k]
            __cse_0 = u[i_k]
            __hcse_2 = a[i_k]
            __hcse_3 = (__hcse_2 * __cse_0) * __cse_0d
            vd[i_k] = ((__cse_0 * __cse_0) * ad[i_k] + __hcse_3) + __hcse_3
            v[i_k] = __hcse_2 * __cse_0 * __cse_0
            push!(branch_stack, 0)
        end
        i_k = i_k + 1
        __wcount_d661fa = __wcount_d661fa + 1
    end
    push!(whilecount_stack, __wcount_d661fa)
    for i_x = 1:i_n
        __hcse_4 = v[i_x]
        lossd[1] = lossd[1] + (2__hcse_4) * vd[i_x]
        loss[1] = loss[1] + __hcse_4 ^ 2
    end
    for i_x = i_n:-1:1
        __hcse_5 = lossb[1]
        __hcse_6 = 2 * v[i_x]
        vbd[i_x] = vbd[i_x] + (__hcse_5 * (2 * vd[i_x]) + __hcse_6 * lossbd[1])
        vb[i_x] = vb[i_x] + __hcse_6 * __hcse_5
    end
    __wcount_d661fa = pop!(whilecount_stack)
    for __witer_d661fa = 1:__wcount_d661fa
        i_k = pop!(__wcarry_i_k_stack)
        __branch = pop!(branch_stack)
        if __branch == 1
            __oldb_0d = vbd[i_k]
            __oldb_0 = vb[i_k]
            vbd[i_k] = 0.0
            vb[i_k] = 0.0
            __hcse_7 = u[i_k]
            abd[i_k] = abd[i_k] + (__oldb_0 * ud[i_k] + __hcse_7 * __oldb_0d)
            ab[i_k] = ab[i_k] + __hcse_7 * __oldb_0
            __hcse_8 = a[i_k]
            ubd[i_k] = ubd[i_k] + (__oldb_0 * ad[i_k] + __hcse_8 * __oldb_0d)
            ub[i_k] = ub[i_k] + __hcse_8 * __oldb_0
        else
            __oldb_0d = vbd[i_k]
            __oldb_0 = vb[i_k]
            vbd[i_k] = 0.0
            vb[i_k] = 0.0
            __cse_1d = ud[i_k]
            __cse_1 = u[i_k]
            __hcse_9 = __cse_1 * __cse_1d
            __hcse_10 = __cse_1 * __cse_1
            abd[i_k] = abd[i_k] + (__oldb_0 * (__hcse_9 + __hcse_9) + __hcse_10 * __oldb_0d)
            ab[i_k] = ab[i_k] + __hcse_10 * __oldb_0
            __hcse_11 = a[i_k]
            __hcse_12 = __hcse_11 * __cse_1
            __cse_2d = __oldb_0 * (__cse_1 * ad[i_k] + __hcse_11 * __cse_1d) + __hcse_12 * __oldb_0d
            __cse_2 = __hcse_12 * __oldb_0
            ubd[i_k] = ubd[i_k] + __cse_2d
            ub[i_k] = ub[i_k] + __cse_2
            ubd[i_k] = ubd[i_k] + __cse_2d
            ub[i_k] = ub[i_k] + __cse_2
        end
    end
    return nothing
end

function while_if_inside(loss, u, a, v, i_n)
    i_k = 1
    while i_k <= i_n
        if u[i_k] > 0.0
            v[i_k] = a[i_k] * u[i_k]
        else
            v[i_k] = a[i_k] * u[i_k] * u[i_k]
        end
        i_k = i_k + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
end
