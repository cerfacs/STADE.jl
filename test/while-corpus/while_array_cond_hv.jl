function initstacks_while_array_cond_b()
    whilecount_stack = Vector{Int64}()
    __wcarry_i_k_stack = Vector{Int64}()
    v_stack = Vector{Float64}()
    return (whilecount_stack, __wcarry_i_k_stack, v_stack)
end

function while_array_cond_hv(loss, lossb, u, ub, a, ab, v, vb, i_n, lossd, lossbd, ud, ubd, ad, abd, vd, vbd, whilecount_stack, __wcarry_i_k_stack, v_stack)
    v_stack_d = Vector{Float64}()
    i_k = 1
    vd[1] = 0.0
    v[1] = 1.0
    __wcount_7c1bc4 = 0
    while v[i_k] > 0.1 && i_k < i_n
        push!(__wcarry_i_k_stack, i_k)
        push!(v_stack_d, vd[i_k + 1])
        push!(v_stack, v[i_k + 1])
        __cse_0d = ad[i_k]
        __cse_0 = a[i_k]
        __hcse_0 = 1.0 + __cse_0 * __cse_0
        __hcse_1 = v[i_k]
        __hcse_2 = __cse_0 * __cse_0d
        vd[i_k + 1] = (0.5__hcse_0) * vd[i_k] + (0.5__hcse_1) * (__hcse_2 + __hcse_2)
        v[i_k + 1] = 0.5 * __hcse_1 * __hcse_0
        i_k = i_k + 1
        __wcount_7c1bc4 = __wcount_7c1bc4 + 1
    end
    push!(whilecount_stack, __wcount_7c1bc4)
    for i_x = 1:i_n
        __hcse_3 = v[i_x]
        __hcse_4 = u[i_x]
        __hcse_5 = __hcse_3 * __hcse_4
        lossd[1] = lossd[1] + (2__hcse_5) * (__hcse_4 * vd[i_x] + __hcse_3 * ud[i_x])
        loss[1] = loss[1] + __hcse_5 ^ 2
    end
    for i_x = i_n:-1:1
        __cse_1d = ud[i_x]
        __cse_1 = u[i_x]
        __cse_2d = vd[i_x]
        __cse_2 = v[i_x]
        __hcse_6 = lossb[1]
        __hcse_7 = 2 * (__cse_2 * __cse_1)
        __cse_3d = __hcse_6 * (2 * (__cse_1 * __cse_2d + __cse_2 * __cse_1d)) + __hcse_7 * lossbd[1]
        __cse_3 = __hcse_7 * __hcse_6
        vbd[i_x] = vbd[i_x] + (__cse_3 * __cse_1d + __cse_1 * __cse_3d)
        vb[i_x] = vb[i_x] + __cse_1 * __cse_3
        ubd[i_x] = ubd[i_x] + (__cse_3 * __cse_2d + __cse_2 * __cse_3d)
        ub[i_x] = ub[i_x] + __cse_2 * __cse_3
    end
    __wcount_7c1bc4 = pop!(whilecount_stack)
    for __witer_7c1bc4 = 1:__wcount_7c1bc4
        i_k = pop!(__wcarry_i_k_stack)
        vd[i_k + 1] = pop!(v_stack_d)
        v[i_k + 1] = pop!(v_stack)
        __oldb_1d = vbd[i_k + 1]
        __oldb_1 = vb[i_k + 1]
        vbd[i_k + 1] = 0.0
        vb[i_k + 1] = 0.0
        __cse_4d = ad[i_k]
        __cse_4 = a[i_k]
        __hcse_8 = __cse_4 * __cse_4d
        __hcse_9 = 0.5 * (1.0 + __cse_4 * __cse_4)
        vbd[i_k] = vbd[i_k] + (__oldb_1 * (0.5 * (__hcse_8 + __hcse_8)) + __hcse_9 * __oldb_1d)
        vb[i_k] = vb[i_k] + __hcse_9 * __oldb_1
        __hcse_10 = 0.5 * v[i_k]
        __hcse_11 = __hcse_10 * __oldb_1
        __cse_5d = __hcse_11 * __cse_4d + __cse_4 * (__oldb_1 * (0.5 * vd[i_k]) + __hcse_10 * __oldb_1d)
        __cse_5 = __cse_4 * __hcse_11
        abd[i_k] = abd[i_k] + __cse_5d
        ab[i_k] = ab[i_k] + __cse_5
        abd[i_k] = abd[i_k] + __cse_5d
        ab[i_k] = ab[i_k] + __cse_5
    end
    vbd[1] = 0.0
    vb[1] = 0.0
    return nothing
end

function while_array_cond(loss, u, a, v, i_n)
    i_k = 1
    v[1] = 1.0
    while v[i_k] > 0.1 && i_k < i_n
        v[i_k + 1] = 0.5 * v[i_k] * (1.0 + a[i_k] * a[i_k])
        i_k = i_k + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + (v[i_x] * u[i_x]) ^ 2
    end
end
