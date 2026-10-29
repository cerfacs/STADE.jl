function initstacks_while_count_b()
    whilecount_stack = Vector{Int64}()
    __wcarry_i_k_stack = Vector{Int64}()
    return (whilecount_stack, __wcarry_i_k_stack)
end

function while_count_hv(loss, lossb, u, ub, a, ab, v, vb, i_n, lossd, lossbd, ud, ubd, ad, abd, vd, vbd, whilecount_stack, __wcarry_i_k_stack)
    i_k = 1
    __wcount_f9ce1f = 0
    while i_k <= i_n
        push!(__wcarry_i_k_stack, i_k)
        __hcse_0 = u[i_k]
        __hcse_1 = a[i_k]
        vd[i_k] = __hcse_0 * ad[i_k] + __hcse_1 * ud[i_k]
        v[i_k] = __hcse_1 * __hcse_0
        i_k = i_k + 1
        __wcount_f9ce1f = __wcount_f9ce1f + 1
    end
    push!(whilecount_stack, __wcount_f9ce1f)
    for i_x = 1:i_n
        __hcse_2 = v[i_x]
        lossd[1] = lossd[1] + (2__hcse_2) * vd[i_x]
        loss[1] = loss[1] + __hcse_2 ^ 2
    end
    for i_x = i_n:-1:1
        __hcse_3 = lossb[1]
        __hcse_4 = 2 * v[i_x]
        vbd[i_x] = vbd[i_x] + (__hcse_3 * (2 * vd[i_x]) + __hcse_4 * lossbd[1])
        vb[i_x] = vb[i_x] + __hcse_4 * __hcse_3
    end
    __wcount_f9ce1f = pop!(whilecount_stack)
    for __witer_f9ce1f = 1:__wcount_f9ce1f
        i_k = pop!(__wcarry_i_k_stack)
        __oldb_0d = vbd[i_k]
        __oldb_0 = vb[i_k]
        vbd[i_k] = 0.0
        vb[i_k] = 0.0
        __hcse_5 = u[i_k]
        abd[i_k] = abd[i_k] + (__oldb_0 * ud[i_k] + __hcse_5 * __oldb_0d)
        ab[i_k] = ab[i_k] + __hcse_5 * __oldb_0
        __hcse_6 = a[i_k]
        ubd[i_k] = ubd[i_k] + (__oldb_0 * ad[i_k] + __hcse_6 * __oldb_0d)
        ub[i_k] = ub[i_k] + __hcse_6 * __oldb_0
    end
    return nothing
end

function while_count(loss, u, a, v, i_n)
    i_k = 1
    while i_k <= i_n
        v[i_k] = a[i_k] * u[i_k]
        i_k = i_k + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
end
