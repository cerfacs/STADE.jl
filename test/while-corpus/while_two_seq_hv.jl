function initstacks_while_two_seq_b()
    whilecount_stack = Vector{Int64}()
    __wcarry_i_k_stack = Vector{Int64}()
    __wcarry_i_q_stack = Vector{Int64}()
    return (whilecount_stack, __wcarry_i_k_stack, __wcarry_i_q_stack)
end

function while_two_seq_hv(loss, lossb, u, ub, a, ab, v, vb, w, wb, i_n, lossd, lossbd, ud, ubd, ad, abd, vd, vbd, wd, wbd, whilecount_stack, __wcarry_i_k_stack, __wcarry_i_q_stack)
    i_k = 1
    __wcount_153f9f = 0
    while i_k <= i_n
        push!(__wcarry_i_k_stack, i_k)
        __hcse_0 = u[i_k]
        __hcse_1 = a[i_k]
        vd[i_k] = __hcse_0 * ad[i_k] + __hcse_1 * ud[i_k]
        v[i_k] = __hcse_1 * __hcse_0
        i_k = i_k + 1
        __wcount_153f9f = __wcount_153f9f + 1
    end
    push!(whilecount_stack, __wcount_153f9f)
    i_q = 1
    __wcount_da9494 = 0
    while i_q < i_n
        push!(__wcarry_i_q_stack, i_q)
        __hcse_2 = v[i_q + 1]
        __hcse_3 = v[i_q]
        wd[i_q] = __hcse_2 * vd[i_q] + __hcse_3 * vd[i_q + 1]
        w[i_q] = __hcse_3 * __hcse_2
        i_q = i_q + 1
        __wcount_da9494 = __wcount_da9494 + 1
    end
    push!(whilecount_stack, __wcount_da9494)
    for i_x = 1:i_n
        __hcse_4 = v[i_x]
        __hcse_5 = w[i_x]
        lossd[1] = (lossd[1] + (2__hcse_4) * vd[i_x]) + (2__hcse_5) * wd[i_x]
        loss[1] = loss[1] + __hcse_4 ^ 2 + __hcse_5 ^ 2
    end
    for i_x = i_n:-1:1
        __cse_0d = lossbd[1]
        __cse_0 = lossb[1]
        __hcse_6 = 2 * v[i_x]
        vbd[i_x] = vbd[i_x] + (__cse_0 * (2 * vd[i_x]) + __hcse_6 * __cse_0d)
        vb[i_x] = vb[i_x] + __hcse_6 * __cse_0
        __hcse_7 = 2 * w[i_x]
        wbd[i_x] = wbd[i_x] + (__cse_0 * (2 * wd[i_x]) + __hcse_7 * __cse_0d)
        wb[i_x] = wb[i_x] + __hcse_7 * __cse_0
    end
    __wcount_da9494 = pop!(whilecount_stack)
    for __witer_da9494 = 1:__wcount_da9494
        i_q = pop!(__wcarry_i_q_stack)
        __oldb_0d = wbd[i_q]
        __oldb_0 = wb[i_q]
        wbd[i_q] = 0.0
        wb[i_q] = 0.0
        __hcse_8 = v[i_q + 1]
        vbd[i_q] = vbd[i_q] + (__oldb_0 * vd[i_q + 1] + __hcse_8 * __oldb_0d)
        vb[i_q] = vb[i_q] + __hcse_8 * __oldb_0
        __hcse_9 = v[i_q]
        vbd[i_q + 1] = vbd[i_q + 1] + (__oldb_0 * vd[i_q] + __hcse_9 * __oldb_0d)
        vb[i_q + 1] = vb[i_q + 1] + __hcse_9 * __oldb_0
    end
    __wcount_153f9f = pop!(whilecount_stack)
    for __witer_153f9f = 1:__wcount_153f9f
        i_k = pop!(__wcarry_i_k_stack)
        __oldb_0d = vbd[i_k]
        __oldb_0 = vb[i_k]
        vbd[i_k] = 0.0
        vb[i_k] = 0.0
        __hcse_10 = u[i_k]
        abd[i_k] = abd[i_k] + (__oldb_0 * ud[i_k] + __hcse_10 * __oldb_0d)
        ab[i_k] = ab[i_k] + __hcse_10 * __oldb_0
        __hcse_11 = a[i_k]
        ubd[i_k] = ubd[i_k] + (__oldb_0 * ad[i_k] + __hcse_11 * __oldb_0d)
        ub[i_k] = ub[i_k] + __hcse_11 * __oldb_0
    end
    return nothing
end

function while_two_seq(loss, u, a, v, w, i_n)
    i_k = 1
    while i_k <= i_n
        v[i_k] = a[i_k] * u[i_k]
        i_k = i_k + 1
    end
    i_q = 1
    while i_q < i_n
        w[i_q] = v[i_q] * v[i_q + 1]
        i_q = i_q + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2 + w[i_x] ^ 2
    end
end
