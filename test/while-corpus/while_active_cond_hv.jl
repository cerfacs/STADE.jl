function initstacks_while_active_cond_b()
    whilecount_stack = Vector{Int64}()
    __wcarry_i_k_stack = Vector{Int64}()
    s_stack = Vector{Float64}()
    return (whilecount_stack, __wcarry_i_k_stack, s_stack)
end

function while_active_cond_hv(loss, lossb, u, ub, a, ab, v, vb, i_n, lossd, lossbd, ud, ubd, ad, abd, vd, vbd, whilecount_stack, __wcarry_i_k_stack, s_stack)
    s_stack_d = Vector{Float64}()
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    sd = 0.0
    s = 1.0
    i_k = 1
    __wcount_33f279 = 0
    while s > 0.05 && i_k <= i_n
        push!(__wcarry_i_k_stack, i_k)
        push!(s_stack_d, sd)
        push!(s_stack, s)
        __cse_0d = ad[i_k]
        __cse_0 = a[i_k]
        __hcse_0 = 0.5 + 0.25 * __cse_0 * __cse_0
        __hcse_1 = (0.25__cse_0) * __cse_0d
        sd = __hcse_0 * sd + s * (__hcse_1 + __hcse_1)
        s = s * __hcse_0
        __hcse_2 = u[i_k]
        vd[i_k] = __hcse_2 * sd + s * ud[i_k]
        v[i_k] = s * __hcse_2
        i_k = i_k + 1
        __wcount_33f279 = __wcount_33f279 + 1
    end
    push!(whilecount_stack, __wcount_33f279)
    for i_x = 1:i_n
        __hcse_3 = v[i_x]
        lossd[1] = lossd[1] + (2__hcse_3) * vd[i_x]
        loss[1] = loss[1] + __hcse_3 ^ 2
    end
    push!(s_stack_d, sd)
    push!(s_stack, s)
    sd = pop!(s_stack_d)
    s = pop!(s_stack)
    for i_x = i_n:-1:1
        __hcse_4 = lossb[1]
        __hcse_5 = 2 * v[i_x]
        vbd[i_x] = vbd[i_x] + (__hcse_4 * (2 * vd[i_x]) + __hcse_5 * lossbd[1])
        vb[i_x] = vb[i_x] + __hcse_5 * __hcse_4
    end
    __wcount_33f279 = pop!(whilecount_stack)
    for __witer_33f279 = 1:__wcount_33f279
        i_k = pop!(__wcarry_i_k_stack)
        __oldb_0d = vbd[i_k]
        __oldb_0 = vb[i_k]
        vbd[i_k] = 0.0
        vb[i_k] = 0.0
        __hcse_6 = u[i_k]
        sbd = sbd + (__oldb_0 * ud[i_k] + __hcse_6 * __oldb_0d)
        sb = sb + __hcse_6 * __oldb_0
        ubd[i_k] = ubd[i_k] + (__oldb_0 * sd + s * __oldb_0d)
        ub[i_k] = ub[i_k] + s * __oldb_0
        sd = pop!(s_stack_d)
        s = pop!(s_stack)
        __oldb_1d = sbd
        __oldb_1 = sb
        sbd = 0.0
        sb = 0.0
        __cse_1d = ad[i_k]
        __cse_1 = a[i_k]
        __hcse_7 = 0.25__cse_1
        __hcse_8 = __hcse_7 * __cse_1d
        __hcse_9 = 0.5 + 0.25 * __cse_1 * __cse_1
        sbd = sbd + (__oldb_1 * (__hcse_8 + __hcse_8) + __hcse_9 * __oldb_1d)
        sb = sb + __hcse_9 * __oldb_1
        __hcse_10 = s * __oldb_1
        __cse_2d = __hcse_10 * (0.25__cse_1d) + __hcse_7 * (__oldb_1 * sd + s * __oldb_1d)
        __cse_2 = __hcse_7 * __hcse_10
        abd[i_k] = abd[i_k] + __cse_2d
        ab[i_k] = ab[i_k] + __cse_2
        abd[i_k] = abd[i_k] + __cse_2d
        ab[i_k] = ab[i_k] + __cse_2
    end
    sbd = 0.0
    sb = 0.0
    return nothing
end

function while_active_cond(loss, u, a, v, i_n)
    s = 1.0
    i_k = 1
    while s > 0.05 && i_k <= i_n
        s = s * (0.5 + 0.25 * a[i_k] * a[i_k])
        v[i_k] = s * u[i_k]
        i_k = i_k + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
end
