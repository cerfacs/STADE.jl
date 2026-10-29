function initstacks_while_array_cond_b()
    whilecount_stack = Vector{Int64}()
    __wcarry_i_k_stack = Vector{Int64}()
    v_stack = Vector{Float64}()
    return (whilecount_stack, __wcarry_i_k_stack, v_stack)
end

function while_array_cond_b(loss, lossb, u, ub, a, ab, v, vb, i_n, whilecount_stack, __wcarry_i_k_stack, v_stack)
    i_k = 1
    v[1] = 1.0
    __wcount_2a3210 = 0
    while v[i_k] > 0.1 && i_k < i_n
        push!(__wcarry_i_k_stack, i_k)
        push!(v_stack, v[i_k + 1])
        __cse_0 = a[i_k]
        v[i_k + 1] = 0.5 * v[i_k] * (1.0 + __cse_0 * __cse_0)
        i_k = i_k + 1
        __wcount_2a3210 = __wcount_2a3210 + 1
    end
    push!(whilecount_stack, __wcount_2a3210)
    for i_x = 1:i_n
        loss[1] = loss[1] + (v[i_x] * u[i_x]) ^ 2
    end
    for i_x = i_n:-1:1
        __cse_1 = u[i_x]
        __cse_2 = v[i_x]
        __cse_3 = (2 * (__cse_2 * __cse_1)) * lossb[1]
        vb[i_x] = vb[i_x] + __cse_1 * __cse_3
        ub[i_x] = ub[i_x] + __cse_2 * __cse_3
    end
    __wcount_2a3210 = pop!(whilecount_stack)
    for __witer_2a3210 = 1:__wcount_2a3210
        i_k = pop!(__wcarry_i_k_stack)
        v[i_k + 1] = pop!(v_stack)
        __oldb_1 = vb[i_k + 1]
        vb[i_k + 1] = 0.0
        __cse_4 = a[i_k]
        vb[i_k] = vb[i_k] + (0.5 * (1.0 + __cse_4 * __cse_4)) * __oldb_1
        __cse_5 = __cse_4 * ((0.5 * v[i_k]) * __oldb_1)
        ab[i_k] = ab[i_k] + __cse_5
        ab[i_k] = ab[i_k] + __cse_5
    end
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
