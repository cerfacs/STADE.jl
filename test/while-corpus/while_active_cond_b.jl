function initstacks_while_active_cond_b()
    whilecount_stack = Vector{Int64}()
    __wcarry_i_k_stack = Vector{Int64}()
    s_stack = Vector{Float64}()
    return (whilecount_stack, __wcarry_i_k_stack, s_stack)
end

function while_active_cond_b(loss, lossb, u, ub, a, ab, v, vb, i_n, whilecount_stack, __wcarry_i_k_stack, s_stack)
    s = 0.0
    sb = 0.0
    s = 1.0
    i_k = 1
    __wcount_e1371a = 0
    while s > 0.05 && i_k <= i_n
        push!(__wcarry_i_k_stack, i_k)
        push!(s_stack, s)
        __cse_0 = a[i_k]
        s = s * (0.5 + 0.25 * __cse_0 * __cse_0)
        v[i_k] = s * u[i_k]
        i_k = i_k + 1
        __wcount_e1371a = __wcount_e1371a + 1
    end
    push!(whilecount_stack, __wcount_e1371a)
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
    push!(s_stack, s)
    s = pop!(s_stack)
    for i_x = i_n:-1:1
        vb[i_x] = vb[i_x] + (2 * v[i_x]) * lossb[1]
    end
    __wcount_e1371a = pop!(whilecount_stack)
    for __witer_e1371a = 1:__wcount_e1371a
        i_k = pop!(__wcarry_i_k_stack)
        __oldb_0 = vb[i_k]
        vb[i_k] = 0.0
        sb = sb + u[i_k] * __oldb_0
        ub[i_k] = ub[i_k] + s * __oldb_0
        s = pop!(s_stack)
        __oldb_1 = sb
        sb = 0.0
        __cse_1 = a[i_k]
        sb = sb + (0.5 + 0.25 * __cse_1 * __cse_1) * __oldb_1
        __cse_2 = (0.25__cse_1) * (s * __oldb_1)
        ab[i_k] = ab[i_k] + __cse_2
        ab[i_k] = ab[i_k] + __cse_2
    end
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
