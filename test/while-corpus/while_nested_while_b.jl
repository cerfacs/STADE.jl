function initstacks_while_nested_while_b()
    whilecount_stack = Vector{Int64}()
    __wcarry_i_p_stack = Vector{Int64}()
    __wcarry_i_k_stack = Vector{Int64}()
    return (whilecount_stack, __wcarry_i_p_stack, __wcarry_i_k_stack)
end

function while_nested_while_b(loss, lossb, u, ub, a, ab, v, vb, i_n, whilecount_stack, __wcarry_i_p_stack, __wcarry_i_k_stack)
    i_p = 1
    __wcount_df7f22 = 0
    while i_p <= 2
        push!(__wcarry_i_p_stack, i_p)
        i_k = 1
        __wcount_99ea1e = 0
        while i_k <= i_n
            push!(__wcarry_i_k_stack, i_k)
            v[i_k] = v[i_k] + a[i_k] * u[i_k] * i_p
            i_k = i_k + 1
            __wcount_99ea1e = __wcount_99ea1e + 1
        end
        push!(whilecount_stack, __wcount_99ea1e)
        i_p = i_p + 1
        __wcount_df7f22 = __wcount_df7f22 + 1
    end
    push!(whilecount_stack, __wcount_df7f22)
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
    for i_x = i_n:-1:1
        vb[i_x] = vb[i_x] + (2 * v[i_x]) * lossb[1]
    end
    __wcount_df7f22 = pop!(whilecount_stack)
    for __witer_df7f22 = 1:__wcount_df7f22
        i_p = pop!(__wcarry_i_p_stack)
        __wcount_99ea1e = pop!(whilecount_stack)
        for __witer_99ea1e = 1:__wcount_99ea1e
            i_k = pop!(__wcarry_i_k_stack)
            __cse_0 = vb[i_k]
            ab[i_k] = ab[i_k] + (u[i_k] * i_p) * __cse_0
            ub[i_k] = ub[i_k] + (a[i_k] * i_p) * __cse_0
        end
    end
    return nothing
end

function while_nested_while(loss, u, a, v, i_n)
    i_p = 1
    while i_p <= 2
        i_k = 1
        while i_k <= i_n
            v[i_k] = v[i_k] + a[i_k] * u[i_k] * i_p
            i_k = i_k + 1
        end
        i_p = i_p + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
end
