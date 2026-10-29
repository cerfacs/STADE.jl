function initstacks_while_count_b()
    whilecount_stack = Vector{Int64}()
    __wcarry_i_k_stack = Vector{Int64}()
    return (whilecount_stack, __wcarry_i_k_stack)
end

function while_count_b(loss, lossb, u, ub, a, ab, v, vb, i_n, whilecount_stack, __wcarry_i_k_stack)
    i_k = 1
    __wcount_b4541f = 0
    while i_k <= i_n
        push!(__wcarry_i_k_stack, i_k)
        v[i_k] = a[i_k] * u[i_k]
        i_k = i_k + 1
        __wcount_b4541f = __wcount_b4541f + 1
    end
    push!(whilecount_stack, __wcount_b4541f)
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
    for i_x = i_n:-1:1
        vb[i_x] = vb[i_x] + (2 * v[i_x]) * lossb[1]
    end
    __wcount_b4541f = pop!(whilecount_stack)
    for __witer_b4541f = 1:__wcount_b4541f
        i_k = pop!(__wcarry_i_k_stack)
        __oldb_0 = vb[i_k]
        vb[i_k] = 0.0
        ab[i_k] = ab[i_k] + u[i_k] * __oldb_0
        ub[i_k] = ub[i_k] + a[i_k] * __oldb_0
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
