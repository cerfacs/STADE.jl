function initstacks_while_two_seq_b()
    whilecount_stack = Vector{Int64}()
    __wcarry_i_k_stack = Vector{Int64}()
    __wcarry_i_q_stack = Vector{Int64}()
    return (whilecount_stack, __wcarry_i_k_stack, __wcarry_i_q_stack)
end

function while_two_seq_b(loss, lossb, u, ub, a, ab, v, vb, w, wb, i_n, whilecount_stack, __wcarry_i_k_stack, __wcarry_i_q_stack)
    i_k = 1
    __wcount_f5277c = 0
    while i_k <= i_n
        push!(__wcarry_i_k_stack, i_k)
        v[i_k] = a[i_k] * u[i_k]
        i_k = i_k + 1
        __wcount_f5277c = __wcount_f5277c + 1
    end
    push!(whilecount_stack, __wcount_f5277c)
    i_q = 1
    __wcount_ce6817 = 0
    while i_q < i_n
        push!(__wcarry_i_q_stack, i_q)
        w[i_q] = v[i_q] * v[i_q + 1]
        i_q = i_q + 1
        __wcount_ce6817 = __wcount_ce6817 + 1
    end
    push!(whilecount_stack, __wcount_ce6817)
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2 + w[i_x] ^ 2
    end
    for i_x = i_n:-1:1
        __cse_0 = lossb[1]
        vb[i_x] = vb[i_x] + (2 * v[i_x]) * __cse_0
        wb[i_x] = wb[i_x] + (2 * w[i_x]) * __cse_0
    end
    __wcount_ce6817 = pop!(whilecount_stack)
    for __witer_ce6817 = 1:__wcount_ce6817
        i_q = pop!(__wcarry_i_q_stack)
        __oldb_0 = wb[i_q]
        wb[i_q] = 0.0
        vb[i_q] = vb[i_q] + v[i_q + 1] * __oldb_0
        vb[i_q + 1] = vb[i_q + 1] + v[i_q] * __oldb_0
    end
    __wcount_f5277c = pop!(whilecount_stack)
    for __witer_f5277c = 1:__wcount_f5277c
        i_k = pop!(__wcarry_i_k_stack)
        __oldb_0 = vb[i_k]
        vb[i_k] = 0.0
        ab[i_k] = ab[i_k] + u[i_k] * __oldb_0
        ub[i_k] = ub[i_k] + a[i_k] * __oldb_0
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
