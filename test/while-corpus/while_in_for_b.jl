function initstacks_while_in_for_b()
    whilecount_stack = Vector{Int64}()
    __wcarry_i_k_stack = Vector{Int64}()
    return (whilecount_stack, __wcarry_i_k_stack)
end

function while_in_for_b(loss, lossb, u, ub, a, ab, v, vb, i_n, i_m, whilecount_stack, __wcarry_i_k_stack)
    for i_j = 1:i_m
        i_k = 1
        __wcount_392499 = 0
        while i_k <= i_n
            push!(__wcarry_i_k_stack, i_k)
            v[i_k] = a[i_k] * u[i_k] + v[i_k]
            i_k = i_k + 1
            __wcount_392499 = __wcount_392499 + 1
        end
        push!(whilecount_stack, __wcount_392499)
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
    for i_x = i_n:-1:1
        vb[i_x] = vb[i_x] + (2 * v[i_x]) * lossb[1]
    end
    for i_j = i_m:-1:1
        __wcount_392499 = pop!(whilecount_stack)
        for __witer_392499 = 1:__wcount_392499
            i_k = pop!(__wcarry_i_k_stack)
            __cse_0 = vb[i_k]
            ab[i_k] = ab[i_k] + u[i_k] * __cse_0
            ub[i_k] = ub[i_k] + a[i_k] * __cse_0
        end
    end
    return nothing
end

function while_in_for(loss, u, a, v, i_n, i_m)
    for i_j = 1:i_m
        i_k = 1
        while i_k <= i_n
            v[i_k] = a[i_k] * u[i_k] + v[i_k]
            i_k = i_k + 1
        end
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
end
