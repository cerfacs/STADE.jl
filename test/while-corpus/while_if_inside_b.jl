function initstacks_while_if_inside_b()
    whilecount_stack = Vector{Int64}()
    __wcarry_i_k_stack = Vector{Int64}()
    branch_stack = Vector{Int64}()
    return (whilecount_stack, __wcarry_i_k_stack, branch_stack)
end

function while_if_inside_b(loss, lossb, u, ub, a, ab, v, vb, i_n, whilecount_stack, __wcarry_i_k_stack, branch_stack)
    i_k = 1
    __wcount_83ac30 = 0
    while i_k <= i_n
        push!(__wcarry_i_k_stack, i_k)
        if u[i_k] > 0.0
            v[i_k] = a[i_k] * u[i_k]
            push!(branch_stack, 1)
        else
            __cse_0 = u[i_k]
            v[i_k] = a[i_k] * __cse_0 * __cse_0
            push!(branch_stack, 0)
        end
        i_k = i_k + 1
        __wcount_83ac30 = __wcount_83ac30 + 1
    end
    push!(whilecount_stack, __wcount_83ac30)
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
    for i_x = i_n:-1:1
        vb[i_x] = vb[i_x] + (2 * v[i_x]) * lossb[1]
    end
    __wcount_83ac30 = pop!(whilecount_stack)
    for __witer_83ac30 = 1:__wcount_83ac30
        i_k = pop!(__wcarry_i_k_stack)
        __branch = pop!(branch_stack)
        if __branch == 1
            __oldb_0 = vb[i_k]
            vb[i_k] = 0.0
            ab[i_k] = ab[i_k] + u[i_k] * __oldb_0
            ub[i_k] = ub[i_k] + a[i_k] * __oldb_0
        else
            __oldb_0 = vb[i_k]
            vb[i_k] = 0.0
            __cse_1 = u[i_k]
            ab[i_k] = ab[i_k] + (__cse_1 * __cse_1) * __oldb_0
            __cse_2 = (a[i_k] * __cse_1) * __oldb_0
            ub[i_k] = ub[i_k] + __cse_2
            ub[i_k] = ub[i_k] + __cse_2
        end
    end
    return nothing
end

function while_if_inside(loss, u, a, v, i_n)
    i_k = 1
    while i_k <= i_n
        if u[i_k] > 0.0
            v[i_k] = a[i_k] * u[i_k]
        else
            v[i_k] = a[i_k] * u[i_k] * u[i_k]
        end
        i_k = i_k + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
end
