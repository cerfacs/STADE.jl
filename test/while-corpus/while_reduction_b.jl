function initstacks_while_reduction_b()
    whilecount_stack = Vector{Int64}()
    __wcarry_i_p_stack = Vector{Int64}()
    s_stack = Vector{Float64}()
    return (whilecount_stack, __wcarry_i_p_stack, s_stack)
end

function while_reduction_b(loss, lossb, u, ub, a, ab, v, vb, i_n, whilecount_stack, __wcarry_i_p_stack, s_stack)
    s = 0.0
    sb = 0.0
    i_p = 1
    __wcount_54bc80 = 0
    while i_p <= 2
        push!(__wcarry_i_p_stack, i_p)
        s = 0.0
        for i_k = 1:i_n
            s = s + a[i_k] * u[i_k]
        end
        for i_x = 1:i_n
            v[i_x] = v[i_x] + s * u[i_x]
        end
        i_p = i_p + 1
        push!(s_stack, s)
        __wcount_54bc80 = __wcount_54bc80 + 1
    end
    push!(whilecount_stack, __wcount_54bc80)
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
    for i_x2 = i_n:-1:1
        vb[i_x2] = vb[i_x2] + (2 * v[i_x2]) * lossb[1]
    end
    __wcount_54bc80 = pop!(whilecount_stack)
    for __witer_54bc80 = 1:__wcount_54bc80
        i_p = pop!(__wcarry_i_p_stack)
        s = pop!(s_stack)
        for i_x = i_n:-1:1
            __cse_0 = vb[i_x]
            sb = sb + u[i_x] * __cse_0
            ub[i_x] = ub[i_x] + s * __cse_0
        end
        for i_k = i_n:-1:1
            ab[i_k] = ab[i_k] + u[i_k] * sb
            ub[i_k] = ub[i_k] + a[i_k] * sb
        end
        sb = 0.0
    end
    return nothing
end

function while_reduction(loss, u, a, v, i_n)
    i_p = 1
    while i_p <= 2
        s = 0.0
        for i_k = 1:i_n
            s = s + a[i_k] * u[i_k]
        end
        for i_x = 1:i_n
            v[i_x] = v[i_x] + s * u[i_x]
        end
        i_p = i_p + 1
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
end
