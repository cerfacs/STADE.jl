function initstacks_while_for_inside_b()
    whilecount_stack = Vector{Int64}()
    __wcarry_i_p_stack = Vector{Int64}()
    return (whilecount_stack, __wcarry_i_p_stack)
end

function while_for_inside_b(loss, lossb, u, ub, a, ab, v, vb, i_n, whilecount_stack, __wcarry_i_p_stack)
    i_p = 1
    __wcount_f86709 = 0
    while i_p <= 2
        push!(__wcarry_i_p_stack, i_p)
        for i_x = 1:i_n
            v[i_x] = v[i_x] + a[i_x] * u[i_x]
        end
        i_p = i_p + 1
        __wcount_f86709 = __wcount_f86709 + 1
    end
    push!(whilecount_stack, __wcount_f86709)
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
    for i_x2 = i_n:-1:1
        vb[i_x2] = vb[i_x2] + (2 * v[i_x2]) * lossb[1]
    end
    __wcount_f86709 = pop!(whilecount_stack)
    for __witer_f86709 = 1:__wcount_f86709
        i_p = pop!(__wcarry_i_p_stack)
        for i_x = i_n:-1:1
            __cse_0 = vb[i_x]
            ab[i_x] = ab[i_x] + u[i_x] * __cse_0
            ub[i_x] = ub[i_x] + a[i_x] * __cse_0
        end
    end
    return nothing
end

function while_for_inside(loss, u, a, v, i_n)
    i_p = 1
    while i_p <= 2
        for i_x = 1:i_n
            v[i_x] = v[i_x] + a[i_x] * u[i_x]
        end
        i_p = i_p + 1
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
end
