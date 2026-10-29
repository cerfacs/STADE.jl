function initstacks_while_for_inside_b()
    whilecount_stack = Vector{Int64}()
    __wcarry_i_p_stack = Vector{Int64}()
    return (whilecount_stack, __wcarry_i_p_stack)
end

function while_for_inside_hv(loss, lossb, u, ub, a, ab, v, vb, i_n, lossd, lossbd, ud, ubd, ad, abd, vd, vbd, whilecount_stack, __wcarry_i_p_stack)
    i_p = 1
    __wcount_c4b675 = 0
    while i_p <= 2
        push!(__wcarry_i_p_stack, i_p)
        for i_x = 1:i_n
            __hcse_0 = u[i_x]
            __hcse_1 = a[i_x]
            vd[i_x] = vd[i_x] + (__hcse_0 * ad[i_x] + __hcse_1 * ud[i_x])
            v[i_x] = v[i_x] + __hcse_1 * __hcse_0
        end
        i_p = i_p + 1
        __wcount_c4b675 = __wcount_c4b675 + 1
    end
    push!(whilecount_stack, __wcount_c4b675)
    for i_x2 = 1:i_n
        __hcse_2 = v[i_x2]
        lossd[1] = lossd[1] + (2__hcse_2) * vd[i_x2]
        loss[1] = loss[1] + __hcse_2 ^ 2
    end
    for i_x2 = i_n:-1:1
        __hcse_3 = lossb[1]
        __hcse_4 = 2 * v[i_x2]
        vbd[i_x2] = vbd[i_x2] + (__hcse_3 * (2 * vd[i_x2]) + __hcse_4 * lossbd[1])
        vb[i_x2] = vb[i_x2] + __hcse_4 * __hcse_3
    end
    __wcount_c4b675 = pop!(whilecount_stack)
    for __witer_c4b675 = 1:__wcount_c4b675
        i_p = pop!(__wcarry_i_p_stack)
        for i_x = i_n:-1:1
            __cse_0d = vbd[i_x]
            __cse_0 = vb[i_x]
            __hcse_5 = u[i_x]
            abd[i_x] = abd[i_x] + (__cse_0 * ud[i_x] + __hcse_5 * __cse_0d)
            ab[i_x] = ab[i_x] + __hcse_5 * __cse_0
            __hcse_6 = a[i_x]
            ubd[i_x] = ubd[i_x] + (__cse_0 * ad[i_x] + __hcse_6 * __cse_0d)
            ub[i_x] = ub[i_x] + __hcse_6 * __cse_0
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
