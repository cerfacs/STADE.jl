function initstacks_while_reduction_b()
    whilecount_stack = Vector{Int64}()
    __wcarry_i_p_stack = Vector{Int64}()
    s_stack = Vector{Float64}()
    return (whilecount_stack, __wcarry_i_p_stack, s_stack)
end

function while_reduction_hv(loss, lossb, u, ub, a, ab, v, vb, i_n, lossd, lossbd, ud, ubd, ad, abd, vd, vbd, whilecount_stack, __wcarry_i_p_stack, s_stack)
    s_stack_d = Vector{Float64}()
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    i_p = 1
    __wcount_48126a = 0
    while i_p <= 2
        push!(__wcarry_i_p_stack, i_p)
        sd = 0.0
        s = 0.0
        for i_k = 1:i_n
            __hcse_0 = u[i_k]
            __hcse_1 = a[i_k]
            sd = sd + (__hcse_0 * ad[i_k] + __hcse_1 * ud[i_k])
            s = s + __hcse_1 * __hcse_0
        end
        for i_x = 1:i_n
            __hcse_2 = u[i_x]
            vd[i_x] = vd[i_x] + (__hcse_2 * sd + s * ud[i_x])
            v[i_x] = v[i_x] + s * __hcse_2
        end
        i_p = i_p + 1
        push!(s_stack_d, sd)
        push!(s_stack, s)
        __wcount_48126a = __wcount_48126a + 1
    end
    push!(whilecount_stack, __wcount_48126a)
    for i_x2 = 1:i_n
        __hcse_3 = v[i_x2]
        lossd[1] = lossd[1] + (2__hcse_3) * vd[i_x2]
        loss[1] = loss[1] + __hcse_3 ^ 2
    end
    for i_x2 = i_n:-1:1
        __hcse_4 = lossb[1]
        __hcse_5 = 2 * v[i_x2]
        vbd[i_x2] = vbd[i_x2] + (__hcse_4 * (2 * vd[i_x2]) + __hcse_5 * lossbd[1])
        vb[i_x2] = vb[i_x2] + __hcse_5 * __hcse_4
    end
    __wcount_48126a = pop!(whilecount_stack)
    for __witer_48126a = 1:__wcount_48126a
        i_p = pop!(__wcarry_i_p_stack)
        sd = pop!(s_stack_d)
        s = pop!(s_stack)
        for i_x = i_n:-1:1
            __cse_0d = vbd[i_x]
            __cse_0 = vb[i_x]
            __hcse_6 = u[i_x]
            sbd = sbd + (__cse_0 * ud[i_x] + __hcse_6 * __cse_0d)
            sb = sb + __hcse_6 * __cse_0
            ubd[i_x] = ubd[i_x] + (__cse_0 * sd + s * __cse_0d)
            ub[i_x] = ub[i_x] + s * __cse_0
        end
        for i_k = i_n:-1:1
            __hcse_7 = u[i_k]
            abd[i_k] = abd[i_k] + (sb * ud[i_k] + __hcse_7 * sbd)
            ab[i_k] = ab[i_k] + __hcse_7 * sb
            __hcse_8 = a[i_k]
            ubd[i_k] = ubd[i_k] + (sb * ad[i_k] + __hcse_8 * sbd)
            ub[i_k] = ub[i_k] + __hcse_8 * sb
        end
        sbd = 0.0
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
