function initstacks_raggedii_b(m0, n)
    __sz_s_stack = 0
    for i_x = 1:n
        __sz_s_stack += 1
        s = 0.0
        for i_y = 1:2
            w = m0 + i_y
            for i_j = 1:w
                __sz_s_stack += 1
            end
            __sz_s_stack += 1
        end
        __sz_s_stack += 1
    end
    __sz_s_stack += 1
    prefix_tripcount_stack_1 = Vector{Int}(undef, div(n - 1, 1) + 1)
    __tot_tripcount_stack_1 = 0
    for i_x = 1:n
        prefix_tripcount_stack_1[(i_x - 1) + 1] = __tot_tripcount_stack_1
        s = 0.0
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + (div(2 - 1, 1) + 1)
    end
    s_stack = Vector{Float64}()
    sizehint!(s_stack, __sz_s_stack)
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    return (s_stack, tripcount_stack, prefix_tripcount_stack_1, __tot_tripcount_stack_1)
end

function raggedii_hv(out, outb, u, ub, x, xb, n, m0, outd, outbd, ud, ubd, xd, xbd, s_stack, tripcount_stack, prefix_tripcount_stack_1, __tot_tripcount_stack_1)
    __sz_s_stack = 0
    for i_x = 1:n
        __sz_s_stack += 1
        s = 0.0
        for i_y = 1:2
            w = m0 + i_y
            for i_j = 1:w
                __sz_s_stack += 1
            end
            __sz_s_stack += 1
        end
        __sz_s_stack += 1
    end
    __sz_s_stack += 1
    s_stack_d = Vector{Float64}()
    sizehint!(s_stack_d, __sz_s_stack)
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    for i_x = 1:n
        push!(s_stack_d, sd)
        push!(s_stack, s)
        sd = 0.0
        s = 0.0
        for i_y = 1:2
            w = m0 + i_y
            __idx_tripcount_stack_1_1 = prefix_tripcount_stack_1[(i_x - 1) + 1] + ((i_y - 1) + 1)
            tripcount_stack[__idx_tripcount_stack_1_1] = w
            for i_j = 1:w
                push!(s_stack_d, sd)
                push!(s_stack, s)
                sd = sd + (u[i_x] * xd[i_j] + x[i_j] * ud[i_x])
                s = s + x[i_j] * u[i_x]
            end
            push!(s_stack_d, sd)
            push!(s_stack, s)
        end
        outd[i_x] = s * sd + s * sd
        out[i_x] = s * s
        push!(s_stack_d, sd)
        push!(s_stack, s)
    end
    push!(s_stack_d, sd)
    push!(s_stack, s)
    sd = pop!(s_stack_d)
    s = pop!(s_stack)
    for i_x = n:-1:1
        sd = pop!(s_stack_d)
        s = pop!(s_stack)
        sbd = sbd + (outb[i_x] * sd + s * outbd[i_x])
        sb = sb + s * outb[i_x]
        sbd = sbd + (outb[i_x] * sd + s * outbd[i_x])
        sb = sb + s * outb[i_x]
        outbd[i_x] = 0.0
        outb[i_x] = 0.0
        for i_y = 2:-1:1
            sd = pop!(s_stack_d)
            s = pop!(s_stack)
            w = m0 + i_y
            __idx_tripcount_stack_1_2 = prefix_tripcount_stack_1[(i_x - 1) + 1] + ((i_y - 1) + 1)
            w = tripcount_stack[__idx_tripcount_stack_1_2]
            for i_j = w:-1:1
                sd = pop!(s_stack_d)
                s = pop!(s_stack)
                xbd[i_j] = xbd[i_j] + (sb * ud[i_x] + u[i_x] * sbd)
                xb[i_j] = xb[i_j] + u[i_x] * sb
                ubd[i_x] = ubd[i_x] + (sb * xd[i_j] + x[i_j] * sbd)
                ub[i_x] = ub[i_x] + x[i_j] * sb
            end
        end
        sd = pop!(s_stack_d)
        s = pop!(s_stack)
        sbd = 0.0
        sb = 0.0
    end
    return nothing
end

function raggedii(out, u, x, n, m0)
    for i_x = 1:n
        s = 0.0
        for i_y = 1:2
            w = m0 + i_y
            for i_j = 1:w
                s = s + x[i_j] * u[i_x]
            end
        end
        out[i_x] = s * s
    end
end
