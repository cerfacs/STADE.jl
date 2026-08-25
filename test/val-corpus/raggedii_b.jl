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

function raggedii_b(out, outb, u, ub, x, xb, n, m0, s_stack, tripcount_stack, prefix_tripcount_stack_1, __tot_tripcount_stack_1)
    s = 0.0
    sb = 0.0
    for i_x = 1:n
        push!(s_stack, s)
        s = 0.0
        for i_y = 1:2
            w = m0 + i_y
            __idx_tripcount_stack_1_1 = prefix_tripcount_stack_1[(i_x - 1) + 1] + ((i_y - 1) + 1)
            tripcount_stack[__idx_tripcount_stack_1_1] = w
            for i_j = 1:w
                push!(s_stack, s)
                s = s + x[i_j] * u[i_x]
            end
            push!(s_stack, s)
        end
        out[i_x] = s * s
        push!(s_stack, s)
    end
    push!(s_stack, s)
    s = pop!(s_stack)
    for i_x = n:-1:1
        s = pop!(s_stack)
        sb = sb + s * outb[i_x]
        sb = sb + s * outb[i_x]
        outb[i_x] = 0.0
        for i_y = 2:-1:1
            s = pop!(s_stack)
            w = m0 + i_y
            __idx_tripcount_stack_1_2 = prefix_tripcount_stack_1[(i_x - 1) + 1] + ((i_y - 1) + 1)
            w = tripcount_stack[__idx_tripcount_stack_1_2]
            for i_j = w:-1:1
                s = pop!(s_stack)
                xb[i_j] = xb[i_j] + u[i_x] * sb
                ub[i_x] = ub[i_x] + x[i_j] * sb
            end
        end
        s = pop!(s_stack)
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
