function initstacks_raggedind_b(m0, n)
    prefix_s_stack_1 = Vector{Int}(undef, div(n - 1, 1) + 1)
    __tot_s_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, div(n - 1, 1) + 1)
    __tot_tripcount_stack_1 = 0
    val_w_1 = Vector{Int64}(undef, div(n - 1, 1) + 1)
    for i_x = 1:n
        prefix_s_stack_1[(i_x - 1) + 1] = __tot_s_stack_1
        prefix_tripcount_stack_1[(i_x - 1) + 1] = __tot_tripcount_stack_1
        w = m0 + i_x
        s = 0.0
        val_w_1[(i_x - 1) + 1] = w
        __tot_s_stack_1 = __tot_s_stack_1 + ((1 + (div(w - 1, 1) + 1)) + 1)
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + 1
    end
    s_stack = Vector{Float64}(undef, __tot_s_stack_1 + 1)
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    return (s_stack, tripcount_stack, prefix_s_stack_1, prefix_tripcount_stack_1, __tot_s_stack_1, __tot_tripcount_stack_1, val_w_1)
end

function raggedind_hv(out, outb, u, ub, x, xb, n, m0, outd, outbd, ud, ubd, xd, xbd, s_stack, tripcount_stack, prefix_s_stack_1, prefix_tripcount_stack_1, __tot_s_stack_1, __tot_tripcount_stack_1, val_w_1)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    for i_x = 1:n
        w = m0 + i_x
        __idx_s_stack_1_1 = prefix_s_stack_1[(i_x - 1) + 1] + 1
        s_stack_d[__idx_s_stack_1_1] = sd
        s_stack[__idx_s_stack_1_1] = s
        sd = 0.0
        s = 0.0
        __idx_tripcount_stack_1_4 = prefix_tripcount_stack_1[(i_x - 1) + 1] + 1
        tripcount_stack[__idx_tripcount_stack_1_4] = w
        for i_j = 1:w
            __idx_s_stack_1_0 = (prefix_s_stack_1[(i_x - 1) + 1] + 1) + ((i_j - 1) + 1)
            s_stack_d[__idx_s_stack_1_0] = sd
            s_stack[__idx_s_stack_1_0] = s
            sd = sd + (u[i_x] * xd[i_j] + x[i_j] * ud[i_x])
            s = s + x[i_j] * u[i_x]
        end
        outd[i_x] = s * sd + s * sd
        out[i_x] = s * s
        __idx_s_stack_1_8 = (prefix_s_stack_1[(i_x - 1) + 1] + (1 + (div(val_w_1[(i_x - 1) + 1] - 1, 1) + 1))) + 1
        s_stack_d[__idx_s_stack_1_8] = sd
        s_stack[__idx_s_stack_1_8] = s
    end
    s_stack_d[__tot_s_stack_1 + 1] = sd
    s_stack[__tot_s_stack_1 + 1] = s
    sd = s_stack_d[__tot_s_stack_1 + 1]
    s = s_stack[__tot_s_stack_1 + 1]
    for i_x = n:-1:1
        __idx_s_stack_1_0 = (prefix_s_stack_1[(i_x - 1) + 1] + (1 + (div(val_w_1[(i_x - 1) + 1] - 1, 1) + 1))) + 1
        sd = s_stack_d[__idx_s_stack_1_0]
        s = s_stack[__idx_s_stack_1_0]
        w = m0 + i_x
        sbd = sbd + (outb[i_x] * sd + s * outbd[i_x])
        sb = sb + s * outb[i_x]
        sbd = sbd + (outb[i_x] * sd + s * outbd[i_x])
        sb = sb + s * outb[i_x]
        outbd[i_x] = 0.0
        outb[i_x] = 0.0
        __idx_tripcount_stack_1_6 = prefix_tripcount_stack_1[(i_x - 1) + 1] + 1
        w = tripcount_stack[__idx_tripcount_stack_1_6]
        for i_j = w:-1:1
            __idx_s_stack_1_0 = (prefix_s_stack_1[(i_x - 1) + 1] + 1) + ((i_j - 1) + 1)
            sd = s_stack_d[__idx_s_stack_1_0]
            s = s_stack[__idx_s_stack_1_0]
            xbd[i_j] = xbd[i_j] + (sb * ud[i_x] + u[i_x] * sbd)
            xb[i_j] = xb[i_j] + u[i_x] * sb
            ubd[i_x] = ubd[i_x] + (sb * xd[i_j] + x[i_j] * sbd)
            ub[i_x] = ub[i_x] + x[i_j] * sb
        end
        __idx_s_stack_1_0 = prefix_s_stack_1[(i_x - 1) + 1] + 1
        sd = s_stack_d[__idx_s_stack_1_0]
        s = s_stack[__idx_s_stack_1_0]
        sbd = 0.0
        sb = 0.0
    end
    return nothing
end

function raggedind(out, u, x, n, m0)
    for i_x = 1:n
        w = m0 + i_x
        s = 0.0
        for i_j = 1:w
            s = s + x[i_j] * u[i_x]
        end
        out[i_x] = s * s
    end
end
