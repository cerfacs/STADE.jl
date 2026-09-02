function initstacks_richardson_substep_b(dt_stage, num_stages, y_init)
    nsub = 1
    prefix_h_stack_1 = Vector{Int}(undef, max(0, div(num_stages - 1, 1) + 1))
    __tot_h_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div(num_stages - 1, 1) + 1))
    __tot_tripcount_stack_1 = 0
    prefix_y_stack_1 = Vector{Int}(undef, max(0, div(num_stages - 1, 1) + 1))
    __tot_y_stack_1 = 0
    val_nsub_1 = Vector{Int64}(undef, max(0, div(num_stages - 1, 1) + 1))
    for i_stage = 1:num_stages
        prefix_h_stack_1[(i_stage - 1) + 1] = __tot_h_stack_1
        prefix_tripcount_stack_1[(i_stage - 1) + 1] = __tot_tripcount_stack_1
        prefix_y_stack_1[(i_stage - 1) + 1] = __tot_y_stack_1
        h = dt_stage / nsub
        y = y_init
        val_nsub_1[(i_stage - 1) + 1] = nsub
        __tot_h_stack_1 = __tot_h_stack_1 + 1
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + 1
        __tot_y_stack_1 = __tot_y_stack_1 + (max(0, div(nsub - 1, 1) + 1) + 1)
        nsub = nsub * 2
    end
    h_stack = Vector{Float64}(undef, __tot_h_stack_1 + 1)
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    y_stack = Vector{Float64}(undef, __tot_y_stack_1)
    return (h_stack, tripcount_stack, y_stack, prefix_h_stack_1, prefix_tripcount_stack_1, prefix_y_stack_1, __tot_h_stack_1, __tot_tripcount_stack_1, __tot_y_stack_1, val_nsub_1)
end

function richardson_substep_hv(y_init, y_initb, out, outb, a_coef, a_coefb, dt_stage, dt_stageb, num_stages, y_initd, y_initbd, outd, outbd, a_coefd, a_coefbd, dt_staged, dt_stagebd, h_stack, tripcount_stack, y_stack, prefix_h_stack_1, prefix_tripcount_stack_1, prefix_y_stack_1, __tot_h_stack_1, __tot_tripcount_stack_1, __tot_y_stack_1, val_nsub_1)
    h_stack_d = Vector{Float64}(undef, length(h_stack))
    y_stack_d = Vector{Float64}(undef, length(y_stack))
    h = 0.0
    y = 0.0
    hb = 0.0
    yb = 0.0
    hd = 0.0
    hbd = 0.0
    yd = 0.0
    ybd = 0.0
    nsub = 1
    for i_stage = 1:num_stages
        __idx_h_stack_1_0 = prefix_h_stack_1[(i_stage - 1) + 1] + 1
        h_stack_d[__idx_h_stack_1_0] = hd
        h_stack[__idx_h_stack_1_0] = h
        hd = (1.0 / nsub) * dt_staged
        h = dt_stage / nsub
        yd = y_initd
        y = y_init
        __idx_tripcount_stack_1_4 = prefix_tripcount_stack_1[(i_stage - 1) + 1] + 1
        tripcount_stack[__idx_tripcount_stack_1_4] = nsub
        for i_sub = 1:nsub
            __idx_y_stack_1_0 = prefix_y_stack_1[(i_stage - 1) + 1] + ((i_sub - 1) + 1)
            y_stack_d[__idx_y_stack_1_0] = yd
            y_stack[__idx_y_stack_1_0] = y
            yd = yd + -((((a_coef * y) * hd + (h * y) * a_coefd) + (h * a_coef) * yd))
            y = y - h * a_coef * y
        end
        outd[i_stage] = yd
        out[i_stage] = y
        nsub = nsub * 2
        __idx_y_stack_1_9 = (prefix_y_stack_1[(i_stage - 1) + 1] + max(0, div(val_nsub_1[(i_stage - 1) + 1] - 1, 1) + 1)) + 1
        y_stack_d[__idx_y_stack_1_9] = yd
        y_stack[__idx_y_stack_1_9] = y
    end
    __ihcse_0 = __tot_h_stack_1 + 1
    __idx_h_stack_2 = __ihcse_0
    h_stack_d[__idx_h_stack_2] = hd
    h_stack[__idx_h_stack_2] = h
    __idx_h_stack_0 = __ihcse_0
    hd = h_stack_d[__idx_h_stack_0]
    h = h_stack[__idx_h_stack_0]
    for i_stage = num_stages:-1:1
        __idx_y_stack_1_0 = (prefix_y_stack_1[(i_stage - 1) + 1] + max(0, div(val_nsub_1[(i_stage - 1) + 1] - 1, 1) + 1)) + 1
        yd = y_stack_d[__idx_y_stack_1_0]
        y = y_stack[__idx_y_stack_1_0]
        __oldb_0d = outbd[i_stage]
        __oldb_0 = outb[i_stage]
        outbd[i_stage] = 0.0
        outb[i_stage] = 0.0
        ybd = ybd + __oldb_0d
        yb = yb + __oldb_0
        __idx_tripcount_stack_1_5 = prefix_tripcount_stack_1[(i_stage - 1) + 1] + 1
        nsub = tripcount_stack[__idx_tripcount_stack_1_5]
        for i_sub = nsub:-1:1
            __idx_y_stack_1_0 = prefix_y_stack_1[(i_stage - 1) + 1] + ((i_sub - 1) + 1)
            yd = y_stack_d[__idx_y_stack_1_0]
            y = y_stack[__idx_y_stack_1_0]
            __oldb_2d = ybd
            __oldb_2 = yb
            ybd = 0.0
            yb = 0.0
            ybd = ybd + __oldb_2d
            yb = yb + __oldb_2
            __cse_0d = -__oldb_2d
            __cse_0 = -__oldb_2
            __hcse_1 = a_coef * y
            hbd = hbd + (__cse_0 * (y * a_coefd + a_coef * yd) + __hcse_1 * __cse_0d)
            hb = hb + __hcse_1 * __cse_0
            __hcse_2 = h * y
            a_coefbd = a_coefbd + (__cse_0 * (y * hd + h * yd) + __hcse_2 * __cse_0d)
            a_coefb = a_coefb + __hcse_2 * __cse_0
            __hcse_3 = h * a_coef
            ybd = ybd + (__cse_0 * (a_coef * hd + h * a_coefd) + __hcse_3 * __cse_0d)
            yb = yb + __hcse_3 * __cse_0
        end
        __oldb_0d = ybd
        __oldb_0 = yb
        ybd = 0.0
        yb = 0.0
        y_initbd = y_initbd + __oldb_0d
        y_initb = y_initb + __oldb_0
        __idx_h_stack_1_0 = prefix_h_stack_1[(i_stage - 1) + 1] + 1
        hd = h_stack_d[__idx_h_stack_1_0]
        h = h_stack[__idx_h_stack_1_0]
        __oldb_2d = hbd
        __oldb_2 = hb
        hbd = 0.0
        hb = 0.0
        __ihcse_4 = 1.0 / nsub
        dt_stagebd = dt_stagebd + __ihcse_4 * __oldb_2d
        dt_stageb = dt_stageb + __ihcse_4 * __oldb_2
    end
    return (y_initb, y_initbd, a_coefb, a_coefbd, dt_stageb, dt_stagebd)
end

function richardson_substep(y_init, out, a_coef, dt_stage, num_stages)
    nsub = 1
    for i_stage = 1:num_stages
        h = dt_stage / nsub
        y = y_init
        for i_sub = 1:nsub
            y = y - h * a_coef * y
        end
        out[i_stage] = y
        nsub = nsub * 2
    end
end
