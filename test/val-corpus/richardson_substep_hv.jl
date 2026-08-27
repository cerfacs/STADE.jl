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
    h_stack_d[__tot_h_stack_1 + 1] = hd
    h_stack[__tot_h_stack_1 + 1] = h
    hd = h_stack_d[__tot_h_stack_1 + 1]
    h = h_stack[__tot_h_stack_1 + 1]
    for i_stage = num_stages:-1:1
        __idx_y_stack_1_0 = (prefix_y_stack_1[(i_stage - 1) + 1] + max(0, div(val_nsub_1[(i_stage - 1) + 1] - 1, 1) + 1)) + 1
        yd = y_stack_d[__idx_y_stack_1_0]
        y = y_stack[__idx_y_stack_1_0]
        ybd = ybd + outbd[i_stage]
        yb = yb + outb[i_stage]
        outbd[i_stage] = 0.0
        outb[i_stage] = 0.0
        __idx_tripcount_stack_1_4 = prefix_tripcount_stack_1[(i_stage - 1) + 1] + 1
        nsub = tripcount_stack[__idx_tripcount_stack_1_4]
        for i_sub = nsub:-1:1
            __idx_y_stack_1_0 = prefix_y_stack_1[(i_stage - 1) + 1] + ((i_sub - 1) + 1)
            yd = y_stack_d[__idx_y_stack_1_0]
            y = y_stack[__idx_y_stack_1_0]
            hbd = hbd + (-yb * (y * a_coefd + a_coef * yd) + (a_coef * y) * -ybd)
            hb = hb + (a_coef * y) * -yb
            a_coefbd = a_coefbd + (-yb * (y * hd + h * yd) + (h * y) * -ybd)
            a_coefb = a_coefb + (h * y) * -yb
            ybd = ybd + (-yb * (a_coef * hd + h * a_coefd) + (h * a_coef) * -ybd)
            yb = yb + (h * a_coef) * -yb
        end
        y_initbd = y_initbd + ybd
        y_initb = y_initb + yb
        ybd = 0.0
        yb = 0.0
        __idx_h_stack_1_0 = prefix_h_stack_1[(i_stage - 1) + 1] + 1
        hd = h_stack_d[__idx_h_stack_1_0]
        h = h_stack[__idx_h_stack_1_0]
        dt_stagebd = dt_stagebd + (1.0 / nsub) * hbd
        dt_stageb = dt_stageb + (1.0 / nsub) * hb
        hbd = 0.0
        hb = 0.0
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
