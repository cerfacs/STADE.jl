function initstacks_richardson_substep_b(dt_stage, num_stages, y_init)
    nsub = 1
    prefix_h_stack_1 = Vector{Int}(undef, div(num_stages - 1, 1) + 1)
    __tot_h_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, div(num_stages - 1, 1) + 1)
    __tot_tripcount_stack_1 = 0
    prefix_y_stack_1 = Vector{Int}(undef, div(num_stages - 1, 1) + 1)
    __tot_y_stack_1 = 0
    val_nsub_1 = Vector{Int64}(undef, div(num_stages - 1, 1) + 1)
    for i_seq_stage = 1:num_stages
        prefix_h_stack_1[(i_seq_stage - 1) + 1] = __tot_h_stack_1
        prefix_tripcount_stack_1[(i_seq_stage - 1) + 1] = __tot_tripcount_stack_1
        prefix_y_stack_1[(i_seq_stage - 1) + 1] = __tot_y_stack_1
        h = dt_stage / nsub
        y = y_init
        val_nsub_1[(i_seq_stage - 1) + 1] = nsub
        __tot_h_stack_1 = __tot_h_stack_1 + 1
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + 1
        __tot_y_stack_1 = __tot_y_stack_1 + ((1 + (div(nsub - 1, 1) + 1)) + 1)
        nsub = nsub * 2
    end
    h_stack = Vector{Float64}(undef, __tot_h_stack_1 + 1)
    y_stack = Vector{Float64}(undef, __tot_y_stack_1 + 1)
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    return (h_stack, y_stack, tripcount_stack, prefix_h_stack_1, prefix_tripcount_stack_1, prefix_y_stack_1, __tot_h_stack_1, __tot_tripcount_stack_1, __tot_y_stack_1, val_nsub_1)
end

function richardson_substep_hv(y_init, y_initb, out, outb, a_coef, a_coefb, dt_stage, dt_stageb, num_stages, y_initd, y_initbd, outd, outbd, a_coefd, a_coefbd, dt_staged, dt_stagebd, h_stack, y_stack, tripcount_stack, prefix_h_stack_1, prefix_tripcount_stack_1, prefix_y_stack_1, __tot_h_stack_1, __tot_tripcount_stack_1, __tot_y_stack_1, val_nsub_1)
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
    for i_seq_stage = 1:num_stages
        h_stack_d[prefix_h_stack_1[(i_seq_stage - 1) + 1] + 1] = hd
        h_stack[prefix_h_stack_1[(i_seq_stage - 1) + 1] + 1] = h
        hd = (1.0 / nsub) * dt_staged
        h = dt_stage / nsub
        y_stack_d[prefix_y_stack_1[(i_seq_stage - 1) + 1] + 1] = yd
        y_stack[prefix_y_stack_1[(i_seq_stage - 1) + 1] + 1] = y
        yd = y_initd
        y = y_init
        tripcount_stack[prefix_tripcount_stack_1[(i_seq_stage - 1) + 1] + 1] = nsub
        for i_seq_sub = 1:nsub
            y_stack_d[(prefix_y_stack_1[(i_seq_stage - 1) + 1] + 1) + ((i_seq_sub - 1) + 1)] = yd
            y_stack[(prefix_y_stack_1[(i_seq_stage - 1) + 1] + 1) + ((i_seq_sub - 1) + 1)] = y
            yd = yd + -((((a_coef * y) * hd + (h * y) * a_coefd) + (h * a_coef) * yd))
            y = y - h * a_coef * y
        end
        outd[i_seq_stage] = yd
        out[i_seq_stage] = y
        nsub = nsub * 2
        y_stack_d[(prefix_y_stack_1[(i_seq_stage - 1) + 1] + (1 + (div(val_nsub_1[(i_seq_stage - 1) + 1] - 1, 1) + 1))) + 1] = yd
        y_stack[(prefix_y_stack_1[(i_seq_stage - 1) + 1] + (1 + (div(val_nsub_1[(i_seq_stage - 1) + 1] - 1, 1) + 1))) + 1] = y
    end
    h_stack_d[__tot_h_stack_1 + 1] = hd
    h_stack[__tot_h_stack_1 + 1] = h
    y_stack_d[__tot_y_stack_1 + 1] = yd
    y_stack[__tot_y_stack_1 + 1] = y
    hd = h_stack_d[__tot_h_stack_1 + 1]
    h = h_stack[__tot_h_stack_1 + 1]
    yd = y_stack_d[__tot_y_stack_1 + 1]
    y = y_stack[__tot_y_stack_1 + 1]
    for i_seq_stage = num_stages:-1:1
        yd = y_stack_d[(prefix_y_stack_1[(i_seq_stage - 1) + 1] + (1 + (div(val_nsub_1[(i_seq_stage - 1) + 1] - 1, 1) + 1))) + 1]
        y = y_stack[(prefix_y_stack_1[(i_seq_stage - 1) + 1] + (1 + (div(val_nsub_1[(i_seq_stage - 1) + 1] - 1, 1) + 1))) + 1]
        ybd = ybd + outbd[i_seq_stage]
        yb = yb + outb[i_seq_stage]
        outbd[i_seq_stage] = 0.0
        outb[i_seq_stage] = 0.0
        nsub = tripcount_stack[prefix_tripcount_stack_1[(i_seq_stage - 1) + 1] + 1]
        for i_seq_sub = nsub:-1:1
            yd = y_stack_d[(prefix_y_stack_1[(i_seq_stage - 1) + 1] + 1) + ((i_seq_sub - 1) + 1)]
            y = y_stack[(prefix_y_stack_1[(i_seq_stage - 1) + 1] + 1) + ((i_seq_sub - 1) + 1)]
            hbd = hbd + (-yb * (y * a_coefd + a_coef * yd) + (a_coef * y) * -ybd)
            hb = hb + (a_coef * y) * -yb
            a_coefbd = a_coefbd + (-yb * (y * hd + h * yd) + (h * y) * -ybd)
            a_coefb = a_coefb + (h * y) * -yb
            ybd = ybd + (-yb * (a_coef * hd + h * a_coefd) + (h * a_coef) * -ybd)
            yb = yb + (h * a_coef) * -yb
        end
        yd = y_stack_d[prefix_y_stack_1[(i_seq_stage - 1) + 1] + 1]
        y = y_stack[prefix_y_stack_1[(i_seq_stage - 1) + 1] + 1]
        y_initbd = y_initbd + ybd
        y_initb = y_initb + yb
        ybd = 0.0
        yb = 0.0
        hd = h_stack_d[prefix_h_stack_1[(i_seq_stage - 1) + 1] + 1]
        h = h_stack[prefix_h_stack_1[(i_seq_stage - 1) + 1] + 1]
        dt_stagebd = dt_stagebd + (1.0 / nsub) * hbd
        dt_stageb = dt_stageb + (1.0 / nsub) * hb
        hbd = 0.0
        hb = 0.0
    end
    return (y_initb, y_initbd, a_coefb, a_coefbd, dt_stageb, dt_stagebd)
end

function richardson_substep(y_init, out, a_coef, dt_stage, num_stages)
    nsub = 1
    for i_seq_stage = 1:num_stages
        h = dt_stage / nsub
        y = y_init
        for i_seq_sub = 1:nsub
            y = y - h * a_coef * y
        end
        out[i_seq_stage] = y
        nsub = nsub * 2
    end
end
