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
        nsub = nsub * 2
        val_nsub_1[(i_seq_stage - 1) + 1] = nsub
        __tot_h_stack_1 = __tot_h_stack_1 + 1
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + 1
        __tot_y_stack_1 = __tot_y_stack_1 + ((1 + (div(nsub - 1, 1) + 1)) + 1)
    end
    h_stack = Vector{Float64}(undef, __tot_h_stack_1 + 1)
    y_stack = Vector{Float64}(undef, __tot_y_stack_1 + 1)
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    return (h_stack, y_stack, tripcount_stack, prefix_h_stack_1, prefix_tripcount_stack_1, prefix_y_stack_1, __tot_h_stack_1, __tot_tripcount_stack_1, __tot_y_stack_1, val_nsub_1)
end

function richardson_substep_b(y_init, y_initb, out, outb, a_coef, a_coefb, dt_stage, dt_stageb, num_stages, h_stack, y_stack, tripcount_stack, prefix_h_stack_1, prefix_tripcount_stack_1, prefix_y_stack_1, __tot_h_stack_1, __tot_tripcount_stack_1, __tot_y_stack_1, val_nsub_1)
    h = 0.0
    y = 0.0
    hb = 0.0
    yb = 0.0
    nsub = 1
    for i_seq_stage = 1:num_stages
        h_stack[prefix_h_stack_1[(i_seq_stage - 1) + 1] + 1] = h
        h = dt_stage / nsub
        y_stack[prefix_y_stack_1[(i_seq_stage - 1) + 1] + 1] = y
        y = y_init
        tripcount_stack[prefix_tripcount_stack_1[(i_seq_stage - 1) + 1] + 1] = nsub
        for i_seq_sub = 1:nsub
            y_stack[(prefix_y_stack_1[(i_seq_stage - 1) + 1] + 1) + ((i_seq_sub - 1) + 1)] = y
            y = y - h * a_coef * y
        end
        out[i_seq_stage] = y
        nsub = nsub * 2
        y_stack[(prefix_y_stack_1[(i_seq_stage - 1) + 1] + (1 + (div(val_nsub_1[(i_seq_stage - 1) + 1] - 1, 1) + 1))) + 1] = y
    end
    h_stack[__tot_h_stack_1 + 1] = h
    y_stack[__tot_y_stack_1 + 1] = y
    h = h_stack[__tot_h_stack_1 + 1]
    y = y_stack[__tot_y_stack_1 + 1]
    for i_seq_stage = num_stages:-1:1
        y = y_stack[(prefix_y_stack_1[(i_seq_stage - 1) + 1] + (1 + (div(val_nsub_1[(i_seq_stage - 1) + 1] - 1, 1) + 1))) + 1]
        yb = yb + outb[i_seq_stage]
        outb[i_seq_stage] = 0.0
        nsub = tripcount_stack[prefix_tripcount_stack_1[(i_seq_stage - 1) + 1] + 1]
        for i_seq_sub = nsub:-1:1
            y = y_stack[(prefix_y_stack_1[(i_seq_stage - 1) + 1] + 1) + ((i_seq_sub - 1) + 1)]
            hb = hb + (a_coef * y) * -yb
            a_coefb = a_coefb + (h * y) * -yb
            yb = yb + (h * a_coef) * -yb
        end
        y = y_stack[prefix_y_stack_1[(i_seq_stage - 1) + 1] + 1]
        y_initb = y_initb + yb
        yb = 0.0
        h = h_stack[prefix_h_stack_1[(i_seq_stage - 1) + 1] + 1]
        dt_stageb = dt_stageb + (1.0 / nsub) * hb
        hb = 0.0
    end
    return (y_initb, a_coefb, dt_stageb)
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
