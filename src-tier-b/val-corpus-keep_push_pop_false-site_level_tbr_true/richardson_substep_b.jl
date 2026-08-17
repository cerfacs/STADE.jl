function initstacks_richardson_substep_b(num_stages)
    h_stack = Vector{Float64}(undef, (div(num_stages - 1, 1) + 1) + 1)
    y_stack = Vector{Float64}()
    tripcount_stack = Vector{Int64}(undef, div(num_stages - 1, 1) + 1)
    return (h_stack, y_stack, tripcount_stack)
end

function richardson_substep_b(y_init, y_initb, out, outb, a_coef, a_coefb, dt_stage, dt_stageb, num_stages, h_stack, y_stack, tripcount_stack)
    h = 0.0
    y = 0.0
    hb = 0.0
    yb = 0.0
    nsub = 1
    for i_seq_stage = 1:num_stages
        h_stack[(i_seq_stage - 1) + 1] = h
        h = dt_stage / nsub
        push!(y_stack, y)
        y = y_init
        tripcount_stack[(i_seq_stage - 1) + 1] = nsub
        for i_seq_sub = 1:nsub
            push!(y_stack, y)
            y = y - h * a_coef * y
        end
        out[i_seq_stage] = y
        nsub = nsub * 2
        push!(y_stack, y)
    end
    h_stack[(div(num_stages - 1, 1) + 1) + 1] = h
    push!(y_stack, y)
    h = h_stack[(div(num_stages - 1, 1) + 1) + 1]
    y = pop!(y_stack)
    for i_seq_stage = num_stages:-1:1
        y = pop!(y_stack)
        yb = yb + outb[i_seq_stage]
        outb[i_seq_stage] = 0.0
        nsub = tripcount_stack[(i_seq_stage - 1) + 1]
        for i_seq_sub = nsub:-1:1
            y = pop!(y_stack)
            hb = hb + (a_coef * y) * -yb
            a_coefb = a_coefb + (h * y) * -yb
            yb = yb + (h * a_coef) * -yb
        end
        y = pop!(y_stack)
        y_initb = y_initb + yb
        yb = 0.0
        h = h_stack[(i_seq_stage - 1) + 1]
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
