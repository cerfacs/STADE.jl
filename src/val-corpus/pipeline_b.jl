function initstacks_pipeline_b()
    v_stack = Vector{Float64}()
    return v_stack
end

function pipeline_b(loss, lossb, u, ub, v, vb, w, wb, i_n, v_stack)
    for i_x = 1:i_n
        push!(v_stack, v[i_x])
        v[i_x] = u[i_x] ^ 2 + 1.0
    end
    for i_x = 1:i_n
        w[i_x] = v[i_x] * u[i_x]
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x]
    end
    for i_seq_x = i_n:-1:1
        wb[i_seq_x] = wb[i_seq_x] + lossb[1]
    end
    for i_x = 1:i_n
        vb[i_x] = vb[i_x] + u[i_x] * wb[i_x]
        ub[i_x] = ub[i_x] + v[i_x] * wb[i_x]
        wb[i_x] = 0.0
    end
    for i_x = i_n:-1:1
        v[i_x] = pop!(v_stack)
        ub[i_x] = ub[i_x] + (2 * u[i_x]) * vb[i_x]
        vb[i_x] = 0.0
    end
    return nothing
end

function pipeline(loss, u, v, w, i_n)
    for i_x = 1:i_n
        v[i_x] = u[i_x] ^ 2 + 1.0
    end
    for i_x = 1:i_n
        w[i_x] = v[i_x] * u[i_x]
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x]
    end
end
