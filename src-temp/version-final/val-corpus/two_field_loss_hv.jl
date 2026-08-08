function initstacks_two_field_loss_b()
    p_stack = Vector{Float64}()
    q_stack = Vector{Float64}()
    loss_stack = Vector{Float64}()
    return (p_stack, q_stack, loss_stack)
end

function two_field_loss_hv(loss, lossb, u, ub, v, vb, p, pb, q, qb, i_n, lossd, lossbd, ud, ubd, vd, vbd, pd, pbd, qd, qbd, p_stack, q_stack, loss_stack)
    p_stack_d = Vector{Float64}()
    q_stack_d = Vector{Float64}()
    loss_stack_d = Vector{Float64}()
    for i_x = 1:i_n
        push!(p_stack_d, pd[i_x])
        push!(p_stack, p[i_x])
        pd[i_x] = (2 * u[i_x]) * ud[i_x]
        p[i_x] = u[i_x] ^ 2
    end
    for i_x = 1:i_n
        push!(q_stack_d, qd[i_x])
        push!(q_stack, q[i_x])
        qd[i_x] = (3 * v[i_x] ^ 2) * vd[i_x]
        q[i_x] = v[i_x] ^ 3
    end
    for i_seq_x = 1:i_n
        push!(loss_stack_d, lossd[1])
        push!(loss_stack, loss[1])
        lossd[1] = (lossd[1] + pd[i_seq_x]) + qd[i_seq_x]
        loss[1] = loss[1] + p[i_seq_x] + q[i_seq_x]
    end
    for i_seq_x = i_n:-1:1
        lossd[1] = pop!(loss_stack_d)
        loss[1] = pop!(loss_stack)
        pbd[i_seq_x] = pbd[i_seq_x] + lossbd[1]
        pb[i_seq_x] = pb[i_seq_x] + lossb[1]
        qbd[i_seq_x] = qbd[i_seq_x] + lossbd[1]
        qb[i_seq_x] = qb[i_seq_x] + lossb[1]
    end
    for i_x = i_n:-1:1
        qd[i_x] = pop!(q_stack_d)
        q[i_x] = pop!(q_stack)
        vbd[i_x] = vbd[i_x] + (qb[i_x] * (3 * ((2 * v[i_x]) * vd[i_x])) + (3 * v[i_x] ^ 2) * qbd[i_x])
        vb[i_x] = vb[i_x] + (3 * v[i_x] ^ 2) * qb[i_x]
        qbd[i_x] = 0.0
        qb[i_x] = 0.0
    end
    for i_x = i_n:-1:1
        pd[i_x] = pop!(p_stack_d)
        p[i_x] = pop!(p_stack)
        ubd[i_x] = ubd[i_x] + (pb[i_x] * (2 * ud[i_x]) + (2 * u[i_x]) * pbd[i_x])
        ub[i_x] = ub[i_x] + (2 * u[i_x]) * pb[i_x]
        pbd[i_x] = 0.0
        pb[i_x] = 0.0
    end
    return nothing
end

function two_field_loss(loss, u, v, p, q, i_n)
    #= none:1 =#
    #= none:2 =#
    for i_x = 1:i_n
        #= none:3 =#
        p[i_x] = u[i_x] ^ 2
        #= none:4 =#
    end
    #= none:5 =#
    for i_x = 1:i_n
        #= none:6 =#
        q[i_x] = v[i_x] ^ 3
        #= none:7 =#
    end
    #= none:8 =#
    for i_seq_x = 1:i_n
        #= none:9 =#
        loss[1] = loss[1] + p[i_seq_x] + q[i_seq_x]
        #= none:10 =#
    end
end
