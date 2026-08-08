function initstacks_geomrecur_b()
    u_stack = Vector{Float64}()
    loss_stack = Vector{Float64}()
    return (u_stack, loss_stack)
end

function geomrecur_b(loss, lossb, u, ub, c, cb, i_n, u_stack, loss_stack)
    for i_seq_x = 2:i_n
        push!(u_stack, u[i_seq_x])
        u[i_seq_x] = c * u[i_seq_x - 1]
    end
    for i_seq_x = 1:i_n
        push!(loss_stack, loss[1])
        loss[1] = loss[1] + u[i_seq_x] ^ 2
    end
    for i_seq_x = i_n:-1:1
        loss[1] = pop!(loss_stack)
        ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * lossb[1]
    end
    for i_seq_x = i_n:-1:2
        u[i_seq_x] = pop!(u_stack)
        cb = cb + u[i_seq_x - 1] * ub[i_seq_x]
        ub[i_seq_x - 1] = ub[i_seq_x - 1] + c * ub[i_seq_x]
        ub[i_seq_x] = 0.0
    end
    return cb
end

function geomrecur(loss, u, c, i_n)
    #= none:1 =#
    #= none:2 =#
    for i_seq_x = 2:i_n
        #= none:3 =#
        u[i_seq_x] = c * u[i_seq_x - 1]
        #= none:4 =#
    end
    #= none:5 =#
    for i_seq_x = 1:i_n
        #= none:6 =#
        loss[1] = loss[1] + u[i_seq_x] ^ 2
        #= none:7 =#
    end
end
