function initstacks_geomrecur_b()
    u_stack = Vector{Float64}()
    loss_stack = Vector{Float64}()
    return (u_stack, loss_stack)
end

function geomrecur_hv(loss, lossb, u, ub, c, cb, i_n, lossd, lossbd, ud, ubd, cd, cbd, u_stack, loss_stack)
    u_stack_d = Vector{Float64}()
    loss_stack_d = Vector{Float64}()
    for i_seq_x = 2:i_n
        push!(u_stack_d, ud[i_seq_x])
        push!(u_stack, u[i_seq_x])
        ud[i_seq_x] = u[i_seq_x - 1] * cd + c * ud[i_seq_x - 1]
        u[i_seq_x] = c * u[i_seq_x - 1]
    end
    for i_seq_x = 1:i_n
        push!(loss_stack_d, lossd[1])
        push!(loss_stack, loss[1])
        lossd[1] = lossd[1] + (2 * u[i_seq_x]) * ud[i_seq_x]
        loss[1] = loss[1] + u[i_seq_x] ^ 2
    end
    for i_seq_x = i_n:-1:1
        lossd[1] = pop!(loss_stack_d)
        loss[1] = pop!(loss_stack)
        ubd[i_seq_x] = ubd[i_seq_x] + (lossb[1] * (2 * ud[i_seq_x]) + (2 * u[i_seq_x]) * lossbd[1])
        ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * lossb[1]
    end
    for i_seq_x = i_n:-1:2
        ud[i_seq_x] = pop!(u_stack_d)
        u[i_seq_x] = pop!(u_stack)
        cbd = cbd + (ub[i_seq_x] * ud[i_seq_x - 1] + u[i_seq_x - 1] * ubd[i_seq_x])
        cb = cb + u[i_seq_x - 1] * ub[i_seq_x]
        ubd[i_seq_x - 1] = ubd[i_seq_x - 1] + (ub[i_seq_x] * cd + c * ubd[i_seq_x])
        ub[i_seq_x - 1] = ub[i_seq_x - 1] + c * ub[i_seq_x]
        ubd[i_seq_x] = 0.0
        ub[i_seq_x] = 0.0
    end
    return (cb, cbd)
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
