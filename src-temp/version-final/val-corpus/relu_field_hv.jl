function initstacks_relu_field_b()
    branch_stack = Vector{Int64}()
    v_stack = Vector{Float64}()
    loss_stack = Vector{Float64}()
    return (branch_stack, v_stack, loss_stack)
end

function relu_field_hv(loss, lossb, u, ub, v, vb, i_n, lossd, lossbd, ud, ubd, vd, vbd, branch_stack, v_stack, loss_stack)
    v_stack_d = Vector{Float64}()
    loss_stack_d = Vector{Float64}()
    for i_x = 1:i_n
        if u[i_x] > 0.0
            push!(branch_stack, 1)
            push!(v_stack_d, vd[i_x])
            push!(v_stack, v[i_x])
            vd[i_x] = (2 * u[i_x]) * ud[i_x]
            v[i_x] = u[i_x] ^ 2
        else
            push!(branch_stack, 0)
            vd[i_x] = 0.0
            v[i_x] = 0.0
        end
    end
    for i_seq_x = 1:i_n
        push!(loss_stack_d, lossd[1])
        push!(loss_stack, loss[1])
        lossd[1] = lossd[1] + vd[i_seq_x]
        loss[1] = loss[1] + v[i_seq_x]
    end
    for i_seq_x = i_n:-1:1
        lossd[1] = pop!(loss_stack_d)
        loss[1] = pop!(loss_stack)
        vbd[i_seq_x] = vbd[i_seq_x] + lossbd[1]
        vb[i_seq_x] = vb[i_seq_x] + lossb[1]
    end
    for i_x = i_n:-1:1
        __branch = pop!(branch_stack)
        if __branch == 1
            vd[i_x] = pop!(v_stack_d)
            v[i_x] = pop!(v_stack)
            ubd[i_x] = ubd[i_x] + (vb[i_x] * (2 * ud[i_x]) + (2 * u[i_x]) * vbd[i_x])
            ub[i_x] = ub[i_x] + (2 * u[i_x]) * vb[i_x]
            vbd[i_x] = 0.0
            vb[i_x] = 0.0
        else
            vbd[i_x] = 0.0
            vb[i_x] = 0.0
        end
    end
    return nothing
end

function relu_field(loss, u, v, i_n)
    #= none:1 =#
    #= none:2 =#
    for i_x = 1:i_n
        #= none:3 =#
        if u[i_x] > 0.0
            #= none:4 =#
            v[i_x] = u[i_x] ^ 2
        else
            #= none:6 =#
            v[i_x] = 0.0
        end
        #= none:8 =#
    end
    #= none:9 =#
    for i_seq_x = 1:i_n
        #= none:10 =#
        loss[1] = loss[1] + v[i_seq_x]
        #= none:11 =#
    end
end
