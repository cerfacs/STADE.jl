function initstacks_clamped_sumsq_b()
    branch_stack = Vector{Int64}()
    w_stack = Vector{Float64}()
    loss_stack = Vector{Float64}()
    return (branch_stack, w_stack, loss_stack)
end

function clamped_sumsq_b(loss, lossb, u, ub, i_n, branch_stack, w_stack, loss_stack)
    w = 0.0
    wb = 0.0
    for i_seq_x = 1:i_n
        if u[i_seq_x] > 0.0
            push!(branch_stack, 1)
            push!(w_stack, w)
            w = u[i_seq_x] ^ 2
        else
            push!(branch_stack, 0)
            w = 0.0
        end
        push!(loss_stack, loss[1])
        loss[1] = loss[1] + w
    end
    for i_seq_x = i_n:-1:1
        __branch_pre_1 = pop!(branch_stack)
        if __branch_pre_1 == 1
            __snap_discard = pop!(w_stack)
        end
        w = 0.0
        if __branch_pre_1 == 1
            w = u[i_seq_x] ^ 2
        else
            w = 0.0
        end
        loss[1] = pop!(loss_stack)
        wb = wb + lossb[1]
        if __branch_pre_1 == 1
            ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * wb
            wb = 0.0
        else
            wb = 0.0
        end
    end
    return nothing
end

function clamped_sumsq(loss, u, i_n)
    #= none:1 =#
    #= none:2 =#
    for i_seq_x = 1:i_n
        #= none:3 =#
        if u[i_seq_x] > 0.0
            #= none:4 =#
            w = u[i_seq_x] ^ 2
        else
            #= none:6 =#
            w = 0.0
        end
        #= none:8 =#
        loss[1] = loss[1] + w
        #= none:9 =#
    end
end
