function initstacks_bilinear_b()
    loss_stack = Vector{Float64}()
    return loss_stack
end

function bilinear_b(loss, lossb, x, xb, a, ab, y, yb, i_m, i_n, loss_stack)
    for i_seq_i = 1:i_m
        for i_seq_j = 1:i_n
            push!(loss_stack, loss[1])
            loss[1] = loss[1] + x[i_seq_i] * a[i_seq_i, i_seq_j] * y[i_seq_j]
        end
    end
    for i_seq_i = i_m:-1:1
        for i_seq_j = i_n:-1:1
            loss[1] = pop!(loss_stack)
            xb[i_seq_i] = xb[i_seq_i] + (a[i_seq_i, i_seq_j] * y[i_seq_j]) * lossb[1]
            ab[i_seq_i, i_seq_j] = ab[i_seq_i, i_seq_j] + (x[i_seq_i] * y[i_seq_j]) * lossb[1]
            yb[i_seq_j] = yb[i_seq_j] + (x[i_seq_i] * a[i_seq_i, i_seq_j]) * lossb[1]
        end
    end
    return nothing
end

function bilinear(loss, x, a, y, i_m, i_n)
    #= none:1 =#
    #= none:2 =#
    for i_seq_i = 1:i_m
        #= none:3 =#
        for i_seq_j = 1:i_n
            #= none:4 =#
            loss[1] = loss[1] + x[i_seq_i] * a[i_seq_i, i_seq_j] * y[i_seq_j]
            #= none:5 =#
        end
        #= none:6 =#
    end
end
