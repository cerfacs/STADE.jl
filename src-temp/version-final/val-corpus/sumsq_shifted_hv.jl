function initstacks_sumsq_shifted_b()
    loss_stack = Vector{Float64}()
    return loss_stack
end

function sumsq_shifted_hv(loss, lossb, u, ub, alpha, alphab, beta, betab, i_n, lossd, lossbd, ud, ubd, alphad, alphabd, betad, betabd, loss_stack)
    loss_stack_d = Vector{Float64}()
    for i_seq_x = 1:i_n
        push!(loss_stack_d, lossd[1])
        push!(loss_stack, loss[1])
        lossd[1] = lossd[1] + (2 * (alpha * u[i_seq_x] + beta)) * ((u[i_seq_x] * alphad + alpha * ud[i_seq_x]) + betad)
        loss[1] = loss[1] + (alpha * u[i_seq_x] + beta) ^ 2
    end
    for i_seq_x = i_n:-1:1
        lossd[1] = pop!(loss_stack_d)
        loss[1] = pop!(loss_stack)
        alphabd = alphabd + (((2 * (alpha * u[i_seq_x] + beta)) * lossb[1]) * ud[i_seq_x] + u[i_seq_x] * (lossb[1] * (2 * ((u[i_seq_x] * alphad + alpha * ud[i_seq_x]) + betad)) + (2 * (alpha * u[i_seq_x] + beta)) * lossbd[1]))
        alphab = alphab + u[i_seq_x] * ((2 * (alpha * u[i_seq_x] + beta)) * lossb[1])
        ubd[i_seq_x] = ubd[i_seq_x] + (((2 * (alpha * u[i_seq_x] + beta)) * lossb[1]) * alphad + alpha * (lossb[1] * (2 * ((u[i_seq_x] * alphad + alpha * ud[i_seq_x]) + betad)) + (2 * (alpha * u[i_seq_x] + beta)) * lossbd[1]))
        ub[i_seq_x] = ub[i_seq_x] + alpha * ((2 * (alpha * u[i_seq_x] + beta)) * lossb[1])
        betabd = betabd + (lossb[1] * (2 * ((u[i_seq_x] * alphad + alpha * ud[i_seq_x]) + betad)) + (2 * (alpha * u[i_seq_x] + beta)) * lossbd[1])
        betab = betab + (2 * (alpha * u[i_seq_x] + beta)) * lossb[1]
    end
    return (alphab, alphabd, betab, betabd)
end

function sumsq_shifted(loss, u, alpha, beta, i_n)
    #= none:1 =#
    #= none:2 =#
    for i_seq_x = 1:i_n
        #= none:3 =#
        loss[1] = loss[1] + (alpha * u[i_seq_x] + beta) ^ 2
        #= none:4 =#
    end
end
