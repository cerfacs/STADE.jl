function initstacks_clamped_sumsq_b()
    branch_stack = Vector{Int64}()
    return branch_stack
end

function clamped_sumsq_hv(loss, lossb, u, ub, i_n, lossd, lossbd, ud, ubd, branch_stack)
    w = 0.0
    wb = 0.0
    wd = 0.0
    wbd = 0.0
    for i_seq_x = 1:i_n
        if u[i_seq_x] > 0.0
            push!(branch_stack, 1)
            wd = (2 * u[i_seq_x]) * ud[i_seq_x]
            w = u[i_seq_x] ^ 2
        else
            push!(branch_stack, 0)
            wd = 0.0
            w = 0.0
        end
        lossd[1] = lossd[1] + wd
        loss[1] = loss[1] + w
    end
    for i_seq_x = i_n:-1:1
        wbd = wbd + lossbd[1]
        wb = wb + lossb[1]
        __branch = pop!(branch_stack)
        if __branch == 1
            ubd[i_seq_x] = ubd[i_seq_x] + (wb * (2 * ud[i_seq_x]) + (2 * u[i_seq_x]) * wbd)
            ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * wb
            wbd = 0.0
            wb = 0.0
        else
            wbd = 0.0
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
