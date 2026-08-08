function initstacks_clamped_sumsq_b()
    branch_stack = Int[]
    return branch_stack
end
function clamped_sumsq_b(loss, lossb, u, ub, i_n, branch_stack)
    for i_seq_x = 1:i_n
        if u[i_seq_x] > 0.0
            push!(branch_stack, 0)
        else
            push!(branch_stack, 1)
        end
    end
    for i_seq_x = i_n:-1:1
        wb = lossb[1]
        branch = pop!(branch_stack)
        if branch == 0
            ub[i_seq_x] = ub[i_seq_x] + 2 * u[i_seq_x] * wb
        end
    end
    return 
end
function clamped_sumsq(loss, u, i_n)
    for i_seq_x = 1:i_n
        if u[i_seq_x] > 0.0
            w = u[i_seq_x] ^ 2
        else
            w = 0.0
        end
        loss[1] = loss[1] + w
    end
end
