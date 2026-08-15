function initstacks_normcomp_b(i_n)
    w_stack = Vector{Float64}(undef, div(i_n - 1, 1) + 1)
    return w_stack
end

function normcomp_hv(loss, lossb, u, ub, v, vb, w, wb, i_n, lossd, lossbd, ud, ubd, vd, vbd, wd, wbd, w_stack)
    w_stack_d = Vector{Float64}(undef, div(i_n - 1, 1) + 1)
    for i_x = 1:i_n
        w_stack_d[(i_x - 1) + 1] = wd[i_x]
        w_stack[(i_x - 1) + 1] = w[i_x]
        wd[i_x] = ud[i_x] + -(vd[i_x])
        w[i_x] = u[i_x] - v[i_x]
    end
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + (2 * w[i_seq_x]) * wd[i_seq_x]
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
    for i_seq_x = i_n:-1:1
        wbd[i_seq_x] = wbd[i_seq_x] + (lossb[1] * (2 * wd[i_seq_x]) + (2 * w[i_seq_x]) * lossbd[1])
        wb[i_seq_x] = wb[i_seq_x] + (2 * w[i_seq_x]) * lossb[1]
    end
    for i_x = i_n:-1:1
        wd[i_x] = w_stack_d[(i_x - 1) + 1]
        w[i_x] = w_stack[(i_x - 1) + 1]
        ubd[i_x] = ubd[i_x] + wbd[i_x]
        ub[i_x] = ub[i_x] + wb[i_x]
        vbd[i_x] = vbd[i_x] + -(wbd[i_x])
        vb[i_x] = vb[i_x] + -(wb[i_x])
        wbd[i_x] = 0.0
        wb[i_x] = 0.0
    end
    return nothing
end

function normcomp(loss, u, v, w, i_n)
    for i_x = 1:i_n
        w[i_x] = u[i_x] - v[i_x]
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
end
