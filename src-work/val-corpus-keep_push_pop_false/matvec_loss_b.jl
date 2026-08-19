function initstacks_matvec_loss_b(i_m, i_n)
    v_stack = Vector{Float64}(undef, (div(i_m - 1, 1) + 1) * (div(i_n - 1, 1) + 1))
    return v_stack
end

function matvec_loss_b(loss, lossb, a, ab, u, ub, v, vb, i_m, i_n, v_stack)
    for i_i = 1:i_m
        for i_seq_j = 1:i_n
            v_stack[((i_i - 1) * (div(i_n - 1, 1) + 1) + (i_seq_j - 1)) + 1] = v[i_i]
            v[i_i] = v[i_i] + a[i_i, i_seq_j] * u[i_seq_j]
        end
    end
    for i_seq_i = 1:i_m
        loss[1] = loss[1] + v[i_seq_i] ^ 2
    end
    for i_seq_i = i_m:-1:1
        vb[i_seq_i] = vb[i_seq_i] + (2 * v[i_seq_i]) * lossb[1]
    end
    for i_i = i_m:-1:1
        for i_seq_j = i_n:-1:1
            v[i_i] = v_stack[((i_i - 1) * (div(i_n - 1, 1) + 1) + (i_seq_j - 1)) + 1]
            ab[i_i, i_seq_j] = ab[i_i, i_seq_j] + u[i_seq_j] * vb[i_i]
            ub[i_seq_j] = ub[i_seq_j] + a[i_i, i_seq_j] * vb[i_i]
        end
    end
    return nothing
end

function matvec_loss(loss, a, u, v, i_m, i_n)
    for i_i = 1:i_m
        for i_seq_j = 1:i_n
            v[i_i] = v[i_i] + a[i_i, i_seq_j] * u[i_seq_j]
        end
    end
    for i_seq_i = 1:i_m
        loss[1] = loss[1] + v[i_seq_i] ^ 2
    end
end
