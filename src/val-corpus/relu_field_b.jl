function initstacks_relu_field_b(i_n)
    branch_stack = Vector{Int64}(undef, div(i_n - 1, 1) + 1)
    return branch_stack
end

function relu_field_b(loss, lossb, u, ub, v, vb, i_n, branch_stack)
    for i_x = 1:i_n
        if u[i_x] > 0.0
            branch_stack[(i_x - 1) + 1] = 1
            v[i_x] = u[i_x] ^ 2
        else
            branch_stack[(i_x - 1) + 1] = 0
            v[i_x] = 0.0
        end
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + v[i_seq_x]
    end
    for i_seq_x = i_n:-1:1
        vb[i_seq_x] = vb[i_seq_x] + lossb[1]
    end
    for i_x = i_n:-1:1
        __branch = branch_stack[(i_x - 1) + 1]
        if __branch == 1
            ub[i_x] = ub[i_x] + (2 * u[i_x]) * vb[i_x]
            vb[i_x] = 0.0
        else
            vb[i_x] = 0.0
        end
    end
    return nothing
end

function relu_field(loss, u, v, i_n)
    for i_x = 1:i_n
        if u[i_x] > 0.0
            v[i_x] = u[i_x] ^ 2
        else
            v[i_x] = 0.0
        end
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + v[i_seq_x]
    end
end
