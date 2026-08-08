function initstacks_affine_loss_b()
    v_stack = Vector{Float64}()
    loss_stack = Vector{Float64}()
    return (v_stack, loss_stack)
end

function affine_loss_b(loss, lossb, u, ub, a, ab, b, bb, v, vb, i_n, v_stack, loss_stack)
    for i_x = 1:i_n
        push!(v_stack, v[i_x])
        v[i_x] = a[i_x] * u[i_x] + b[i_x]
    end
    for i_seq_x = 1:i_n
        push!(loss_stack, loss[1])
        loss[1] = loss[1] + v[i_seq_x] ^ 2
    end
    for i_seq_x = i_n:-1:1
        loss[1] = pop!(loss_stack)
        vb[i_seq_x] = vb[i_seq_x] + (2 * v[i_seq_x]) * lossb[1]
    end
    for i_x = i_n:-1:1
        v[i_x] = pop!(v_stack)
        ab[i_x] = ab[i_x] + u[i_x] * vb[i_x]
        ub[i_x] = ub[i_x] + a[i_x] * vb[i_x]
        bb[i_x] = bb[i_x] + vb[i_x]
        vb[i_x] = 0.0
    end
    return nothing
end

function affine_loss(loss, u, a, b, v, i_n)
    #= none:1 =#
    #= none:2 =#
    for i_x = 1:i_n
        #= none:3 =#
        v[i_x] = a[i_x] * u[i_x] + b[i_x]
        #= none:4 =#
    end
    #= none:5 =#
    for i_seq_x = 1:i_n
        #= none:6 =#
        loss[1] = loss[1] + v[i_seq_x] ^ 2
        #= none:7 =#
    end
end
