function initstacks_affine_loss_b()
    v_stack = Vector{Float64}()
    return v_stack
end

function affine_loss_hv(loss, lossb, u, ub, a, ab, b, bb, v, vb, i_n, lossd, lossbd, ud, ubd, ad, abd, bd, bbd, vd, vbd, v_stack)
    v_stack_d = Vector{Float64}()
    for i_x = 1:i_n
        push!(v_stack_d, vd[i_x])
        push!(v_stack, v[i_x])
        vd[i_x] = (u[i_x] * ad[i_x] + a[i_x] * ud[i_x]) + bd[i_x]
        v[i_x] = a[i_x] * u[i_x] + b[i_x]
    end
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + (2 * v[i_seq_x]) * vd[i_seq_x]
        loss[1] = loss[1] + v[i_seq_x] ^ 2
    end
    for i_seq_x = i_n:-1:1
        vbd[i_seq_x] = vbd[i_seq_x] + (lossb[1] * (2 * vd[i_seq_x]) + (2 * v[i_seq_x]) * lossbd[1])
        vb[i_seq_x] = vb[i_seq_x] + (2 * v[i_seq_x]) * lossb[1]
    end
    for i_x = i_n:-1:1
        vd[i_x] = pop!(v_stack_d)
        v[i_x] = pop!(v_stack)
        abd[i_x] = abd[i_x] + (vb[i_x] * ud[i_x] + u[i_x] * vbd[i_x])
        ab[i_x] = ab[i_x] + u[i_x] * vb[i_x]
        ubd[i_x] = ubd[i_x] + (vb[i_x] * ad[i_x] + a[i_x] * vbd[i_x])
        ub[i_x] = ub[i_x] + a[i_x] * vb[i_x]
        bbd[i_x] = bbd[i_x] + vbd[i_x]
        bb[i_x] = bb[i_x] + vb[i_x]
        vbd[i_x] = 0.0
        vb[i_x] = 0.0
    end
    return nothing
end

function affine_loss(loss, u, a, b, v, i_n)
    for i_x = 1:i_n
        v[i_x] = a[i_x] * u[i_x] + b[i_x]
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + v[i_seq_x] ^ 2
    end
end
