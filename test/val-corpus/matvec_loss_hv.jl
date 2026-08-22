function initstacks_matvec_loss_b()
    return nothing
end

function matvec_loss_hv(loss, lossb, a, ab, u, ub, v, vb, i_m, i_n, lossd, lossbd, ad, abd, ud, ubd, vd, vbd)
    for i_i = 1:i_m
        for i_seq_j = 1:i_n
            vd[i_i] = vd[i_i] + (u[i_seq_j] * ad[i_i, i_seq_j] + a[i_i, i_seq_j] * ud[i_seq_j])
            v[i_i] = v[i_i] + a[i_i, i_seq_j] * u[i_seq_j]
        end
    end
    for i_seq_i = 1:i_m
        lossd[1] = lossd[1] + (2 * v[i_seq_i]) * vd[i_seq_i]
        loss[1] = loss[1] + v[i_seq_i] ^ 2
    end
    for i_seq_i = i_m:-1:1
        vbd[i_seq_i] = vbd[i_seq_i] + (lossb[1] * (2 * vd[i_seq_i]) + (2 * v[i_seq_i]) * lossbd[1])
        vb[i_seq_i] = vb[i_seq_i] + (2 * v[i_seq_i]) * lossb[1]
    end
    for i_i = 1:i_m
        for i_seq_j = i_n:-1:1
            abd[i_i, i_seq_j] = abd[i_i, i_seq_j] + (vb[i_i] * ud[i_seq_j] + u[i_seq_j] * vbd[i_i])
            ab[i_i, i_seq_j] = ab[i_i, i_seq_j] + u[i_seq_j] * vb[i_i]
            ubd[i_seq_j] = ubd[i_seq_j] + (vb[i_i] * ad[i_i, i_seq_j] + a[i_i, i_seq_j] * vbd[i_i])
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
