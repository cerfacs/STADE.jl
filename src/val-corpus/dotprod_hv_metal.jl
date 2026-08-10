import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function initstacks_dotprod_b_metal()
    return nothing
end

function dotprod_hv_metal(loss, lossb, u, ub, v, vb, i_n, lossd, lossbd, ud, ubd, vd, vbd)
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + (v[i_seq_x] * ud[i_seq_x] + u[i_seq_x] * vd[i_seq_x])
        loss[1] = loss[1] + u[i_seq_x] * v[i_seq_x]
    end
    for i_seq_x = i_n:-1:1
        ubd[i_seq_x] = ubd[i_seq_x] + (lossb[1] * vd[i_seq_x] + v[i_seq_x] * lossbd[1])
        ub[i_seq_x] = ub[i_seq_x] + v[i_seq_x] * lossb[1]
        vbd[i_seq_x] = vbd[i_seq_x] + (lossb[1] * ud[i_seq_x] + u[i_seq_x] * lossbd[1])
        vb[i_seq_x] = vb[i_seq_x] + u[i_seq_x] * lossb[1]
    end
    return nothing
end

function dotprod_metal(loss, u, v, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + u[i_seq_x] * v[i_seq_x]
    end
    return nothing
end
