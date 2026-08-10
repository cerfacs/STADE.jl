import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function initstacks_dotprod_b_metal()
    return nothing
end

function dotprod_b_metal(loss, lossb, u, ub, v, vb, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + u[i_seq_x] * v[i_seq_x]
    end
    for i_seq_x = i_n:-1:1
        ub[i_seq_x] = ub[i_seq_x] + v[i_seq_x] * lossb[1]
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
