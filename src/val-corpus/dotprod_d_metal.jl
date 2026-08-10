import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function dotprod_d_metal(loss, lossd, u, ud, v, vd, i_n)
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + (v[i_seq_x] * ud[i_seq_x] + u[i_seq_x] * vd[i_seq_x])
        loss[1] = loss[1] + u[i_seq_x] * v[i_seq_x]
    end
    return nothing
end

function dotprod_metal(loss, u, v, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + u[i_seq_x] * v[i_seq_x]
    end
    return nothing
end
