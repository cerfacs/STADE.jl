import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function weightedsumsq_metal(loss, u, w, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x] * u[i_seq_x] ^ 2
    end
    return nothing
end
