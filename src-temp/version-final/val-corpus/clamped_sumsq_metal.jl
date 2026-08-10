import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function clamped_sumsq_metal(loss, u, i_n)
    for i_seq_x = 1:i_n
        if u[i_seq_x] > 0.0f0
            w = u[i_seq_x] ^ 2
        else
            w = 0.0f0
        end
        loss[1] = loss[1] + w
    end
    return nothing
end
