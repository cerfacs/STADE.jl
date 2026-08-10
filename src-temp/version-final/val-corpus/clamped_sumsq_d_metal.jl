import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function clamped_sumsq_d_metal(loss, lossd, u, ud, i_n)
    for i_seq_x = 1:i_n
        if u[i_seq_x] > 0.0f0
            wd = (2 * u[i_seq_x]) * ud[i_seq_x]
            w = u[i_seq_x] ^ 2
        else
            wd = 0.0f0
            w = 0.0f0
        end
        lossd[1] = lossd[1] + wd
        loss[1] = loss[1] + w
    end
    return nothing
end

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
