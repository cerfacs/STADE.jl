import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function sumsq_shifted_d_metal(loss, lossd, u, ud, alpha, alphad, beta, betad, i_n)
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + (2 * (alpha * u[i_seq_x] + beta)) * ((u[i_seq_x] * alphad + alpha * ud[i_seq_x]) + betad)
        loss[1] = loss[1] + (alpha * u[i_seq_x] + beta) ^ 2
    end
    return nothing
end

function sumsq_shifted_metal(loss, u, alpha, beta, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + (alpha * u[i_seq_x] + beta) ^ 2
    end
    return nothing
end
