import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function sumsq_shifted_cuda(loss, u, alpha, beta, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + (alpha * u[i_seq_x] + beta) ^ 2
    end
    return nothing
end
