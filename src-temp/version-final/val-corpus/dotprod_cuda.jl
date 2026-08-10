import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function dotprod_cuda(loss, u, v, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + u[i_seq_x] * v[i_seq_x]
    end
    return nothing
end
