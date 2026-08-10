import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_stencil_loss_1!(i_n, u, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function stencil_loss_cuda(loss, u, w, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block) cuda_kernel_stencil_loss_1!(i_n, u, w)
    for i_seq_x = 2:i_n - 1
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
    return nothing
end
