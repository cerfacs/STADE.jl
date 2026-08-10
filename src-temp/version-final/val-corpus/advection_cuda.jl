import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_advection_1!(du, i_nnode, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    du[i_x] = u[i_x] - u[i_x - 1]
    return nothing
end

function cuda_kernel_advection_2!(c, dt, du, dx, i_nnode, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
    return nothing
end

function advection_cuda(u, du, c, dx, dt, i_nstep, i_nnode)
    nthread_per_block = 256
    for i_seq_ = 1:i_nstep
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) cuda_kernel_advection_1!(du, i_nnode, u)
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) cuda_kernel_advection_2!(c, dt, du, dx, i_nnode, u)
    end
    return nothing
end
