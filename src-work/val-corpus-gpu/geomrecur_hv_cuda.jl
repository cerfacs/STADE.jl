import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_geomrecur_hv_1!(i_n, loss, lossd, u, ud)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic lossd[1] += (2 * u[i_seq_x]) * ud[i_seq_x]
    CUDA.@atomic loss[1] += u[i_seq_x] ^ 2
    return nothing
end

function cuda_kernel_geomrecur_hv_2!(i_n, lossb, lossbd, u, ub, ubd, ud)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_n, -1) + 1
        return nothing
    end
    i_seq_x = i_n + (__tid - 1) * -1
    ubd[i_seq_x] = ubd[i_seq_x] + (lossb[1] * (2 * ud[i_seq_x]) + (2 * u[i_seq_x]) * lossbd[1])
    ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * lossb[1]
    return nothing
end

function initstacks_geomrecur_b_cuda(i_n)
    u_stack = CuArray{Float64}(undef, div(i_n - 2, 1) + 1)
    return u_stack
end

function geomrecur_hv_cuda(loss, lossb, u, ub, c, cb, i_n, lossd, lossbd, ud, ubd, cd, cbd, u_stack)
    nthread_per_block = 256
    u_stack_d = CuArray{Float64}(undef, length(u_stack))
    for i_seq_x = 2:i_n
        CUDA.@allowscalar begin
                u_stack_d[(i_seq_x - 2) + 1] = ud[i_seq_x]
                u_stack[(i_seq_x - 2) + 1] = u[i_seq_x]
                ud[i_seq_x] = u[i_seq_x - 1] * cd + c * ud[i_seq_x - 1]
                u[i_seq_x] = c * u[i_seq_x - 1]
            end
    end
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_geomrecur_hv_1!(i_n, loss, lossd, u, ud)
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_n, -1) + 1, nthread_per_block) cuda_kernel_geomrecur_hv_2!(i_n, lossb, lossbd, u, ub, ubd, ud)
    for i_seq_x = i_n:-1:2
        CUDA.@allowscalar begin
                ud[i_seq_x] = u_stack_d[(i_seq_x - 2) + 1]
                u[i_seq_x] = u_stack[(i_seq_x - 2) + 1]
                cbd = cbd + (ub[i_seq_x] * ud[i_seq_x - 1] + u[i_seq_x - 1] * ubd[i_seq_x])
                cb = cb + u[i_seq_x - 1] * ub[i_seq_x]
                ubd[i_seq_x - 1] = ubd[i_seq_x - 1] + (ub[i_seq_x] * cd + c * ubd[i_seq_x])
                ub[i_seq_x - 1] = ub[i_seq_x - 1] + c * ub[i_seq_x]
                ubd[i_seq_x] = 0.0
                ub[i_seq_x] = 0.0
            end
    end
    return (cb, cbd)
end

function geomrecur_cuda(loss, u, c, i_n)
    for i_seq_x = 2:i_n
        CUDA.@allowscalar begin
                u[i_seq_x] = c * u[i_seq_x - 1]
            end
    end
    CUDA.@allowscalar begin
            loss[1] = loss[1] + sum(abs2, u)
        end
    return nothing
end
