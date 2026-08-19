import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_geomrecur_b_1!(i_n, lossb, u, ub)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_n, -1) + 1
        return nothing
    end
    i_seq_x = i_n + (__tid - 1) * -1
    ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * lossb[1]
    return nothing
end

function initstacks_geomrecur_b_cuda(i_n)
    u_stack = CuArray{Float64}(undef, div(i_n - 2, 1) + 1)
    return u_stack
end

function geomrecur_b_cuda(loss, lossb, u, ub, c, cb, i_n, u_stack)
    nthread_per_block = 256
    for i_seq_x = 2:i_n
        CUDA.@allowscalar begin
                u_stack[(i_seq_x - 2) + 1] = u[i_seq_x]
                u[i_seq_x] = c * u[i_seq_x - 1]
            end
    end
    CUDA.@allowscalar begin
            loss[1] = loss[1] + sum(abs2, u)
        end
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_n, -1) + 1, nthread_per_block) cuda_kernel_geomrecur_b_1!(i_n, lossb, u, ub)
    for i_seq_x = i_n:-1:2
        CUDA.@allowscalar begin
                u[i_seq_x] = u_stack[(i_seq_x - 2) + 1]
                cb = cb + u[i_seq_x - 1] * ub[i_seq_x]
                ub[i_seq_x - 1] = ub[i_seq_x - 1] + c * ub[i_seq_x]
                ub[i_seq_x] = 0.0
            end
    end
    return cb
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
