import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_richardson_substep_d_1!(a_coef, a_coefd, h, hd, nsub, y, yd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(nsub - 1, 1) + 1
        return nothing
    end
    i_seq_sub = 1 + (__tid - 1)
    CUDA.@atomic yd[1] += -((a_coef * y) * hd) + -((h * y) * a_coefd) + -((h * a_coef) * yd)
    CUDA.@atomic y[1] += -(h * a_coef * y)
    return nothing
end

function cuda_kernel_richardson_substep_1!(a_coef, h, nsub, y)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(nsub - 1, 1) + 1
        return nothing
    end
    i_seq_sub = 1 + (__tid - 1)
    CUDA.@atomic y[1] += -(h * a_coef * y)
    return nothing
end

function richardson_substep_d_cuda(y_init, y_initd, out, outd, a_coef, a_coefd, dt_stage, dt_staged, num_stages)
    y = CuArray([y])
    yd = CuArray([yd])
    nthread_per_block = 256
    nsubd = 0.0
    nsub = 1
    for i_seq_stage = 1:num_stages
        hd = (1.0 / nsub) * dt_staged
        h = dt_stage / nsub
        yd = y_initd
        y = y_init
        @cuda threads = nthread_per_block blocks = cld(div(nsub - 1, 1) + 1, nthread_per_block) cuda_kernel_richardson_substep_d_1!(a_coef, a_coefd, h, hd, nsub, y, yd)
        CUDA.@allowscalar begin
                outd[i_seq_stage] = yd
                out[i_seq_stage] = y
                nsubd = 0.0
                nsub = nsub * 2
            end
    end
    return nothing
end

function richardson_substep_cuda(y_init, out, a_coef, dt_stage, num_stages)
    y = CuArray([y])
    nthread_per_block = 256
    nsub = 1
    for i_seq_stage = 1:num_stages
        h = dt_stage / nsub
        y = y_init
        @cuda threads = nthread_per_block blocks = cld(div(nsub - 1, 1) + 1, nthread_per_block) cuda_kernel_richardson_substep_1!(a_coef, h, nsub, y)
        CUDA.@allowscalar begin
                out[i_seq_stage] = y
                nsub = nsub * 2
            end
    end
    return nothing
end
