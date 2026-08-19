import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_cond_loop_choice_b_1!(i_n, lossb, u, ub)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_n, -1) + 1
        return nothing
    end
    i_seq_x = i_n + (__tid - 1) * -1
    ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * lossb[1]
    return nothing
end

function cuda_kernel_cond_loop_choice_b_2!(i_n, lossb, v, vb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_n, -1) + 1
        return nothing
    end
    i_seq_x = i_n + (__tid - 1) * -1
    vb[i_seq_x] = vb[i_seq_x] + (2 * v[i_seq_x]) * lossb[1]
    return nothing
end

function initstacks_cond_loop_choice_b_cuda()
    branch_stack = CuArray{Int64}(undef, 1)
    return branch_stack
end

function cond_loop_choice_b_cuda(loss, lossb, u, ub, v, vb, i_branch, i_n, branch_stack)
    nthread_per_block = 256
    if i_branch == 1
        CUDA.@allowscalar begin
                branch_stack[1] = 1
            end
        CUDA.@allowscalar begin
                loss[1] = loss[1] + sum(abs2, u)
            end
    else
        CUDA.@allowscalar begin
                branch_stack[1] = 0
            end
        CUDA.@allowscalar begin
                loss[1] = loss[1] + sum(abs2, v)
            end
    end
    CUDA.@allowscalar begin
            __branch = branch_stack[1]
        end
    if __branch == 1
        @cuda threads = nthread_per_block blocks = cld(div(1 - i_n, -1) + 1, nthread_per_block) cuda_kernel_cond_loop_choice_b_1!(i_n, lossb, u, ub)
    else
        @cuda threads = nthread_per_block blocks = cld(div(1 - i_n, -1) + 1, nthread_per_block) cuda_kernel_cond_loop_choice_b_2!(i_n, lossb, v, vb)
    end
    return nothing
end

function cond_loop_choice_cuda(loss, u, v, i_branch, i_n)
    if i_branch == 1
        CUDA.@allowscalar begin
                loss[1] = loss[1] + sum(abs2, u)
            end
    else
        CUDA.@allowscalar begin
                loss[1] = loss[1] + sum(abs2, v)
            end
    end
    return nothing
end
