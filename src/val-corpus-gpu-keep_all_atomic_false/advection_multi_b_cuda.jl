import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_advection_multi_b_1!(du, du_stack, i_nnode, i_seq_, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x_advection_diff_c1 = 2 + (__tid - 1)
    du_stack[((i_seq_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x_advection_diff_c1 - 2)) + 1] = du[i_x_advection_diff_c1]
    du[i_x_advection_diff_c1] = u[i_x_advection_diff_c1] - u[i_x_advection_diff_c1 - 1]
    return nothing
end

function cuda_kernel_advection_multi_b_2!(c, dt, du, dx, i_nnode, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x_advection_update_c2 = 2 + (__tid - 1)
    u[i_x_advection_update_c2] = u[i_x_advection_update_c2] - (c * dt * du[i_x_advection_update_c2]) / dx
    return nothing
end

function cuda_kernel_advection_multi_b_3!(c, cb, dt, dtb, du, dub, dx, dxb, i_nnode, ub)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x_advection_update_c2 = 2 + (__tid - 1)
    CUDA.@atomic cb[1] += (dt * du[i_x_advection_update_c2]) * ((1.0 / dx) * -(ub[i_x_advection_update_c2]))
    CUDA.@atomic dtb[1] += (c * du[i_x_advection_update_c2]) * ((1.0 / dx) * -(ub[i_x_advection_update_c2]))
    dub[i_x_advection_update_c2] = dub[i_x_advection_update_c2] + (c * dt) * ((1.0 / dx) * -(ub[i_x_advection_update_c2]))
    CUDA.@atomic dxb[1] += -((c * dt * du[i_x_advection_update_c2]) / dx ^ 2) * -(ub[i_x_advection_update_c2])
    return nothing
end

function cuda_kernel_advection_multi_b_4!(du, du_stack, dub, i_nnode, i_seq_, ub)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(2 - i_nnode, -1) + 1
        return nothing
    end
    i_x_advection_diff_c1 = i_nnode + (__tid - 1) * -1
    du[i_x_advection_diff_c1] = du_stack[((i_seq_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x_advection_diff_c1 - 2)) + 1]
    ub[i_x_advection_diff_c1] = ub[i_x_advection_diff_c1] + dub[i_x_advection_diff_c1]
    ub[i_x_advection_diff_c1 - 1] = ub[i_x_advection_diff_c1 - 1] + -(dub[i_x_advection_diff_c1])
    dub[i_x_advection_diff_c1] = 0.0
    return nothing
end

function cuda_kernel_advection_multi_1!(du, i_nnode, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x_advection_diff_c1 = 2 + (__tid - 1)
    du[i_x_advection_diff_c1] = u[i_x_advection_diff_c1] - u[i_x_advection_diff_c1 - 1]
    return nothing
end

function cuda_kernel_advection_multi_2!(c, dt, du, dx, i_nnode, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x_advection_update_c2 = 2 + (__tid - 1)
    u[i_x_advection_update_c2] = u[i_x_advection_update_c2] - (c * dt * du[i_x_advection_update_c2]) / dx
    return nothing
end

function initstacks_advection_multi_b_cuda(i_nnode, i_nstep)
    du_stack = CuArray{Float64}(undef, (div(i_nstep - 1, 1) + 1) * (div(i_nnode - 2, 1) + 1))
    return du_stack
end

function advection_multi_b_cuda(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, du_stack)
    cb = CuArray([cb])
    dtb = CuArray([dtb])
    dxb = CuArray([dxb])
    nthread_per_block = 256
    for i_seq_ = 1:i_nstep
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) cuda_kernel_advection_multi_b_1!(du, du_stack, i_nnode, i_seq_, u)
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) cuda_kernel_advection_multi_b_2!(c, dt, du, dx, i_nnode, u)
    end
    for i_seq_ = i_nstep:-1:1
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) cuda_kernel_advection_multi_b_3!(c, cb, dt, dtb, du, dub, dx, dxb, i_nnode, ub)
        @cuda threads = nthread_per_block blocks = cld(div(2 - i_nnode, -1) + 1, nthread_per_block) cuda_kernel_advection_multi_b_4!(du, du_stack, dub, i_nnode, i_seq_, ub)
    end
    cb = (Array(cb))[1]
    dtb = (Array(dtb))[1]
    dxb = (Array(dxb))[1]
    return (cb, dxb, dtb)
end

function advection_multi_cuda(u, du, c, dx, dt, i_nstep, i_nnode)
    nthread_per_block = 256
    for i_seq_ = 1:i_nstep
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) cuda_kernel_advection_multi_1!(du, i_nnode, u)
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) cuda_kernel_advection_multi_2!(c, dt, du, dx, i_nnode, u)
    end
    return nothing
end
