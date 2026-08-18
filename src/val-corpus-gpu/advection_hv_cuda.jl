import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_advection_hv_1!(du, du_stack, du_stack_d, dud, i_nnode, i_seq_, u, ud)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    du_stack_d[((i_seq_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1] = dud[i_x]
    du_stack[((i_seq_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1] = du[i_x]
    dud[i_x] = ud[i_x] + -(ud[i_x - 1])
    du[i_x] = u[i_x] - u[i_x - 1]
    return nothing
end

function cuda_kernel_advection_hv_2!(c, cd, dt, dtd, du, dud, dx, dxd, i_nnode, u, ud)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    ud[i_x] = ud[i_x] + -(((1.0 / dx) * (((dt * du[i_x]) * cd + (c * du[i_x]) * dtd) + (c * dt) * dud[i_x]) + -((c * dt * du[i_x]) / dx ^ 2) * dxd))
    u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
    return nothing
end

function cuda_kernel_advection_hv_3!(c, cb, cbd, cd, dt, dtb, dtbd, dtd, du, dub, dubd, dud, dx, dxb, dxbd, dxd, i_nnode, ub, ubd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    CUDA.@atomic cbd[1] += ((1.0 / dx) * -(ub[i_x])) * (du[i_x] * dtd + dt * dud[i_x]) + (dt * du[i_x]) * (-(ub[i_x]) * (-(1.0 / dx ^ 2) * dxd) + (1.0 / dx) * -(ubd[i_x]))
    CUDA.@atomic cb[1] += (dt * du[i_x]) * ((1.0 / dx) * -(ub[i_x]))
    CUDA.@atomic dtbd[1] += ((1.0 / dx) * -(ub[i_x])) * (du[i_x] * cd + c * dud[i_x]) + (c * du[i_x]) * (-(ub[i_x]) * (-(1.0 / dx ^ 2) * dxd) + (1.0 / dx) * -(ubd[i_x]))
    CUDA.@atomic dtb[1] += (c * du[i_x]) * ((1.0 / dx) * -(ub[i_x]))
    dubd[i_x] = dubd[i_x] + (((1.0 / dx) * -(ub[i_x])) * (dt * cd + c * dtd) + (c * dt) * (-(ub[i_x]) * (-(1.0 / dx ^ 2) * dxd) + (1.0 / dx) * -(ubd[i_x])))
    dub[i_x] = dub[i_x] + (c * dt) * ((1.0 / dx) * -(ub[i_x]))
    CUDA.@atomic dxbd[1] += -(ub[i_x]) * -(((1.0 / dx ^ 2) * (((dt * du[i_x]) * cd + (c * du[i_x]) * dtd) + (c * dt) * dud[i_x]) + -((c * dt * du[i_x]) / (dx ^ 2) ^ 2) * ((2dx) * dxd))) + -((c * dt * du[i_x]) / dx ^ 2) * -(ubd[i_x])
    CUDA.@atomic dxb[1] += -((c * dt * du[i_x]) / dx ^ 2) * -(ub[i_x])
    return nothing
end

function cuda_kernel_advection_hv_4!(du, du_stack, du_stack_d, dub, dubd, dud, i_nnode, i_seq_, ub, ubd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(2 - i_nnode, -1) + 1
        return nothing
    end
    i_x = i_nnode + (__tid - 1) * -1
    dud[i_x] = du_stack_d[((i_seq_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1]
    du[i_x] = du_stack[((i_seq_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1]
    ubd[i_x] = ubd[i_x] + dubd[i_x]
    ub[i_x] = ub[i_x] + dub[i_x]
    ubd[i_x - 1] = ubd[i_x - 1] + -(dubd[i_x])
    ub[i_x - 1] = ub[i_x - 1] + -(dub[i_x])
    dubd[i_x] = 0.0
    dub[i_x] = 0.0
    return nothing
end

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

function initstacks_advection_b_cuda(i_nnode, i_nstep)
    du_stack = CuArray{Float64}(undef, (div(i_nstep - 1, 1) + 1) * (div(i_nnode - 2, 1) + 1))
    return du_stack
end

function advection_hv_cuda(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, ud, ubd, dud, dubd, cd, cbd, dxd, dxbd, dtd, dtbd, du_stack)
    cb = CuArray([cb])
    cbd = CuArray([cbd])
    dtb = CuArray([dtb])
    dtbd = CuArray([dtbd])
    dxb = CuArray([dxb])
    dxbd = CuArray([dxbd])
    nthread_per_block = 256
    du_stack_d = CuArray{Float64}(undef, (div(i_nstep - 1, 1) + 1) * (div(i_nnode - 2, 1) + 1))
    for i_seq_ = 1:i_nstep
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) cuda_kernel_advection_hv_1!(du, du_stack, du_stack_d, dud, i_nnode, i_seq_, u, ud)
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) cuda_kernel_advection_hv_2!(c, cd, dt, dtd, du, dud, dx, dxd, i_nnode, u, ud)
    end
    for i_seq_ = i_nstep:-1:1
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) cuda_kernel_advection_hv_3!(c, cb, cbd, cd, dt, dtb, dtbd, dtd, du, dub, dubd, dud, dx, dxb, dxbd, dxd, i_nnode, ub, ubd)
        @cuda threads = nthread_per_block blocks = cld(div(2 - i_nnode, -1) + 1, nthread_per_block) cuda_kernel_advection_hv_4!(du, du_stack, du_stack_d, dub, dubd, dud, i_nnode, i_seq_, ub, ubd)
    end
    cb = (Array(cb))[1]
    cbd = (Array(cbd))[1]
    dtb = (Array(dtb))[1]
    dtbd = (Array(dtbd))[1]
    dxb = (Array(dxb))[1]
    dxbd = (Array(dxbd))[1]
    return (cb, cbd, dxb, dxbd, dtb, dtbd)
end

function advection_cuda(u, du, c, dx, dt, i_nstep, i_nnode)
    nthread_per_block = 256
    for i_seq_ = 1:i_nstep
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) cuda_kernel_advection_1!(du, i_nnode, u)
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) cuda_kernel_advection_2!(c, dt, du, dx, i_nnode, u)
    end
    return nothing
end
