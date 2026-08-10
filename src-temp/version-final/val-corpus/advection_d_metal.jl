import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function metal_kernel_advection_d_1!(du, dud, i_nnode, u, ud)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    dud[i_x] = ud[i_x] + -(ud[i_x - 1])
    du[i_x] = u[i_x] - u[i_x - 1]
    return nothing
end

function metal_kernel_advection_d_2!(c, cd, dt, dtd, du, dud, dx, dxd, i_nnode, u, ud)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    ud[i_x] = ud[i_x] + -(((Float32(1.0f0) / Float32(dx)) * (((dt * du[i_x]) * cd + (c * du[i_x]) * dtd) + (c * dt) * dud[i_x]) + -(Float32(c * dt * du[i_x]) / Float32(dx ^ 2)) * dxd))
    u[i_x] = u[i_x] - Float32(c * dt * du[i_x]) / Float32(dx)
    return nothing
end

function metal_kernel_advection_1!(du, i_nnode, u)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    du[i_x] = u[i_x] - u[i_x - 1]
    return nothing
end

function metal_kernel_advection_2!(c, dt, du, dx, i_nnode, u)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    u[i_x] = u[i_x] - Float32(c * dt * du[i_x]) / Float32(dx)
    return nothing
end

function advection_d_metal(u, ud, du, dud, c, cd, dx, dxd, dt, dtd, i_nstep, i_nnode)
    nthread_per_block = 256
    for i_seq_ = 1:i_nstep
        @metal threads = nthread_per_block groups = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) metal_kernel_advection_d_1!(du, dud, i_nnode, u, ud)
        @metal threads = nthread_per_block groups = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) metal_kernel_advection_d_2!(c, cd, dt, dtd, du, dud, dx, dxd, i_nnode, u, ud)
    end
    return nothing
end

function advection_metal(u, du, c, dx, dt, i_nstep, i_nnode)
    nthread_per_block = 256
    for i_seq_ = 1:i_nstep
        @metal threads = nthread_per_block groups = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) metal_kernel_advection_1!(du, i_nnode, u)
        @metal threads = nthread_per_block groups = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) metal_kernel_advection_2!(c, dt, du, dx, i_nnode, u)
    end
    return nothing
end
