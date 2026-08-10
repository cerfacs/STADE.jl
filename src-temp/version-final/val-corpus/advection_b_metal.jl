import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function metal_kernel_advection_b_1!(c, dt, du, dx, i_nnode, u)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    u[i_x] = u[i_x] - Float32(c * dt * du[i_x]) / Float32(dx)
    return nothing
end

function metal_kernel_advection_b_2!(c, dt, du, dub, dx, i_nnode, ub)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    cb = cb + (dt * du[i_x]) * ((Float32(1.0f0) / Float32(dx)) * -(ub[i_x]))
    dtb = dtb + (c * du[i_x]) * ((Float32(1.0f0) / Float32(dx)) * -(ub[i_x]))
    dub[i_x] = dub[i_x] + (c * dt) * ((Float32(1.0f0) / Float32(dx)) * -(ub[i_x]))
    dxb = dxb + -(Float32(c * dt * du[i_x]) / Float32(dx ^ 2)) * -(ub[i_x])
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

function initstacks_advection_b_metal()
    du_stack = Vector{Float64}()
    return du_stack
end

function advection_b_metal(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, du_stack)
    nthread_per_block = 256
    for i_seq_ = 1:i_nstep
        for i_x = 2:i_nnode
            push!(du_stack, du[i_x])
            du[i_x] = u[i_x] - u[i_x - 1]
        end
        @metal threads = nthread_per_block groups = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) metal_kernel_advection_b_1!(c, dt, du, dx, i_nnode, u)
    end
    for i_seq_ = i_nstep:-1:1
        @metal threads = nthread_per_block groups = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) metal_kernel_advection_b_2!(c, dt, du, dub, dx, i_nnode, ub)
        for i_x = i_nnode:-1:2
            du[i_x] = pop!(du_stack)
            ub[i_x] = ub[i_x] + dub[i_x]
            ub[i_x - 1] = ub[i_x - 1] + -(dub[i_x])
            dub[i_x] = 0.0f0
        end
    end
    return (cb, dxb, dtb)
end

function advection_metal(u, du, c, dx, dt, i_nstep, i_nnode)
    nthread_per_block = 256
    for i_seq_ = 1:i_nstep
        @metal threads = nthread_per_block groups = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) metal_kernel_advection_1!(du, i_nnode, u)
        @metal threads = nthread_per_block groups = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) metal_kernel_advection_2!(c, dt, du, dx, i_nnode, u)
    end
    return nothing
end
