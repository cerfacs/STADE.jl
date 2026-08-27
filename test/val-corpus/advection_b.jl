function initstacks_advection_b(i_nnode, i_nstep)
    du_stack = Vector{Float64}(undef, max(0, div(i_nstep - 1, 1) + 1) * max(0, div(i_nnode - 2, 1) + 1))
    return du_stack
end

function advection_b(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, du_stack)
    for i_ = 1:i_nstep
        for i_x = 2:i_nnode
            du_stack[((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1] = du[i_x]
            du[i_x] = u[i_x] - u[i_x - 1]
        end
        for i_x = 2:i_nnode
            u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
        end
    end
    for i_ = i_nstep:-1:1
        for i_x = i_nnode:-1:2
            cb = cb + (dt * du[i_x]) * ((1.0 / dx) * -(ub[i_x]))
            dtb = dtb + (c * du[i_x]) * ((1.0 / dx) * -(ub[i_x]))
            dub[i_x] = dub[i_x] + (c * dt) * ((1.0 / dx) * -(ub[i_x]))
            dxb = dxb + -((c * dt * du[i_x]) / dx ^ 2) * -(ub[i_x])
        end
        for i_x = i_nnode:-1:2
            du[i_x] = du_stack[((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1]
            ub[i_x] = ub[i_x] + dub[i_x]
            ub[i_x - 1] = ub[i_x - 1] + -(dub[i_x])
            dub[i_x] = 0.0
        end
    end
    return (cb, dxb, dtb)
end

function advection(u, du, c, dx, dt, i_nstep, i_nnode)
    for i_ = 1:i_nstep
        for i_x = 2:i_nnode
            du[i_x] = u[i_x] - u[i_x - 1]
        end
        for i_x = 2:i_nnode
            u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
        end
    end
end
