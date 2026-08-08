function initstacks_advection_b(du)
    du_stack = Vector{typeof(du)}()
    return du_stack
end
function advection_b(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, du_stack)
    for i_seq_ = 1:i_nstep
        push!(du_stack, copy(du))
        for i_x = 2:i_nnode
            du[i_x] = u[i_x] - u[i_x - 1]
        end
        push!(du_stack, copy(du))
        for i_x = 2:i_nnode
            u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
        end
    end
    for i_seq_ = i_nstep:-1:1
        du .= pop!(du_stack)
        for i_x = 2:i_nnode
            temp = du[i_x] / dx
            cb = cb - dt * temp * ub[i_x]
            dtb = dtb - c * temp * ub[i_x]
            tempb = -((c * dt * ub[i_x]) / dx)
            dub[i_x] = dub[i_x] + tempb
            dxb = dxb - temp * tempb
        end
        du .= pop!(du_stack)
        for i_x = 2:i_nnode
            ub[i_x] = ub[i_x] + dub[i_x]
            ub[i_x - 1] = ub[i_x - 1] - dub[i_x]
            dub[i_x] = 0.0
        end
    end
    return cb,dxb,dtb
end
function advection(u, du, c, dx, dt, i_nstep, i_nnode)
    for i_seq_ = 1:i_nstep
        for i_x = 2:i_nnode
            du[i_x] = u[i_x] - u[i_x - 1]
        end
        for i_x = 2:i_nnode
            u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
        end
    end
end
