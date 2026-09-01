function initstacks_advection_b(i_nnode, i_nstep)
    du_stack = Vector{Float64}(undef, max(0, div(i_nstep - 1, 1) + 1) * max(0, div(i_nnode - 2, 1) + 1))
    return du_stack
end

function advection_hv(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, ud, ubd, dud, dubd, cd, cbd, dxd, dxbd, dtd, dtbd, du_stack)
    du_stack_d = Vector{Float64}(undef, length(du_stack))
    for i_ = 1:i_nstep
        for i_x = 2:i_nnode
            __idx_du_stack_0 = ((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1
            du_stack_d[__idx_du_stack_0] = dud[i_x]
            du_stack[__idx_du_stack_0] = du[i_x]
            dud[i_x] = ud[i_x] + -(ud[i_x - 1])
            du[i_x] = u[i_x] - u[i_x - 1]
        end
        for i_x = 2:i_nnode
            ud[i_x] = ud[i_x] + -(((1.0 / dx) * (((dt * du[i_x]) * cd + (c * du[i_x]) * dtd) + (c * dt) * dud[i_x]) + -((c * dt * du[i_x]) / dx ^ 2) * dxd))
            u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
        end
    end
    for i_ = i_nstep:-1:1
        for i_x = i_nnode:-1:2
            cbd = cbd + (((1.0 / dx) * -(ub[i_x])) * (du[i_x] * dtd + dt * dud[i_x]) + (dt * du[i_x]) * (-(ub[i_x]) * (-(1.0 / dx ^ 2) * dxd) + (1.0 / dx) * -(ubd[i_x])))
            cb = cb + (dt * du[i_x]) * ((1.0 / dx) * -(ub[i_x]))
            dtbd = dtbd + (((1.0 / dx) * -(ub[i_x])) * (du[i_x] * cd + c * dud[i_x]) + (c * du[i_x]) * (-(ub[i_x]) * (-(1.0 / dx ^ 2) * dxd) + (1.0 / dx) * -(ubd[i_x])))
            dtb = dtb + (c * du[i_x]) * ((1.0 / dx) * -(ub[i_x]))
            dubd[i_x] = dubd[i_x] + (((1.0 / dx) * -(ub[i_x])) * (dt * cd + c * dtd) + (c * dt) * (-(ub[i_x]) * (-(1.0 / dx ^ 2) * dxd) + (1.0 / dx) * -(ubd[i_x])))
            dub[i_x] = dub[i_x] + (c * dt) * ((1.0 / dx) * -(ub[i_x]))
            dxbd = dxbd + (-(ub[i_x]) * -(((1.0 / dx ^ 2) * (((dt * du[i_x]) * cd + (c * du[i_x]) * dtd) + (c * dt) * dud[i_x]) + -((c * dt * du[i_x]) / (dx ^ 2) ^ 2) * ((2dx) * dxd))) + -((c * dt * du[i_x]) / dx ^ 2) * -(ubd[i_x]))
            dxb = dxb + -((c * dt * du[i_x]) / dx ^ 2) * -(ub[i_x])
        end
        for i_x = i_nnode:-1:2
            __idx_du_stack_0 = ((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1
            dud[i_x] = du_stack_d[__idx_du_stack_0]
            du[i_x] = du_stack[__idx_du_stack_0]
            ubd[i_x] = ubd[i_x] + dubd[i_x]
            ub[i_x] = ub[i_x] + dub[i_x]
            ubd[i_x - 1] = ubd[i_x - 1] + -(dubd[i_x])
            ub[i_x - 1] = ub[i_x - 1] + -(dub[i_x])
            dubd[i_x] = 0.0
            dub[i_x] = 0.0
        end
    end
    return (cb, cbd, dxb, dxbd, dtb, dtbd)
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
