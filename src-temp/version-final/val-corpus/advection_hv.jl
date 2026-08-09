function initstacks_advection_b()
    du_stack = Vector{Float64}()
    return du_stack
end

function advection_hv(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, ud, ubd, dud, dubd, cd, cbd, dxd, dxbd, dtd, dtbd, du_stack)
    du_stack_d = Vector{Float64}()
    for i_seq_ = 1:i_nstep
        for i_x = 2:i_nnode
            push!(du_stack_d, dud[i_x])
            push!(du_stack, du[i_x])
            dud[i_x] = ud[i_x] + -(ud[i_x - 1])
            du[i_x] = u[i_x] - u[i_x - 1]
        end
        for i_x = 2:i_nnode
            ud[i_x] = ud[i_x] + -(((1.0 / dx) * (((dt * du[i_x]) * cd + (c * du[i_x]) * dtd) + (c * dt) * dud[i_x]) + -((c * dt * du[i_x]) / dx ^ 2) * dxd))
            u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
        end
    end
    for i_seq_ = i_nstep:-1:1
        for i_x = 2:i_nnode
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
            dud[i_x] = pop!(du_stack_d)
            du[i_x] = pop!(du_stack)
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
    #= none:1 =#
    #= none:2 =#
    for i_seq_ = 1:i_nstep
        #= none:3 =#
        for i_x = 2:i_nnode
            #= none:4 =#
            du[i_x] = u[i_x] - u[i_x - 1]
            #= none:5 =#
        end
        #= none:6 =#
        for i_x = 2:i_nnode
            #= none:7 =#
            u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
            #= none:8 =#
        end
        #= none:9 =#
    end
end
