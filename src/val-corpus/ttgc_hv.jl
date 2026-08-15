function initstacks_ttgc_b()
    re_stack = Vector{Float64}()
    return re_stack
end

function ttgc_hv(u, ub, i_cell_to_node, cell_vol, cell_volb, i_ncell, res, resb, ud, ubd, cell_vold, cell_volbd, resd, resbd, re_stack)
    re_stack_d = Vector{Float64}()
    re = 0.0
    reb = 0.0
    red = 0.0
    rebd = 0.0
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            push!(re_stack_d, red)
            push!(re_stack, re)
            red = (1.0 / cell_vol[i_cell]) * ud[i_node] + -(u[i_node] / cell_vol[i_cell] ^ 2) * cell_vold[i_cell]
            re = u[i_node] / cell_vol[i_cell]
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
                resd[i_k_node] = resd[i_k_node] + (u[i_node] * red + re * ud[i_node])
                res[i_k_node] = res[i_k_node] + re * u[i_node]
            end
        end
        push!(re_stack_d, red)
        push!(re_stack, re)
    end
    push!(re_stack_d, red)
    push!(re_stack, re)
    red = pop!(re_stack_d)
    re = pop!(re_stack)
    for i_cell = i_ncell:-1:1
        red = pop!(re_stack_d)
        re = pop!(re_stack)
        for i_loc = 4:-1:1
            i_node = i_cell_to_node[i_loc, i_cell]
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
                rebd = rebd + (resb[i_k_node] * ud[i_node] + u[i_node] * resbd[i_k_node])
                reb = reb + u[i_node] * resb[i_k_node]
                ubd[i_node] = ubd[i_node] + (resb[i_k_node] * red + re * resbd[i_k_node])
                ub[i_node] = ub[i_node] + re * resb[i_k_node]
            end
            red = pop!(re_stack_d)
            re = pop!(re_stack)
            ubd[i_node] = ubd[i_node] + (reb * (-(1.0 / cell_vol[i_cell] ^ 2) * cell_vold[i_cell]) + (1.0 / cell_vol[i_cell]) * rebd)
            ub[i_node] = ub[i_node] + (1.0 / cell_vol[i_cell]) * reb
            cell_volbd[i_cell] = cell_volbd[i_cell] + (reb * -(((1.0 / cell_vol[i_cell] ^ 2) * ud[i_node] + -(u[i_node] / (cell_vol[i_cell] ^ 2) ^ 2) * ((2 * cell_vol[i_cell]) * cell_vold[i_cell]))) + -(u[i_node] / cell_vol[i_cell] ^ 2) * rebd)
            cell_volb[i_cell] = cell_volb[i_cell] + -(u[i_node] / cell_vol[i_cell] ^ 2) * reb
            rebd = 0.0
            reb = 0.0
        end
    end
    return nothing
end

function ttgc(u, i_cell_to_node, cell_vol, i_ncell, res)
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            re = u[i_node] / cell_vol[i_cell]
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
                res[i_k_node] = res[i_k_node] + re * u[i_node]
            end
        end
    end
end
