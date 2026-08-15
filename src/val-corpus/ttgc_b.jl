function initstacks_ttgc_b()
    re_stack = Vector{Float64}()
    return re_stack
end

function ttgc_b(u, ub, i_cell_to_node, cell_vol, cell_volb, i_ncell, res, resb, re_stack)
    re = 0.0
    reb = 0.0
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            push!(re_stack, re)
            re = u[i_node] / cell_vol[i_cell]
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
                res[i_k_node] = res[i_k_node] + re * u[i_node]
            end
        end
        push!(re_stack, re)
    end
    push!(re_stack, re)
    re = pop!(re_stack)
    for i_cell = i_ncell:-1:1
        re = pop!(re_stack)
        for i_loc = 4:-1:1
            i_node = i_cell_to_node[i_loc, i_cell]
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
                reb = reb + u[i_node] * resb[i_k_node]
                ub[i_node] = ub[i_node] + re * resb[i_k_node]
            end
            re = pop!(re_stack)
            ub[i_node] = ub[i_node] + (1.0 / cell_vol[i_cell]) * reb
            cell_volb[i_cell] = cell_volb[i_cell] + -(u[i_node] / cell_vol[i_cell] ^ 2) * reb
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
