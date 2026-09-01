function initstacks_cellscatter_b(i_ncell, i_njac, i_nnode)
    up_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    auxu_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1))
    return (up_stack, auxu_stack)
end

function cellscatter_b(i_cell_to_node, cell_vol, cell_volb, i_ncell, i_nnode, i_njac, up, upb, mup, mupb, up_stack, auxu_stack)
    auxu = 0.0
    auxub = 0.0
    for i_ = 1:i_njac
        for i_cell = 1:i_ncell
            auxu = 0.0
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxu = auxu + up[i_lnode]
            end
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                mup[i_lnode] = mup[i_lnode] + (auxu + up[i_lnode]) * cell_vol[i_cell]
            end
            __idx_auxu_stack_3 = ((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1
            auxu_stack[__idx_auxu_stack_3] = auxu
        end
        for i_node = 1:i_nnode
            __idx_up_stack_0 = ((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1
            __cse_0 = up[i_node]
            up_stack[__idx_up_stack_0] = __cse_0
            up[i_node] = __cse_0 - mup[i_node]
        end
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __idx_up_stack_0 = ((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1
            up[i_node] = up_stack[__idx_up_stack_0]
            mupb[i_node] = mupb[i_node] + -(upb[i_node])
        end
        for i_cell = i_ncell:-1:1
            __idx_auxu_stack_0 = ((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1
            auxu = auxu_stack[__idx_auxu_stack_0]
            for i_loc = 4:-1:1
                i_lnode = i_cell_to_node[i_loc, i_cell]
                __cse_1 = mupb[i_lnode]
                __cse_2 = cell_vol[i_cell] * __cse_1
                auxub = auxub + __cse_2
                upb[i_lnode] = upb[i_lnode] + __cse_2
                cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_lnode]) * __cse_1
            end
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxu = auxu + up[i_lnode]
                i_lnode = i_cell_to_node[i_loc, i_cell]
                upb[i_lnode] = upb[i_lnode] + auxub
            end
            auxub = 0.0
        end
    end
    return nothing
end

function cellscatter(i_cell_to_node, cell_vol, i_ncell, i_nnode, i_njac, up, mup)
    for i_ = 1:i_njac
        for i_cell = 1:i_ncell
            auxu = 0.0
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxu = auxu + up[i_lnode]
            end
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                mup[i_lnode] = mup[i_lnode] + (auxu + up[i_lnode]) * cell_vol[i_cell]
            end
        end
        for i_node = 1:i_nnode
            up[i_node] = up[i_node] - mup[i_node]
        end
    end
end
