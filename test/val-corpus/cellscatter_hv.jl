function initstacks_cellscatter_b(i_ncell, i_njac, i_nnode)
    up_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    auxu_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1))
    return (up_stack, auxu_stack)
end

function cellscatter_hv(i_cell_to_node, cell_vol, cell_volb, i_ncell, i_nnode, i_njac, up, upb, mup, mupb, cell_vold, cell_volbd, upd, upbd, mupd, mupbd, up_stack, auxu_stack)
    up_stack_d = Vector{Float64}(undef, length(up_stack))
    auxu_stack_d = Vector{Float64}(undef, length(auxu_stack))
    auxu = 0.0
    auxub = 0.0
    auxud = 0.0
    auxubd = 0.0
    for i_ = 1:i_njac
        for i_cell = 1:i_ncell
            auxud = 0.0
            auxu = 0.0
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxud = auxud + upd[i_lnode]
                auxu = auxu + up[i_lnode]
            end
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                mupd[i_lnode] = mupd[i_lnode] + (cell_vol[i_cell] * (auxud + upd[i_lnode]) + (auxu + up[i_lnode]) * cell_vold[i_cell])
                mup[i_lnode] = mup[i_lnode] + (auxu + up[i_lnode]) * cell_vol[i_cell]
            end
            auxu_stack_d[((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1] = auxud
            auxu_stack[((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1] = auxu
        end
        for i_node = 1:i_nnode
            up_stack_d[((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1] = upd[i_node]
            up_stack[((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1] = up[i_node]
            upd[i_node] = upd[i_node] + -(mupd[i_node])
            up[i_node] = up[i_node] - mup[i_node]
        end
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            upd[i_node] = up_stack_d[((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1]
            up[i_node] = up_stack[((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1]
            mupbd[i_node] = mupbd[i_node] + -(upbd[i_node])
            mupb[i_node] = mupb[i_node] + -(upb[i_node])
        end
        for i_cell = i_ncell:-1:1
            auxud = auxu_stack_d[((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1]
            auxu = auxu_stack[((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1]
            for i_loc = 4:-1:1
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxubd = auxubd + (mupb[i_lnode] * cell_vold[i_cell] + cell_vol[i_cell] * mupbd[i_lnode])
                auxub = auxub + cell_vol[i_cell] * mupb[i_lnode]
                upbd[i_lnode] = upbd[i_lnode] + (mupb[i_lnode] * cell_vold[i_cell] + cell_vol[i_cell] * mupbd[i_lnode])
                upb[i_lnode] = upb[i_lnode] + cell_vol[i_cell] * mupb[i_lnode]
                cell_volbd[i_cell] = cell_volbd[i_cell] + (mupb[i_lnode] * (auxud + upd[i_lnode]) + (auxu + up[i_lnode]) * mupbd[i_lnode])
                cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_lnode]) * mupb[i_lnode]
            end
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxud = auxud + upd[i_lnode]
                auxu = auxu + up[i_lnode]
                i_lnode = i_cell_to_node[i_loc, i_cell]
                upbd[i_lnode] = upbd[i_lnode] + auxubd
                upb[i_lnode] = upb[i_lnode] + auxub
            end
            auxubd = 0.0
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
