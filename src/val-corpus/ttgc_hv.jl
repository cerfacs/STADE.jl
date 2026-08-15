function initstacks_ttgc_b()
    mup_stack = Vector{Float64}()
    return mup_stack
end

function ttgc_hv(i_cell_to_node, node_vol, node_volb, i_ncell, i_nnode, i_njac, up, upb, mup, mupb, node_vold, node_volbd, upd, upbd, mupd, mupbd, mup_stack)
    mup_stack_d = Vector{Float64}()
    for i_seq_ = 1:i_njac
        for i_node = 1:i_nnode
            mupd[i_node] = 0.0
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            i_node1 = i_cell_to_node[1, i_cell]
            push!(mup_stack_d, mupd[i_node1])
            push!(mup_stack, mup[i_node1])
            mupd[i_node1] = mupd[i_node1] + upd[i_node1]
            mup[i_node1] = mup[i_node1] + up[i_node1]
        end
        for i_node = 1:i_nnode
            upd[i_node] = upd[i_node] + -(((1.0 / node_vol[i_node]) * mupd[i_node] + -(mup[i_node] / node_vol[i_node] ^ 2) * node_vold[i_node]))
            up[i_node] = up[i_node] - mup[i_node] / node_vol[i_node]
        end
    end
    for i_seq_ = i_njac:-1:1
        for i_node = 1:i_nnode
            mupbd[i_node] = mupbd[i_node] + (-(upb[i_node]) * (-(1.0 / node_vol[i_node] ^ 2) * node_vold[i_node]) + (1.0 / node_vol[i_node]) * -(upbd[i_node]))
            mupb[i_node] = mupb[i_node] + (1.0 / node_vol[i_node]) * -(upb[i_node])
            node_volbd[i_node] = node_volbd[i_node] + (-(upb[i_node]) * -(((1.0 / node_vol[i_node] ^ 2) * mupd[i_node] + -(mup[i_node] / (node_vol[i_node] ^ 2) ^ 2) * ((2 * node_vol[i_node]) * node_vold[i_node]))) + -(mup[i_node] / node_vol[i_node] ^ 2) * -(upbd[i_node]))
            node_volb[i_node] = node_volb[i_node] + -(mup[i_node] / node_vol[i_node] ^ 2) * -(upb[i_node])
        end
        for i_cell = i_ncell:-1:1
            i_node1 = i_cell_to_node[1, i_cell]
            mupd[i_node1] = pop!(mup_stack_d)
            mup[i_node1] = pop!(mup_stack)
            upbd[i_node1] = upbd[i_node1] + mupbd[i_node1]
            upb[i_node1] = upb[i_node1] + mupb[i_node1]
        end
        for i_node = 1:i_nnode
            mupbd[i_node] = 0.0
            mupb[i_node] = 0.0
        end
    end
    return nothing
end

function ttgc(i_cell_to_node, node_vol, i_ncell, i_nnode, i_njac, up, mup)
    for i_seq_ = 1:i_njac
        for i_node = 1:i_nnode
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            i_node1 = i_cell_to_node[1, i_cell]
            mup[i_node1] = mup[i_node1] + up[i_node1]
        end
        for i_node = 1:i_nnode
            up[i_node] = up[i_node] - mup[i_node] / node_vol[i_node]
        end
    end
end
