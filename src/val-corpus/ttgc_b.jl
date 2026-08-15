function initstacks_ttgc_b()
    mup_stack = Vector{Float64}()
    return mup_stack
end

function ttgc_b(i_cell_to_node, node_vol, node_volb, i_ncell, i_nnode, i_njac, up, upb, mup, mupb, mup_stack)
    for i_seq_ = 1:i_njac
        for i_node = 1:i_nnode
            push!(mup_stack, mup[i_node])
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            i_node1 = i_cell_to_node[1, i_cell]
            push!(mup_stack, mup[i_node1])
            mup[i_node1] = mup[i_node1] + up[i_node1]
        end
        for i_node = 1:i_nnode
            up[i_node] = up[i_node] - mup[i_node] / node_vol[i_node]
        end
    end
    for i_seq_ = i_njac:-1:1
        for i_node = 1:i_nnode
            mupb[i_node] = mupb[i_node] + (1.0 / node_vol[i_node]) * -(upb[i_node])
            node_volb[i_node] = node_volb[i_node] + -(mup[i_node] / node_vol[i_node] ^ 2) * -(upb[i_node])
        end
        for i_cell = i_ncell:-1:1
            i_node1 = i_cell_to_node[1, i_cell]
            mup[i_node1] = pop!(mup_stack)
            upb[i_node1] = upb[i_node1] + mupb[i_node1]
        end
        for i_node = i_nnode:-1:1
            mup[i_node] = pop!(mup_stack)
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
