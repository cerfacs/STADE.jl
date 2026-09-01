function initstacks_entry_dead_b(i_ncell)
    v_stack = Vector{Float64}(undef, (max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1) + max(0, div(i_ncell - 1, 1) + 1)) + 1)
    return v_stack
end

function entry_dead_b(x, xb, y, yb, i_cell_to_node, i_ncell, i_nnode, res, resb, out, outb, v_stack)
    v = 0.0
    vb = 0.0
    v = 0.0
    for i_cell = 1:i_ncell
        v = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            v = x[i_node] * y[i_cell]
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
                res[i_k_node] = res[i_k_node] + v * v
            end
        end
    end
    for i_x = 1:i_nnode
        out[i_x] = out[i_x] + res[i_x] * res[i_x]
    end
    __idx_v_stack_3 = (max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1) + max(0, div(i_ncell - 1, 1) + 1)) + 1
    v_stack[__idx_v_stack_3] = v
    __idx_v_stack_0 = (max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1) + max(0, div(i_ncell - 1, 1) + 1)) + 1
    v = v_stack[__idx_v_stack_0]
    for i_x = i_nnode:-1:1
        resb[i_x] = resb[i_x] + res[i_x] * outb[i_x]
        resb[i_x] = resb[i_x] + res[i_x] * outb[i_x]
    end
    for i_cell = 1:i_ncell
        v = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            v = x[i_node] * y[i_cell]
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
            end
        end
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            v = x[i_node] * y[i_cell]
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
            end
            i_node = i_cell_to_node[i_loc, i_cell]
            for i_k = 4:-1:1
                i_k_node = i_cell_to_node[i_k, i_cell]
                vb = vb + v * resb[i_k_node]
                vb = vb + v * resb[i_k_node]
            end
            xb[i_node] = xb[i_node] + y[i_cell] * vb
            yb[i_cell] = yb[i_cell] + x[i_node] * vb
            vb = 0.0
        end
        vb = 0.0
    end
    vb = 0.0
    return nothing
end

function entry_dead(x, y, i_cell_to_node, i_ncell, i_nnode, res, out)
    v = 0.0
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            v = x[i_node] * y[i_cell]
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
                res[i_k_node] = res[i_k_node] + v * v
            end
        end
    end
    for i_x = 1:i_nnode
        out[i_x] = out[i_x] + res[i_x] * res[i_x]
    end
end
