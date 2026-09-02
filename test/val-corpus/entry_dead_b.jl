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
        __cse_0 = res[i_x]
        out[i_x] = out[i_x] + __cse_0 * __cse_0
    end
    __icse_1 = max(0, div(i_ncell - 1, 1) + 1)
    __icse_2 = (__icse_1 * max(0, div(4 - 1, 1) + 1) + __icse_1) + 1
    __idx_v_stack_3 = __icse_2
    v_stack[__idx_v_stack_3] = v
    __idx_v_stack_0 = __icse_2
    v = v_stack[__idx_v_stack_0]
    for i_x = i_nnode:-1:1
        __cse_3 = res[i_x] * outb[i_x]
        resb[i_x] = resb[i_x] + __cse_3
        resb[i_x] = resb[i_x] + __cse_3
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
                __cse_4 = v * resb[i_k_node]
                vb = vb + __cse_4
                vb = vb + __cse_4
            end
            __oldb_0 = vb
            vb = 0.0
            xb[i_node] = xb[i_node] + y[i_cell] * __oldb_0
            yb[i_cell] = yb[i_cell] + x[i_node] * __oldb_0
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
