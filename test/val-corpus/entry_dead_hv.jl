function initstacks_entry_dead_b(i_ncell)
    v_stack = Vector{Float64}(undef, (max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1) + max(0, div(i_ncell - 1, 1) + 1)) + 1)
    return v_stack
end

function entry_dead_hv(x, xb, y, yb, i_cell_to_node, i_ncell, i_nnode, res, resb, out, outb, xd, xbd, yd, ybd, resd, resbd, outd, outbd, v_stack)
    v_stack_d = Vector{Float64}(undef, length(v_stack))
    v = 0.0
    vb = 0.0
    vd = 0.0
    vbd = 0.0
    vd = 0.0
    v = 0.0
    for i_cell = 1:i_ncell
        vd = 0.0
        v = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            __hcse_0 = y[i_cell]
            __hcse_1 = x[i_node]
            vd = __hcse_0 * xd[i_node] + __hcse_1 * yd[i_cell]
            v = __hcse_1 * __hcse_0
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
                __hcse_2 = v * vd
                resd[i_k_node] = resd[i_k_node] + (__hcse_2 + __hcse_2)
                res[i_k_node] = res[i_k_node] + v * v
            end
        end
    end
    for i_x = 1:i_nnode
        __cse_0d = resd[i_x]
        __cse_0 = res[i_x]
        __hcse_3 = __cse_0 * __cse_0d
        outd[i_x] = outd[i_x] + (__hcse_3 + __hcse_3)
        out[i_x] = out[i_x] + __cse_0 * __cse_0
    end
    __ihcse_4 = max(0, div(i_ncell - 1, 1) + 1)
    __icse_1 = __ihcse_4
    __ihcse_5 = max(0, div(4 - 1, 1) + 1)
    __idx_v_stack_3 = (__icse_1 * __ihcse_5 + __icse_1) + 1
    v_stack_d[__idx_v_stack_3] = vd
    v_stack[__idx_v_stack_3] = v
    __icse_2 = __ihcse_4
    __idx_v_stack_0 = (__icse_2 * __ihcse_5 + __icse_2) + 1
    vd = v_stack_d[__idx_v_stack_0]
    v = v_stack[__idx_v_stack_0]
    for i_x = i_nnode:-1:1
        __hcse_6 = outb[i_x]
        __hcse_7 = res[i_x]
        __cse_3d = __hcse_6 * resd[i_x] + __hcse_7 * outbd[i_x]
        __cse_3 = __hcse_7 * __hcse_6
        resbd[i_x] = resbd[i_x] + __cse_3d
        resb[i_x] = resb[i_x] + __cse_3
        resbd[i_x] = resbd[i_x] + __cse_3d
        resb[i_x] = resb[i_x] + __cse_3
    end
    for i_cell = 1:i_ncell
        vd = 0.0
        v = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            __hcse_8 = y[i_cell]
            __hcse_9 = x[i_node]
            vd = __hcse_8 * xd[i_node] + __hcse_9 * yd[i_cell]
            v = __hcse_9 * __hcse_8
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
            end
        end
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            __hcse_10 = y[i_cell]
            __hcse_11 = x[i_node]
            vd = __hcse_10 * xd[i_node] + __hcse_11 * yd[i_cell]
            v = __hcse_11 * __hcse_10
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
            end
            i_node = i_cell_to_node[i_loc, i_cell]
            for i_k = 4:-1:1
                i_k_node = i_cell_to_node[i_k, i_cell]
                __hcse_12 = resb[i_k_node]
                __cse_4d = __hcse_12 * vd + v * resbd[i_k_node]
                __cse_4 = v * __hcse_12
                vbd = vbd + __cse_4d
                vb = vb + __cse_4
                vbd = vbd + __cse_4d
                vb = vb + __cse_4
            end
            __oldb_0d = vbd
            __oldb_0 = vb
            vbd = 0.0
            vb = 0.0
            __hcse_13 = y[i_cell]
            xbd[i_node] = xbd[i_node] + (__oldb_0 * yd[i_cell] + __hcse_13 * __oldb_0d)
            xb[i_node] = xb[i_node] + __hcse_13 * __oldb_0
            __hcse_14 = x[i_node]
            ybd[i_cell] = ybd[i_cell] + (__oldb_0 * xd[i_node] + __hcse_14 * __oldb_0d)
            yb[i_cell] = yb[i_cell] + __hcse_14 * __oldb_0
        end
        vbd = 0.0
        vb = 0.0
    end
    vbd = 0.0
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
