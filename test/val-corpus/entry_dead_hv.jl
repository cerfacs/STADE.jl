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
            __cse_3 = y[i_cell]
            __cse_4 = x[i_node]
            vd = __cse_3 * xd[i_node] + __cse_4 * yd[i_cell]
            v = __cse_4 * __cse_3
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
                __cse_5 = v * vd
                resd[i_k_node] = resd[i_k_node] + (__cse_5 + __cse_5)
                res[i_k_node] = res[i_k_node] + v * v
            end
        end
    end
    for i_x = 1:i_nnode
        __cse_0 = res[i_x]
        __cse_6 = __cse_0 * resd[i_x]
        outd[i_x] = outd[i_x] + (__cse_6 + __cse_6)
        out[i_x] = out[i_x] + __cse_0 * __cse_0
    end
    __idx_v_stack_3 = (max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1) + max(0, div(i_ncell - 1, 1) + 1)) + 1
    v_stack_d[__idx_v_stack_3] = vd
    v_stack[__idx_v_stack_3] = v
    __idx_v_stack_0 = (max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1) + max(0, div(i_ncell - 1, 1) + 1)) + 1
    vd = v_stack_d[__idx_v_stack_0]
    v = v_stack[__idx_v_stack_0]
    for i_x = i_nnode:-1:1
        __cse_7 = outb[i_x]
        __cse_8 = res[i_x]
        __cse_1d = __cse_7 * resd[i_x] + __cse_8 * outbd[i_x]
        __cse_1 = __cse_8 * __cse_7
        resbd[i_x] = resbd[i_x] + __cse_1d
        resb[i_x] = resb[i_x] + __cse_1
        resbd[i_x] = resbd[i_x] + __cse_1d
        resb[i_x] = resb[i_x] + __cse_1
    end
    for i_cell = 1:i_ncell
        vd = 0.0
        v = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            __cse_9 = y[i_cell]
            __cse_10 = x[i_node]
            vd = __cse_9 * xd[i_node] + __cse_10 * yd[i_cell]
            v = __cse_10 * __cse_9
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
            end
        end
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            __cse_11 = y[i_cell]
            __cse_12 = x[i_node]
            vd = __cse_11 * xd[i_node] + __cse_12 * yd[i_cell]
            v = __cse_12 * __cse_11
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
            end
            i_node = i_cell_to_node[i_loc, i_cell]
            for i_k = 4:-1:1
                i_k_node = i_cell_to_node[i_k, i_cell]
                __cse_13 = resb[i_k_node]
                __cse_2d = __cse_13 * vd + v * resbd[i_k_node]
                __cse_2 = v * __cse_13
                vbd = vbd + __cse_2d
                vb = vb + __cse_2
                vbd = vbd + __cse_2d
                vb = vb + __cse_2
            end
            __cse_14 = y[i_cell]
            xbd[i_node] = xbd[i_node] + (vb * yd[i_cell] + __cse_14 * vbd)
            xb[i_node] = xb[i_node] + __cse_14 * vb
            __cse_15 = x[i_node]
            ybd[i_cell] = ybd[i_cell] + (vb * xd[i_node] + __cse_15 * vbd)
            yb[i_cell] = yb[i_cell] + __cse_15 * vb
            vbd = 0.0
            vb = 0.0
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
