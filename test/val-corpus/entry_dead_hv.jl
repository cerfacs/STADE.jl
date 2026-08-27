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
            vd = y[i_cell] * xd[i_node] + x[i_node] * yd[i_cell]
            v = x[i_node] * y[i_cell]
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
                resd[i_k_node] = resd[i_k_node] + (v * vd + v * vd)
                res[i_k_node] = res[i_k_node] + v * v
            end
        end
    end
    for i_x = 1:i_nnode
        outd[i_x] = outd[i_x] + (res[i_x] * resd[i_x] + res[i_x] * resd[i_x])
        out[i_x] = out[i_x] + res[i_x] * res[i_x]
    end
    v_stack_d[(max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1) + max(0, div(i_ncell - 1, 1) + 1)) + 1] = vd
    v_stack[(max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1) + max(0, div(i_ncell - 1, 1) + 1)) + 1] = v
    vd = v_stack_d[(max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1) + max(0, div(i_ncell - 1, 1) + 1)) + 1]
    v = v_stack[(max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1) + max(0, div(i_ncell - 1, 1) + 1)) + 1]
    for i_x = i_nnode:-1:1
        resbd[i_x] = resbd[i_x] + (outb[i_x] * resd[i_x] + res[i_x] * outbd[i_x])
        resb[i_x] = resb[i_x] + res[i_x] * outb[i_x]
        resbd[i_x] = resbd[i_x] + (outb[i_x] * resd[i_x] + res[i_x] * outbd[i_x])
        resb[i_x] = resb[i_x] + res[i_x] * outb[i_x]
    end
    for i_cell = 1:i_ncell
        vd = 0.0
        v = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            vd = y[i_cell] * xd[i_node] + x[i_node] * yd[i_cell]
            v = x[i_node] * y[i_cell]
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
            end
        end
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            vd = y[i_cell] * xd[i_node] + x[i_node] * yd[i_cell]
            v = x[i_node] * y[i_cell]
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
            end
            i_node = i_cell_to_node[i_loc, i_cell]
            for i_k = 4:-1:1
                i_k_node = i_cell_to_node[i_k, i_cell]
                vbd = vbd + (resb[i_k_node] * vd + v * resbd[i_k_node])
                vb = vb + v * resb[i_k_node]
                vbd = vbd + (resb[i_k_node] * vd + v * resbd[i_k_node])
                vb = vb + v * resb[i_k_node]
            end
            xbd[i_node] = xbd[i_node] + (vb * yd[i_cell] + y[i_cell] * vbd)
            xb[i_node] = xb[i_node] + y[i_cell] * vb
            ybd[i_cell] = ybd[i_cell] + (vb * xd[i_node] + x[i_node] * vbd)
            yb[i_cell] = yb[i_cell] + x[i_node] * vb
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
