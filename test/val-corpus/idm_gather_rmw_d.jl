function idm_gather_rmw_d(m, md, x, xd, i_cell_to_node, loss, lossd, i_ncell, i_nnode)
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_noded = 0.0
            i_node = i_cell_to_node[i_loc, i_cell]
            md[i_node] = 0.5 * md[i_node] + xd[i_cell]
            m[i_node] = m[i_node] * 0.5 + x[i_cell]
        end
    end
    for i_n = 1:i_nnode
        __cse_0 = m[i_n]
        __cse_1 = __cse_0 * md[i_n]
        lossd[1] = lossd[1] + (__cse_1 + __cse_1)
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    return nothing
end

function idm_gather_rmw(m, x, i_cell_to_node, loss, i_ncell, i_nnode)
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            m[i_node] = m[i_node] * 0.5 + x[i_cell]
        end
    end
    for i_n = 1:i_nnode
        loss[1] = loss[1] + m[i_n] * m[i_n]
    end
end
