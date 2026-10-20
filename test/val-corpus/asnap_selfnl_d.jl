function asnap_selfnl_d(x, xd, w, wd, i_cell_to_node, m, md, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            md[i_node] = 0.0
            m[i_node] = 1.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_nd = 0.0
                i_n = i_cell_to_node[i_loc, i_cell]
                __cse_0 = w[i_cell]
                __cse_1 = x[i_n]
                __cse_2 = 1.0 + __cse_0 * __cse_1 * 0.05
                __cse_3 = m[i_n]
                md[i_n] = __cse_2 * md[i_n] + __cse_3 * ((__cse_1 * 0.05) * wd[i_cell] + (__cse_0 * 0.05) * xd[i_n])
                m[i_n] = __cse_3 * __cse_2
            end
        end
        for i_node = 1:i_nnode
            __cse_4 = m[i_node]
            __cse_5 = (__cse_4 * 0.1) * md[i_node]
            xd[i_node] = xd[i_node] + (__cse_5 + __cse_5)
            x[i_node] = x[i_node] + __cse_4 * __cse_4 * 0.1
        end
    end
    return nothing
end

function asnap_selfnl(x, w, i_cell_to_node, m, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            m[i_node] = 1.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] * (1.0 + w[i_cell] * x[i_n] * 0.05)
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.1
        end
    end
end
