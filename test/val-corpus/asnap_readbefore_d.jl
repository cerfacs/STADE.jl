function asnap_readbefore_d(x, xd, w, wd, i_cell_to_node, m, md, y, yd, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            __cse_0 = m[i_node]
            __cse_1 = (__cse_0 * 0.3) * md[i_node]
            yd[i_node] = yd[i_node] + (__cse_1 + __cse_1)
            y[i_node] = y[i_node] + __cse_0 * __cse_0 * 0.3
        end
        for i_node = 1:i_nnode
            md[i_node] = 0.0
            m[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_nd = 0.0
                i_n = i_cell_to_node[i_loc, i_cell]
                __cse_2 = x[i_n]
                __cse_3 = w[i_cell]
                md[i_n] = md[i_n] + ((__cse_2 * 0.1) * wd[i_cell] + (__cse_3 * 0.1) * xd[i_n])
                m[i_n] = m[i_n] + __cse_3 * __cse_2 * 0.1
            end
        end
        for i_node = 1:i_nnode
            xd[i_node] = xd[i_node] + 0.5 * md[i_node]
            x[i_node] = x[i_node] + m[i_node] * 0.5
        end
    end
    return nothing
end

function asnap_readbefore(x, w, i_cell_to_node, m, y, i_ncell, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            y[i_node] = y[i_node] + m[i_node] * m[i_node] * 0.3
        end
        for i_node = 1:i_nnode
            m[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            for i_loc = 1:3
                i_n = i_cell_to_node[i_loc, i_cell]
                m[i_n] = m[i_n] + w[i_cell] * x[i_n] * 0.1
            end
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * 0.5
        end
    end
end
