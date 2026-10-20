function initstacks_idm_cse_acc_b()
    return nothing
end

function idm_cse_acc_hv(m, mb, x, xb, i_cell_to_node, loss, lossb, i_ncell, i_nnode, md, mbd, xd, xbd, lossd, lossbd)
    old = 0.0
    oldb = 0.0
    oldd = 0.0
    oldbd = 0.0
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            oldd = md[i_node]
            old = m[i_node]
            md[i_node] = oldd + 0.5 * xd[i_cell]
            m[i_node] = old + x[i_cell] * 0.5
        end
    end
    for i_n = 1:i_nnode
        __cse_0d = md[i_n]
        __cse_0 = m[i_n]
        __hcse_0 = __cse_0 * __cse_0d
        lossd[1] = lossd[1] + (__hcse_0 + __hcse_0)
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    for i_n = i_nnode:-1:1
        __hcse_1 = lossb[1]
        __hcse_2 = m[i_n]
        __cse_1d = __hcse_1 * md[i_n] + __hcse_2 * lossbd[1]
        __cse_1 = __hcse_2 * __hcse_1
        mbd[i_n] = mbd[i_n] + __cse_1d
        mb[i_n] = mb[i_n] + __cse_1
        mbd[i_n] = mbd[i_n] + __cse_1d
        mb[i_n] = mb[i_n] + __cse_1
    end
    for i_cell = i_ncell:-1:1
        for i_loc = 4:-1:1
            i_node = i_cell_to_node[i_loc, i_cell]
            __oldb_0d = mbd[i_node]
            __oldb_0 = mb[i_node]
            mbd[i_node] = 0.0
            mb[i_node] = 0.0
            oldbd = oldbd + __oldb_0d
            oldb = oldb + __oldb_0
            xbd[i_cell] = xbd[i_cell] + 0.5__oldb_0d
            xb[i_cell] = xb[i_cell] + 0.5__oldb_0
            __oldb_0d = oldbd
            __oldb_0 = oldb
            oldbd = 0.0
            oldb = 0.0
            mbd[i_node] = mbd[i_node] + __oldb_0d
            mb[i_node] = mb[i_node] + __oldb_0
        end
    end
    return nothing
end

function idm_cse_acc(m, x, i_cell_to_node, loss, i_ncell, i_nnode)
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            old = m[i_node]
            m[i_node] = old + x[i_cell] * 0.5
        end
    end
    for i_n = 1:i_nnode
        loss[1] = loss[1] + m[i_n] * m[i_n]
    end
end
