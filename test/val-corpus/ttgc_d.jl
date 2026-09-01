function ttgc_d(u, ud, uref, urefd, i_cell_to_node, cell_vol, cell_vold, node_vol, node_vold, skx, skxd, sky, skyd, skz, skzd, i_ncell, i_nnode, c, cd, dt, dtd, beta, betad, gamma, gammad, i_njac, res, resd, res2, res2d, up, upd, mup, mupd, loss, lossd)
    for i_cell = 1:i_ncell
        cavgxd = 0.0
        cavgx = 0.0
        cavgyd = 0.0
        cavgy = 0.0
        cavgzd = 0.0
        cavgz = 0.0
        for i_loc = 1:4
            i_noded = 0.0
            i_node = i_cell_to_node[i_loc, i_cell]
            cavgxd = cavgxd + 0.25 * cd[1, i_node]
            cavgx = cavgx + c[1, i_node] / 4
            cavgyd = cavgyd + 0.25 * cd[2, i_node]
            cavgy = cavgy + c[2, i_node] / 4
            cavgzd = cavgzd + 0.25 * cd[3, i_node]
            cavgz = cavgz + c[3, i_node] / 4
        end
        for i_loc = 1:4
            i_noded = 0.0
            i_node = i_cell_to_node[i_loc, i_cell]
            __cse_0 = c[1, i_node]
            __cse_1 = skx[i_loc, i_cell]
            __cse_2 = c[2, i_node]
            __cse_3 = sky[i_loc, i_cell]
            __cse_4 = c[3, i_node]
            __cse_5 = skz[i_loc, i_cell]
            __cse_6 = __cse_0 * __cse_1 + __cse_2 * __cse_3 + __cse_4 * __cse_5
            __cse_7 = u[i_node]
            vered = (-(1.0 / 3.0) * __cse_6) * ud[i_node] + (-(1.0 / 3.0) * __cse_7) * (((__cse_1 * cd[1, i_node] + __cse_0 * skxd[i_loc, i_cell]) + (__cse_3 * cd[2, i_node] + __cse_2 * skyd[i_loc, i_cell])) + (__cse_5 * cd[3, i_node] + __cse_4 * skzd[i_loc, i_cell]))
            vere = -(1.0 / 3.0) * __cse_7 * __cse_6
            __cse_8 = cell_vol[i_cell]
            red = (1.0 / __cse_8) * vered + -(vere / __cse_8 ^ 2) * cell_vold[i_cell]
            re = vere / __cse_8
            aerexd = re * cavgxd + cavgx * red
            aerex = cavgx * re
            aereyd = re * cavgyd + cavgy * red
            aerey = cavgy * re
            aerezd = re * cavgzd + cavgz * red
            aerez = cavgz * re
            __cse_9 = 0.5 - gamma[i_cell]
            __cse_10 = -(dt / 4)
            resid = ((__cse_9 * vere) * -(0.25dtd) + (__cse_10 * vere) * -(gammad[i_cell])) + (__cse_10 * __cse_9) * vered
            resi = __cse_10 * __cse_9 * vere
            for i_k = 1:4
                __cse_11 = skx[i_k, i_cell]
                __cse_12 = sky[i_k, i_cell]
                __cse_13 = skz[i_k, i_cell]
                aereskd = ((__cse_11 * aerexd + aerex * skxd[i_k, i_cell]) + (__cse_12 * aereyd + aerey * skyd[i_k, i_cell])) + (__cse_13 * aerezd + aerez * skzd[i_k, i_cell])
                aeresk = aerex * __cse_11 + aerey * __cse_12 + aerez * __cse_13
                __cse_14 = dt ^ 2 / 3
                factord = aeresk * (0.3333333333333333 * ((2dt) * dtd)) + __cse_14 * aereskd
                factor = __cse_14 * aeresk
                __cse_15 = beta[i_cell]
                auxresd = resid + (__cse_15 * factord + factor * betad[i_cell])
                auxres = resi + factor * __cse_15
                __cse_16 = gamma[i_cell]
                auxres2d = __cse_16 * factord + factor * gammad[i_cell]
                auxres2 = factor * __cse_16
                i_k_noded = 0.0
                i_k_node = i_cell_to_node[i_k, i_cell]
                resd[i_k_node] = resd[i_k_node] + auxresd
                res[i_k_node] = res[i_k_node] + auxres
                res2d[i_k_node] = res2d[i_k_node] + auxres2d
                res2[i_k_node] = res2[i_k_node] + auxres2
            end
        end
    end
    for i_node = 1:i_nnode
        __cse_17 = node_vol[i_node]
        __cse_18 = res[i_node]
        upd[i_node] = (1.0 / __cse_17) * resd[i_node] + -(__cse_18 / __cse_17 ^ 2) * node_vold[i_node]
        up[i_node] = __cse_18 / __cse_17
    end
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            mupd[i_node] = 0.0
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            auxud = 0.0
            auxu = 0.0
            for i_loc = 1:4
                i_lnoded = 0.0
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxud = auxud + upd[i_lnode]
                auxu = auxu + up[i_lnode]
            end
            for i_loc = 1:4
                i_lnoded = 0.0
                i_lnode = i_cell_to_node[i_loc, i_cell]
                __cse_19 = cell_vol[i_cell]
                __cse_20 = auxu + up[i_lnode]
                mupd[i_lnode] = mupd[i_lnode] + 0.05 * (__cse_19 * (auxud + upd[i_lnode]) + __cse_20 * cell_vold[i_cell])
                mup[i_lnode] = mup[i_lnode] + (__cse_20 * __cse_19) / 20.0
            end
        end
        for i_node = 1:i_nnode
            __cse_21 = node_vol[i_node]
            __cse_22 = res[i_node] - mup[i_node]
            upd[i_node] = upd[i_node] + ((1.0 / __cse_21) * (resd[i_node] + -(mupd[i_node])) + -(__cse_22 / __cse_21 ^ 2) * node_vold[i_node])
            up[i_node] = up[i_node] + __cse_22 / __cse_21
        end
    end
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_noded = 0.0
            i_node = i_cell_to_node[i_loc, i_cell]
            __cse_23 = c[1, i_node]
            __cse_24 = skx[i_loc, i_cell]
            __cse_25 = c[2, i_node]
            __cse_26 = sky[i_loc, i_cell]
            __cse_27 = __cse_23 * __cse_24 + __cse_25 * __cse_26
            __cse_28 = u[i_node] + up[i_node]
            vered = (-(1.0 / 3.0) * __cse_27) * (ud[i_node] + upd[i_node]) + (-(1.0 / 3.0) * __cse_28) * ((__cse_24 * cd[1, i_node] + __cse_23 * skxd[i_loc, i_cell]) + (__cse_26 * cd[2, i_node] + __cse_25 * skyd[i_loc, i_cell]))
            vere = -(1.0 / 3.0) * __cse_28 * __cse_27
            __cse_29 = -(dt / 4)
            resid = vere * -(0.25dtd) + __cse_29 * vered
            resi = __cse_29 * vere
            for i_k = 1:4
                i_k_noded = 0.0
                i_k_node = i_cell_to_node[i_k, i_cell]
                res2d[i_k_node] = res2d[i_k_node] + resid
                res2[i_k_node] = res2[i_k_node] + resi
            end
        end
    end
    for i_node = 1:i_nnode
        __cse_30 = node_vol[i_node]
        __cse_31 = res2[i_node]
        upd[i_node] = (1.0 / __cse_30) * res2d[i_node] + -(__cse_31 / __cse_30 ^ 2) * node_vold[i_node]
        up[i_node] = __cse_31 / __cse_30
    end
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            mupd[i_node] = 0.0
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            auxud = 0.0
            auxu = 0.0
            for i_loc = 1:4
                i_lnoded = 0.0
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxud = auxud + upd[i_lnode]
                auxu = auxu + up[i_lnode]
            end
            for i_loc = 1:4
                i_lnoded = 0.0
                i_lnode = i_cell_to_node[i_loc, i_cell]
                __cse_32 = cell_vol[i_cell]
                __cse_33 = auxu + up[i_lnode]
                mupd[i_lnode] = mupd[i_lnode] + 0.05 * (__cse_32 * (auxud + upd[i_lnode]) + __cse_33 * cell_vold[i_cell])
                mup[i_lnode] = mup[i_lnode] + (__cse_33 * __cse_32) / 20.0
            end
        end
        for i_node = 1:i_nnode
            __cse_34 = node_vol[i_node]
            __cse_35 = res2[i_node] - mup[i_node]
            upd[i_node] = upd[i_node] + ((1.0 / __cse_34) * (res2d[i_node] + -(mupd[i_node])) + -(__cse_35 / __cse_34 ^ 2) * node_vold[i_node])
            up[i_node] = up[i_node] + __cse_35 / __cse_34
        end
    end
    for i_node2 = 1:i_nnode
        __cse_36 = (u[i_node2] + up[i_node2]) - uref[i_node2]
        lossd[1] = lossd[1] + (2__cse_36) * ((ud[i_node2] + upd[i_node2]) + -(urefd[i_node2]))
        loss[1] = loss[1] + __cse_36 ^ 2
    end
    return nothing
end

function ttgc(u, uref, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup, loss)
    for i_cell = 1:i_ncell
        cavgx = 0.0
        cavgy = 0.0
        cavgz = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            cavgx = cavgx + c[1, i_node] / 4
            cavgy = cavgy + c[2, i_node] / 4
            cavgz = cavgz + c[3, i_node] / 4
        end
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            vere = -(1.0 / 3.0) * u[i_node] * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])
            re = vere / cell_vol[i_cell]
            aerex = cavgx * re
            aerey = cavgy * re
            aerez = cavgz * re
            resi = -(dt / 4) * (0.5 - gamma[i_cell]) * vere
            for i_k = 1:4
                aeresk = aerex * skx[i_k, i_cell] + aerey * sky[i_k, i_cell] + aerez * skz[i_k, i_cell]
                factor = (dt ^ 2 / 3) * aeresk
                auxres = resi + factor * beta[i_cell]
                auxres2 = factor * gamma[i_cell]
                i_k_node = i_cell_to_node[i_k, i_cell]
                res[i_k_node] = res[i_k_node] + auxres
                res2[i_k_node] = res2[i_k_node] + auxres2
            end
        end
    end
    for i_node = 1:i_nnode
        up[i_node] = res[i_node] / node_vol[i_node]
    end
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            auxu = 0.0
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxu = auxu + up[i_lnode]
            end
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                mup[i_lnode] = mup[i_lnode] + ((auxu + up[i_lnode]) * cell_vol[i_cell]) / 20.0
            end
        end
        for i_node = 1:i_nnode
            up[i_node] = up[i_node] + (res[i_node] - mup[i_node]) / node_vol[i_node]
        end
    end
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            vere = -(1.0 / 3.0) * (u[i_node] + up[i_node]) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])
            resi = -(dt / 4) * vere
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
                res2[i_k_node] = res2[i_k_node] + resi
            end
        end
    end
    for i_node = 1:i_nnode
        up[i_node] = res2[i_node] / node_vol[i_node]
    end
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            auxu = 0.0
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxu = auxu + up[i_lnode]
            end
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                mup[i_lnode] = mup[i_lnode] + ((auxu + up[i_lnode]) * cell_vol[i_cell]) / 20.0
            end
        end
        for i_node = 1:i_nnode
            up[i_node] = up[i_node] + (res2[i_node] - mup[i_node]) / node_vol[i_node]
        end
    end
    for i_node2 = 1:i_nnode
        loss[1] = loss[1] + ((u[i_node2] + up[i_node2]) - uref[i_node2]) ^ 2
    end
end
