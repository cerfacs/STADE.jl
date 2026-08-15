function ttgc_d(u, ud, uref, urefd, i_cell_to_node, cell_vol, cell_vold, node_vol, node_vold, skx, skxd, sky, skyd, skz, skzd, i_ncell, i_nnode, c, cd, dt, dtd, beta, betad, gamma, gammad, i_njac, res, resd, res2, res2d, up, upd, mup, mupd, npernode_half, resperio, resperiod, i_node_perio, loss, lossd)
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
            vered = (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])) * ud[i_node] + (-(1.0 / 3.0) * u[i_node]) * (((skx[i_loc, i_cell] * cd[1, i_node] + c[1, i_node] * skxd[i_loc, i_cell]) + (sky[i_loc, i_cell] * cd[2, i_node] + c[2, i_node] * skyd[i_loc, i_cell])) + (skz[i_loc, i_cell] * cd[3, i_node] + c[3, i_node] * skzd[i_loc, i_cell]))
            vere = -(1.0 / 3.0) * u[i_node] * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])
            red = (1.0 / cell_vol[i_cell]) * vered + -(vere / cell_vol[i_cell] ^ 2) * cell_vold[i_cell]
            re = vere / cell_vol[i_cell]
            aerexd = re * cavgxd + cavgx * red
            aerex = cavgx * re
            aereyd = re * cavgyd + cavgy * red
            aerey = cavgy * re
            aerezd = re * cavgzd + cavgz * red
            aerez = cavgz * re
            resid = (((0.5 - gamma[i_cell]) * vere) * -(0.25dtd) + (-(dt / 4) * vere) * -(gammad[i_cell])) + (-(dt / 4) * (0.5 - gamma[i_cell])) * vered
            resi = -(dt / 4) * (0.5 - gamma[i_cell]) * vere
            for i_k = 1:4
                aereskd = ((skx[i_k, i_cell] * aerexd + aerex * skxd[i_k, i_cell]) + (sky[i_k, i_cell] * aereyd + aerey * skyd[i_k, i_cell])) + (skz[i_k, i_cell] * aerezd + aerez * skzd[i_k, i_cell])
                aeresk = aerex * skx[i_k, i_cell] + aerey * sky[i_k, i_cell] + aerez * skz[i_k, i_cell]
                factord = aeresk * (0.3333333333333333 * ((2dt) * dtd)) + (dt ^ 2 / 3) * aereskd
                factor = (dt ^ 2 / 3) * aeresk
                auxresd = resid + (beta[i_cell] * factord + factor * betad[i_cell])
                auxres = resi + factor * beta[i_cell]
                auxres2d = gamma[i_cell] * factord + factor * gammad[i_cell]
                auxres2 = factor * gamma[i_cell]
                i_k_noded = 0.0
                i_k_node = i_cell_to_node[i_k, i_cell]
                resd[i_k_node] = resd[i_k_node] + auxresd
                res[i_k_node] = res[i_k_node] + auxres
                res2d[i_k_node] = res2d[i_k_node] + auxres2d
                res2[i_k_node] = res2[i_k_node] + auxres2
            end
        end
    end
    for k = 1:npernode_half
        i1d = 0.0
        i1 = i_node_perio[k, 1]
        i2d = 0.0
        i2 = i_node_perio[k, 2]
        resperiod[k, 1] = resd[i1]
        resperio[k, 1] = res[i1]
        resperiod[k, 2] = resd[i2]
        resperio[k, 2] = res[i2]
    end
    for k = 1:npernode_half
        i1d = 0.0
        i1 = i_node_perio[k, 1]
        i2d = 0.0
        i2 = i_node_perio[k, 2]
        resd[i1] = resd[i1] + resperiod[k, 2]
        res[i1] = res[i1] + resperio[k, 2]
        resd[i2] = resd[i2] + resperiod[k, 1]
        res[i2] = res[i2] + resperio[k, 1]
    end
    for i_node = 1:i_nnode
        upd[i_node] = (1.0 / node_vol[i_node]) * resd[i_node] + -(res[i_node] / node_vol[i_node] ^ 2) * node_vold[i_node]
        up[i_node] = res[i_node] / node_vol[i_node]
    end
    for i_seq_ = 1:i_njac
        for i_node = 1:i_nnode
            mupd[i_node] = 0.0
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            i_node1d = 0.0
            i_node1 = i_cell_to_node[1, i_cell]
            i_node2d = 0.0
            i_node2 = i_cell_to_node[2, i_cell]
            i_node3d = 0.0
            i_node3 = i_cell_to_node[3, i_cell]
            i_node4d = 0.0
            i_node4 = i_cell_to_node[4, i_cell]
            auxud = ((upd[i_node1] + upd[i_node2]) + upd[i_node3]) + upd[i_node4]
            auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
            mupd[i_node1] = mupd[i_node1] + 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node1]) + (auxu + up[i_node1]) * cell_vold[i_cell])
            mup[i_node1] = mup[i_node1] + ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20
            mupd[i_node2] = mupd[i_node2] + 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node2]) + (auxu + up[i_node2]) * cell_vold[i_cell])
            mup[i_node2] = mup[i_node2] + ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20
            mupd[i_node3] = mupd[i_node3] + 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node3]) + (auxu + up[i_node3]) * cell_vold[i_cell])
            mup[i_node3] = mup[i_node3] + ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20
            mupd[i_node4] = mupd[i_node4] + 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node4]) + (auxu + up[i_node4]) * cell_vold[i_cell])
            mup[i_node4] = mup[i_node4] + ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20
        end
        for k = 1:npernode_half
            i1d = 0.0
            i1 = i_node_perio[k, 1]
            i2d = 0.0
            i2 = i_node_perio[k, 2]
            resperiod[k, 1] = mupd[i1]
            resperio[k, 1] = mup[i1]
            resperiod[k, 2] = mupd[i2]
            resperio[k, 2] = mup[i2]
        end
        for k = 1:npernode_half
            i1d = 0.0
            i1 = i_node_perio[k, 1]
            i2d = 0.0
            i2 = i_node_perio[k, 2]
            mupd[i1] = mupd[i1] + resperiod[k, 2]
            mup[i1] = mup[i1] + resperio[k, 2]
            mupd[i2] = mupd[i2] + resperiod[k, 1]
            mup[i2] = mup[i2] + resperio[k, 1]
        end
        for i_node = 1:i_nnode
            upd[i_node] = upd[i_node] + ((1.0 / node_vol[i_node]) * (resd[i_node] + -(mupd[i_node])) + -((res[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * node_vold[i_node])
            up[i_node] = up[i_node] + (res[i_node] - mup[i_node]) / node_vol[i_node]
        end
    end
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_noded = 0.0
            i_node = i_cell_to_node[i_loc, i_cell]
            vered = (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])) * (ud[i_node] + upd[i_node]) + (-(1.0 / 3.0) * (u[i_node] + up[i_node])) * ((skx[i_loc, i_cell] * cd[1, i_node] + c[1, i_node] * skxd[i_loc, i_cell]) + (sky[i_loc, i_cell] * cd[2, i_node] + c[2, i_node] * skyd[i_loc, i_cell]))
            vere = -(1.0 / 3.0) * (u[i_node] + up[i_node]) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])
            resid = vere * -(0.25dtd) + -(dt / 4) * vered
            resi = -(dt / 4) * vere
            for i_k = 1:4
                i_k_noded = 0.0
                i_k_node = i_cell_to_node[i_k, i_cell]
                res2d[i_k_node] = res2d[i_k_node] + resid
                res2[i_k_node] = res2[i_k_node] + resi
            end
        end
    end
    for k = 1:npernode_half
        i1d = 0.0
        i1 = i_node_perio[k, 1]
        i2d = 0.0
        i2 = i_node_perio[k, 2]
        resperiod[k, 1] = res2d[i1]
        resperio[k, 1] = res2[i1]
        resperiod[k, 2] = res2d[i2]
        resperio[k, 2] = res2[i2]
    end
    for k = 1:npernode_half
        i1d = 0.0
        i1 = i_node_perio[k, 1]
        i2d = 0.0
        i2 = i_node_perio[k, 2]
        res2d[i1] = res2d[i1] + resperiod[k, 2]
        res2[i1] = res2[i1] + resperio[k, 2]
        res2d[i2] = res2d[i2] + resperiod[k, 1]
        res2[i2] = res2[i2] + resperio[k, 1]
    end
    for i_node = 1:i_nnode
        upd[i_node] = (1.0 / node_vol[i_node]) * res2d[i_node] + -(res2[i_node] / node_vol[i_node] ^ 2) * node_vold[i_node]
        up[i_node] = res2[i_node] / node_vol[i_node]
    end
    for i_seq_ = 1:i_njac
        for i_node = 1:i_nnode
            mupd[i_node] = 0.0
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            i_node1d = 0.0
            i_node1 = i_cell_to_node[1, i_cell]
            i_node2d = 0.0
            i_node2 = i_cell_to_node[2, i_cell]
            i_node3d = 0.0
            i_node3 = i_cell_to_node[3, i_cell]
            i_node4d = 0.0
            i_node4 = i_cell_to_node[4, i_cell]
            auxud = ((upd[i_node1] + upd[i_node2]) + upd[i_node3]) + upd[i_node4]
            auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
            mupd[i_node1] = mupd[i_node1] + 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node1]) + (auxu + up[i_node1]) * cell_vold[i_cell])
            mup[i_node1] = mup[i_node1] + ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20
            mupd[i_node2] = mupd[i_node2] + 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node2]) + (auxu + up[i_node2]) * cell_vold[i_cell])
            mup[i_node2] = mup[i_node2] + ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20
            mupd[i_node3] = mupd[i_node3] + 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node3]) + (auxu + up[i_node3]) * cell_vold[i_cell])
            mup[i_node3] = mup[i_node3] + ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20
            mupd[i_node4] = mupd[i_node4] + 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node4]) + (auxu + up[i_node4]) * cell_vold[i_cell])
            mup[i_node4] = mup[i_node4] + ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20
        end
        for k = 1:npernode_half
            i1d = 0.0
            i1 = i_node_perio[k, 1]
            i2d = 0.0
            i2 = i_node_perio[k, 2]
            resperiod[k, 1] = mupd[i1]
            resperio[k, 1] = mup[i1]
            resperiod[k, 2] = mupd[i2]
            resperio[k, 2] = mup[i2]
        end
        for k = 1:npernode_half
            i1d = 0.0
            i1 = i_node_perio[k, 1]
            i2d = 0.0
            i2 = i_node_perio[k, 2]
            mupd[i1] = mupd[i1] + resperiod[k, 2]
            mup[i1] = mup[i1] + resperio[k, 2]
            mupd[i2] = mupd[i2] + resperiod[k, 1]
            mup[i2] = mup[i2] + resperio[k, 1]
        end
        for i_node = 1:i_nnode
            upd[i_node] = upd[i_node] + ((1.0 / node_vol[i_node]) * (res2d[i_node] + -(mupd[i_node])) + -((res2[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * node_vold[i_node])
            up[i_node] = up[i_node] + (res2[i_node] - mup[i_node]) / node_vol[i_node]
        end
    end
    for i_seq_node = 1:i_nnode
        lossd[1] = lossd[1] + (2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * ((ud[i_seq_node] + upd[i_seq_node]) + -(urefd[i_seq_node]))
        loss[1] = loss[1] + ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node]) ^ 2
    end
    return nothing
end

function ttgc(u, uref, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup, npernode_half, resperio, i_node_perio, loss)
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
    for k = 1:npernode_half
        i1 = i_node_perio[k, 1]
        i2 = i_node_perio[k, 2]
        resperio[k, 1] = res[i1]
        resperio[k, 2] = res[i2]
    end
    for k = 1:npernode_half
        i1 = i_node_perio[k, 1]
        i2 = i_node_perio[k, 2]
        res[i1] = res[i1] + resperio[k, 2]
        res[i2] = res[i2] + resperio[k, 1]
    end
    for i_node = 1:i_nnode
        up[i_node] = res[i_node] / node_vol[i_node]
    end
    for i_seq_ = 1:i_njac
        for i_node = 1:i_nnode
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            i_node1 = i_cell_to_node[1, i_cell]
            i_node2 = i_cell_to_node[2, i_cell]
            i_node3 = i_cell_to_node[3, i_cell]
            i_node4 = i_cell_to_node[4, i_cell]
            auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
            mup[i_node1] = mup[i_node1] + ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20
            mup[i_node2] = mup[i_node2] + ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20
            mup[i_node3] = mup[i_node3] + ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20
            mup[i_node4] = mup[i_node4] + ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20
        end
        for k = 1:npernode_half
            i1 = i_node_perio[k, 1]
            i2 = i_node_perio[k, 2]
            resperio[k, 1] = mup[i1]
            resperio[k, 2] = mup[i2]
        end
        for k = 1:npernode_half
            i1 = i_node_perio[k, 1]
            i2 = i_node_perio[k, 2]
            mup[i1] = mup[i1] + resperio[k, 2]
            mup[i2] = mup[i2] + resperio[k, 1]
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
    for k = 1:npernode_half
        i1 = i_node_perio[k, 1]
        i2 = i_node_perio[k, 2]
        resperio[k, 1] = res2[i1]
        resperio[k, 2] = res2[i2]
    end
    for k = 1:npernode_half
        i1 = i_node_perio[k, 1]
        i2 = i_node_perio[k, 2]
        res2[i1] = res2[i1] + resperio[k, 2]
        res2[i2] = res2[i2] + resperio[k, 1]
    end
    for i_node = 1:i_nnode
        up[i_node] = res2[i_node] / node_vol[i_node]
    end
    for i_seq_ = 1:i_njac
        for i_node = 1:i_nnode
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            i_node1 = i_cell_to_node[1, i_cell]
            i_node2 = i_cell_to_node[2, i_cell]
            i_node3 = i_cell_to_node[3, i_cell]
            i_node4 = i_cell_to_node[4, i_cell]
            auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
            mup[i_node1] = mup[i_node1] + ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20
            mup[i_node2] = mup[i_node2] + ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20
            mup[i_node3] = mup[i_node3] + ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20
            mup[i_node4] = mup[i_node4] + ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20
        end
        for k = 1:npernode_half
            i1 = i_node_perio[k, 1]
            i2 = i_node_perio[k, 2]
            resperio[k, 1] = mup[i1]
            resperio[k, 2] = mup[i2]
        end
        for k = 1:npernode_half
            i1 = i_node_perio[k, 1]
            i2 = i_node_perio[k, 2]
            mup[i1] = mup[i1] + resperio[k, 2]
            mup[i2] = mup[i2] + resperio[k, 1]
        end
        for i_node = 1:i_nnode
            up[i_node] = up[i_node] + (res2[i_node] - mup[i_node]) / node_vol[i_node]
        end
    end
    for i_seq_node = 1:i_nnode
        loss[1] = loss[1] + ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node]) ^ 2
    end
end
