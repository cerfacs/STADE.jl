function ttgc_d(u, ud, i_cell_to_node, cell_vol, cell_vold, skx, skxd, sky, skyd, skz, skzd, i_ncell, i_nnode, c, cd, dt, dtd, beta, betad, gamma, gammad, res, resd, up, upd, loss, lossd)
    for i_cell = 1:i_ncell
        cavgxd = 0.0
        cavgx = 0.0
        for i_loc = 1:4
            i_noded = 0.0
            i_node = i_cell_to_node[i_loc, i_cell]
            cavgxd = cavgxd + 0.25 * cd[1, i_node]
            cavgx = cavgx + c[1, i_node] / 4
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
            resid = (((0.5 - gamma[i_cell]) * vere) * -(0.25dtd) + (-(dt / 4) * vere) * -(gammad[i_cell])) + (-(dt / 4) * (0.5 - gamma[i_cell])) * vered
            resi = -(dt / 4) * (0.5 - gamma[i_cell]) * vere
            for i_k = 1:4
                aereskd = ((skx[i_k, i_cell] * aerexd + aerex * skxd[i_k, i_cell]) + aerey * skyd[i_k, i_cell]) + aerez * skzd[i_k, i_cell]
                aeresk = aerex * skx[i_k, i_cell] + aerey * sky[i_k, i_cell] + aerez * skz[i_k, i_cell]
                factord = aeresk * (0.3333333333333333 * ((2dt) * dtd)) + (dt ^ 2 / 3) * aereskd
                factor = (dt ^ 2 / 3) * aeresk
                auxresd = resid + (beta[i_cell] * factord + factor * betad[i_cell])
                auxres = resi + factor * beta[i_cell]
                i_k_noded = 0.0
                i_k_node = i_cell_to_node[i_k, i_cell]
                resd[i_k_node] = resd[i_k_node] + auxresd
                res[i_k_node] = res[i_k_node] + auxres
            end
        end
    end
    for i_seq_node = 1:i_nnode
        lossd[1] = lossd[1] + (2 * (u[i_seq_node] + up[i_seq_node])) * (ud[i_seq_node] + upd[i_seq_node])
        loss[1] = loss[1] + (u[i_seq_node] + up[i_seq_node]) ^ 2
    end
    return nothing
end

function ttgc(u, i_cell_to_node, cell_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, res, up, loss)
    for i_cell = 1:i_ncell
        cavgx = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            cavgx = cavgx + c[1, i_node] / 4
        end
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            vere = -(1.0 / 3.0) * u[i_node] * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])
            re = vere / cell_vol[i_cell]
            aerex = cavgx * re
            resi = -(dt / 4) * (0.5 - gamma[i_cell]) * vere
            for i_k = 1:4
                aeresk = aerex * skx[i_k, i_cell] + aerey * sky[i_k, i_cell] + aerez * skz[i_k, i_cell]
                factor = (dt ^ 2 / 3) * aeresk
                auxres = resi + factor * beta[i_cell]
                i_k_node = i_cell_to_node[i_k, i_cell]
                res[i_k_node] = res[i_k_node] + auxres
            end
        end
    end
    for i_seq_node = 1:i_nnode
        loss[1] = loss[1] + (u[i_seq_node] + up[i_seq_node]) ^ 2
    end
end
