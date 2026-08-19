function initstacks_ttgc_b()
    return
end
function ttgc_b(u, ub, i_cell_to_node, cell_vol, cell_volb, skx, skxb, sky, skyb, skz, skzb, i_ncell, i_nnode, c, cb, dt, dtb, beta, betab, gamma, gammab, res, resb, up, upb, loss, lossb)
    for i_seq_node = i_nnode:-1:1
        tempb = 2 * (u[i_seq_node] + up[i_seq_node]) * lossb[1]
        ub[i_seq_node] = ub[i_seq_node] + tempb
        upb[i_seq_node] = upb[i_seq_node] + tempb
    end
    for i_cell = 1:i_ncell
        cavgx = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            cavgx = cavgx + c[1, i_node] / 4
        end
        cavgxb = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            vere = -((1.0 / 3.0) * u[i_node] * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell]))
            re = vere / cell_vol[i_cell]
            aerex = cavgx * re
            resib = 0.0
            aerexb = 0.0
            for i_k = 1:4
                aeresk = aerex * skx[i_k, i_cell] + aerey * sky[i_k, i_cell] + aerez * skz[i_k, i_cell]
                factor = aeresk * dt ^ 2
                i_k_node = i_cell_to_node[i_k, i_cell]
                auxresb = resb[i_k_node]
                resib = resib + auxresb
                factorb = beta[i_cell] * auxresb
                betab[i_cell] = betab[i_cell] + factor * auxresb
                aereskb = dt ^ 2 * factorb
                dtb = dtb + 2 * dt * aeresk * factorb
                aerexb = aerexb + skx[i_k, i_cell] * aereskb
                skxb[i_k, i_cell] = skxb[i_k, i_cell] + aerex * aereskb
                skyb[i_k, i_cell] = skyb[i_k, i_cell] + aerey * aereskb
                skzb[i_k, i_cell] = skzb[i_k, i_cell] + aerez * aereskb
            end
            dtb = dtb - ((0.5 - gamma[i_cell]) * vere * resib) / 4
            tempb = -((dt * resib) / 4)
            gammab[i_cell] = gammab[i_cell] - vere * tempb
            vereb = (0.5 - gamma[i_cell]) * tempb
            cavgxb = cavgxb + re * aerexb
            reb = cavgx * aerexb
            tempb = reb / cell_vol[i_cell]
            vereb = vereb + tempb
            cell_volb[i_cell] = cell_volb[i_cell] - (vere * tempb) / cell_vol[i_cell]
            ub[i_node] = ub[i_node] - ((c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell]) * vereb) / 3.0
            tempb = -((u[i_node] * vereb) / 3.0)
            cb[1, i_node] = cb[1, i_node] + skx[i_loc, i_cell] * tempb
            skxb[i_loc, i_cell] = skxb[i_loc, i_cell] + c[1, i_node] * tempb
            cb[2, i_node] = cb[2, i_node] + sky[i_loc, i_cell] * tempb
            skyb[i_loc, i_cell] = skyb[i_loc, i_cell] + c[2, i_node] * tempb
            cb[3, i_node] = cb[3, i_node] + skz[i_loc, i_cell] * tempb
            skzb[i_loc, i_cell] = skzb[i_loc, i_cell] + c[3, i_node] * tempb
        end
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            cb[1, i_node] = cb[1, i_node] + cavgxb / 4
        end
    end
    return dtb
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
            vere = -((1.0 / 3.0) * u[i_node] * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell]))
            re = vere / cell_vol[i_cell]
            aerex = cavgx * re
            resi = -((dt / 4) * (0.5 - gamma[i_cell]) * vere)
            for i_k = 1:4
                aeresk = aerex * skx[i_k, i_cell] + aerey * sky[i_k, i_cell] + aerez * skz[i_k, i_cell]
                factor = aeresk * dt ^ 2
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
