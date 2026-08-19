function initstacks_ttgc_b()
    cavgx_stack = Vector{Float64}()
    vere_stack = Vector{Float64}()
    re_stack = Vector{Float64}()
    aerex_stack = Vector{Float64}()
    aeresk_stack = Vector{Float64}()
    factor_stack = Vector{Float64}()
    return (cavgx_stack, vere_stack, re_stack, aerex_stack, aeresk_stack, factor_stack)
end

function ttgc_b(u, ub, i_cell_to_node, cell_vol, cell_volb, skx, skxb, i_ncell, i_nnode, c, cb, dt, dtb, beta, betab, gamma, gammab, res, resb, up, upb, loss, lossb, cavgx_stack, vere_stack, re_stack, aerex_stack, aeresk_stack, factor_stack)
    aeresk = 0.0
    aerex = 0.0
    auxres = 0.0
    cavgx = 0.0
    factor = 0.0
    re = 0.0
    resi = 0.0
    vere = 0.0
    aereskb = 0.0
    aerexb = 0.0
    auxresb = 0.0
    cavgxb = 0.0
    factorb = 0.0
    reb = 0.0
    resib = 0.0
    vereb = 0.0
    for i_cell = 1:i_ncell
        push!(cavgx_stack, cavgx)
        cavgx = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            push!(cavgx_stack, cavgx)
            cavgx = cavgx + c[1, i_node] / 4
        end
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            push!(vere_stack, vere)
            vere = -(1.0 / 3.0) * u[i_node] * c[1, i_node] * skx[i_loc, i_cell]
            push!(re_stack, re)
            re = vere / cell_vol[i_cell]
            push!(aerex_stack, aerex)
            aerex = cavgx * re
            resi = -(dt / 4) * (0.5 - gamma[i_cell]) * vere
            for i_k = 1:4
                push!(aeresk_stack, aeresk)
                aeresk = aerex * skx[i_k, i_cell]
                push!(factor_stack, factor)
                factor = aeresk * dt ^ 2
                auxres = resi + factor * beta[i_cell]
                i_k_node = i_cell_to_node[i_k, i_cell]
                res[i_k_node] = res[i_k_node] + auxres
            end
            push!(aeresk_stack, aeresk)
            push!(factor_stack, factor)
        end
        push!(aeresk_stack, aeresk)
        push!(aerex_stack, aerex)
        push!(cavgx_stack, cavgx)
        push!(factor_stack, factor)
        push!(re_stack, re)
        push!(vere_stack, vere)
    end
    for i_seq_node = 1:i_nnode
        loss[1] = loss[1] + (u[i_seq_node] + up[i_seq_node]) ^ 2
    end
    push!(aeresk_stack, aeresk)
    push!(aerex_stack, aerex)
    push!(cavgx_stack, cavgx)
    push!(factor_stack, factor)
    push!(re_stack, re)
    push!(vere_stack, vere)
    aeresk = pop!(aeresk_stack)
    aerex = pop!(aerex_stack)
    cavgx = pop!(cavgx_stack)
    factor = pop!(factor_stack)
    re = pop!(re_stack)
    vere = pop!(vere_stack)
    for i_seq_node = i_nnode:-1:1
        ub[i_seq_node] = ub[i_seq_node] + (2 * (u[i_seq_node] + up[i_seq_node])) * lossb[1]
        upb[i_seq_node] = upb[i_seq_node] + (2 * (u[i_seq_node] + up[i_seq_node])) * lossb[1]
    end
    for i_cell = i_ncell:-1:1
        aeresk = pop!(aeresk_stack)
        aerex = pop!(aerex_stack)
        cavgx = pop!(cavgx_stack)
        factor = pop!(factor_stack)
        re = pop!(re_stack)
        vere = pop!(vere_stack)
        for i_loc = 4:-1:1
            aeresk = pop!(aeresk_stack)
            factor = pop!(factor_stack)
            i_node = i_cell_to_node[i_loc, i_cell]
            for i_k = 4:-1:1
                i_k_node = i_cell_to_node[i_k, i_cell]
                auxresb = auxresb + resb[i_k_node]
                resib = resib + auxresb
                factorb = factorb + beta[i_cell] * auxresb
                betab[i_cell] = betab[i_cell] + factor * auxresb
                auxresb = 0.0
                factor = pop!(factor_stack)
                aereskb = aereskb + dt ^ 2 * factorb
                dtb = dtb + (2dt) * (aeresk * factorb)
                factorb = 0.0
                aeresk = pop!(aeresk_stack)
                aerexb = aerexb + skx[i_k, i_cell] * aereskb
                skxb[i_k, i_cell] = skxb[i_k, i_cell] + aerex * aereskb
                aereskb = 0.0
            end
            dtb = dtb + 0.25 * -(((0.5 - gamma[i_cell]) * vere) * resib)
            gammab[i_cell] = gammab[i_cell] + -((-(dt / 4) * vere) * resib)
            vereb = vereb + (-(dt / 4) * (0.5 - gamma[i_cell])) * resib
            resib = 0.0
            aerex = pop!(aerex_stack)
            cavgxb = cavgxb + re * aerexb
            reb = reb + cavgx * aerexb
            aerexb = 0.0
            re = pop!(re_stack)
            vereb = vereb + (1.0 / cell_vol[i_cell]) * reb
            cell_volb[i_cell] = cell_volb[i_cell] + -(vere / cell_vol[i_cell] ^ 2) * reb
            reb = 0.0
            vere = pop!(vere_stack)
            ub[i_node] = ub[i_node] + ((-(1.0 / 3.0) * c[1, i_node]) * skx[i_loc, i_cell]) * vereb
            cb[1, i_node] = cb[1, i_node] + ((-(1.0 / 3.0) * u[i_node]) * skx[i_loc, i_cell]) * vereb
            skxb[i_loc, i_cell] = skxb[i_loc, i_cell] + ((-(1.0 / 3.0) * u[i_node]) * c[1, i_node]) * vereb
            vereb = 0.0
        end
        for i_loc = 4:-1:1
            i_node = i_cell_to_node[i_loc, i_cell]
            cavgx = pop!(cavgx_stack)
            cb[1, i_node] = cb[1, i_node] + 0.25cavgxb
        end
        cavgx = pop!(cavgx_stack)
        cavgxb = 0.0
    end
    return dtb
end

function ttgc(u, i_cell_to_node, cell_vol, skx, i_ncell, i_nnode, c, dt, beta, gamma, res, up, loss)
    for i_cell = 1:i_ncell
        cavgx = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            cavgx = cavgx + c[1, i_node] / 4
        end
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            vere = -(1.0 / 3.0) * u[i_node] * c[1, i_node] * skx[i_loc, i_cell]
            re = vere / cell_vol[i_cell]
            aerex = cavgx * re
            resi = -(dt / 4) * (0.5 - gamma[i_cell]) * vere
            for i_k = 1:4
                aeresk = aerex * skx[i_k, i_cell]
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
