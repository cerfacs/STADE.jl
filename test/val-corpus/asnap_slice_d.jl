function asnap_slice_d(x, xd, w, wd, dst, agg, aggd, y, yd, i_nedge, i_nnode, i_nnoded, i_nfeat)
    for e = 1:i_nedge
        dd = 0.0
        d = dst[e]
        for f = 1:i_nfeat
            __cse_0 = w[f]
            __cse_1 = x[(e - 1) * i_nfeat + f]
            aggd[(d - 1) * i_nfeat + f] = aggd[(d - 1) * i_nfeat + f] + (__cse_0 * xd[(e - 1) * i_nfeat + f] + __cse_1 * wd[f])
            agg[(d - 1) * i_nfeat + f] = agg[(d - 1) * i_nfeat + f] + __cse_1 * __cse_0
        end
        for f = 1:i_nfeat
            __cse_2 = agg[(d - 1) * i_nfeat + f]
            __cse_3 = __cse_2 * aggd[(d - 1) * i_nfeat + f]
            yd[(e - 1) * i_nfeat + f] = __cse_3 + __cse_3
            y[(e - 1) * i_nfeat + f] = __cse_2 * __cse_2
        end
    end
    return nothing
end

function asnap_slice(x, w, dst, agg, y, i_nedge, i_nnode, i_nfeat)
    for e = 1:i_nedge
        d = dst[e]
        for f = 1:i_nfeat
            agg[(d - 1) * i_nfeat + f] = agg[(d - 1) * i_nfeat + f] + x[(e - 1) * i_nfeat + f] * w[f]
        end
        for f = 1:i_nfeat
            y[(e - 1) * i_nfeat + f] = agg[(d - 1) * i_nfeat + f] * agg[(d - 1) * i_nfeat + f]
        end
    end
end
