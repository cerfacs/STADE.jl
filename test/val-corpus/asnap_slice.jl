# asnap_slice(x, w, dst, agg, y, i_nedge, i_nnode, i_nfeat)
#
# A loop over edges scatters into agg at the receiver's slice, then reads agg back. The array snapshot would be correct here, but one
# copy of agg per edge costs length(agg) where the per-write saves cost i_nfeat slots. The write index follows the loop variable
# (through dst[e]), so the snapshot must be refused (guard G13).
#
# x: edge feature array, length i_nedge * i_nfeat
# w: weight per feature
# dst: receiver node of each edge, 1-based, length i_nedge
# agg: node accumulator, length i_nnode * i_nfeat, updated in place
# y: edge output, length i_nedge * i_nfeat
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
    return nothing
end
