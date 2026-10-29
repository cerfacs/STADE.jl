function initstacks_asnap_slice_b(i_nedge, i_nfeat)
    agg_stack = Vector{Float64}(undef, max(0, div(i_nedge - 1, 1) + 1) * max(0, div(i_nfeat - 1, 1) + 1))
    return agg_stack
end

function asnap_slice_b(x, xb, w, wb, dst, agg, aggb, y, yb, i_nedge, i_nnode, i_nnodeb, i_nfeat, agg_stack)
    for e = 1:i_nedge
        d = dst[e]
        for f = 1:i_nfeat
            __idx_agg_stack_0 = ((e - 1) + (f - 1) * (div(i_nedge - 1, 1) + 1)) + 1
            __cse_0 = agg[(d - 1) * i_nfeat + f]
            agg_stack[__idx_agg_stack_0] = __cse_0
            agg[(d - 1) * i_nfeat + f] = __cse_0 + x[(e - 1) * i_nfeat + f] * w[f]
        end
        for f = 1:i_nfeat
            __cse_1 = agg[(d - 1) * i_nfeat + f]
            y[(e - 1) * i_nfeat + f] = __cse_1 * __cse_1
        end
    end
    for e = i_nedge:-1:1
        d = dst[e]
        for f = i_nfeat:-1:1
            __oldb_0 = yb[(e - 1) * i_nfeat + f]
            yb[(e - 1) * i_nfeat + f] = 0.0
            __cse_2 = agg[(d - 1) * i_nfeat + f] * __oldb_0
            aggb[(d - 1) * i_nfeat + f] = aggb[(d - 1) * i_nfeat + f] + __cse_2
            aggb[(d - 1) * i_nfeat + f] = aggb[(d - 1) * i_nfeat + f] + __cse_2
        end
        for f = i_nfeat:-1:1
            __idx_agg_stack_0 = ((e - 1) + (f - 1) * (div(i_nedge - 1, 1) + 1)) + 1
            agg[(d - 1) * i_nfeat + f] = agg_stack[__idx_agg_stack_0]
            __cse_3 = aggb[(d - 1) * i_nfeat + f]
            xb[(e - 1) * i_nfeat + f] = xb[(e - 1) * i_nfeat + f] + w[f] * __cse_3
            wb[f] = wb[f] + x[(e - 1) * i_nfeat + f] * __cse_3
        end
    end
    return i_nnodeb
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
