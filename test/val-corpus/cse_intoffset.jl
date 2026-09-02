# cse_intoffset(out, u, i_nrow, i_base)
#
# Coverage kernel for integer temporaries -- the newest path through
# emit_cse_bindable, and the one that first tripped validate_cse's shadow
# invariant on transformer.
#
# The row offset arithmetic repeats verbatim across three statements in one
# run and mentions the loop variable, so it is bindable only because it sits
# outside any array subscript. Where the same arithmetic DOES appear inside a
# subscript it must be left inline, since that text is what cgen_/jgen_ read
# to map threads and decide atomics.
#
# The resulting `__icse_*` temporaries are integer-valued and must never be
# given a second-order shadow in the HVP. The inner bound varies per row, so
# the adjoint takes the ragged (Tier B) layout and the offsets end up beside
# a prefix-table lookup.
#
# out: output array of length i_nrow, filled in place
# u: input array of length >= i_base * i_nrow + i_nrow
# i_nrow: number of rows
# i_base: base row stride
function cse_intoffset(out, u, i_nrow, i_base)
    for i_r = 1:i_nrow
        i_len = i_base + i_r
        # the same offset expression, three times, none of them in a subscript
        i_off = (i_r - 1) * i_base + i_len
        i_lo = (i_r - 1) * i_base + i_len - i_len
        i_hi = (i_r - 1) * i_base + i_len + 1
        s = 0.0
        for i_j = 1:i_len
            # inside a subscript: must stay inline
            s = s + u[(i_r - 1) * i_base + i_j] * u[i_lo + i_j]
        end
        out[i_r] = s * u[i_off] + s * u[i_hi]
    end
    return nothing
end
