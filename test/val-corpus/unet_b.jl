function initstacks_unet_b(c1, c2, c3, h, w)
    two = 2
    h2 = div(h, two)
    w2 = div(w, two)
    hw2 = h2 * w2
    four = 4
    h4 = div(h, four)
    w4 = div(w, four)
    hw4 = h4 * w4
    n_b_mid = c3 * hw4
    n_b_out = c3 * hw4
    hw = h * w
    n_d1_mid = c1 * hw
    n_d1_out = c1 * hw
    n_d2_mid = c2 * hw2
    n_d2_out = c2 * hw2
    n_e1_mid = c1 * hw
    n_e1_out = c1 * hw
    n_e2_mid = c2 * hw2
    n_e2_out = c2 * hw2
    t_e1_stack = Vector{Float64}(undef, max(0, div(n_e1_mid - 1, 1) + 1))
    skip1_stack = Vector{Float64}(undef, max(0, div(n_e1_out - 1, 1) + 1))
    t_e2_stack = Vector{Float64}(undef, max(0, div(n_e2_mid - 1, 1) + 1))
    skip2_stack = Vector{Float64}(undef, max(0, div(n_e2_out - 1, 1) + 1))
    t_b_stack = Vector{Float64}(undef, max(0, div(n_b_mid - 1, 1) + 1))
    bott_stack = Vector{Float64}(undef, max(0, div(n_b_out - 1, 1) + 1))
    t_d2_stack = Vector{Float64}(undef, max(0, div(n_d2_mid - 1, 1) + 1))
    dec2out_stack = Vector{Float64}(undef, max(0, div(n_d2_out - 1, 1) + 1))
    t_d1_stack = Vector{Float64}(undef, max(0, div(n_d1_mid - 1, 1) + 1))
    dec1out_stack = Vector{Float64}(undef, max(0, div(n_d1_out - 1, 1) + 1))
    return (t_e1_stack, skip1_stack, t_e2_stack, skip2_stack, t_b_stack, bott_stack, t_d2_stack, dec2out_stack, t_d1_stack, dec1out_stack)
end

function unet_b(x, xb, h, w, c_in, c1, c2, c3, c_out, w_e1a, w_e1ab, b_e1a, b_e1ab, w_e1b, w_e1bb, b_e1b, b_e1bb, w_e2a, w_e2ab, b_e2a, b_e2ab, w_e2b, w_e2bb, b_e2b, b_e2bb, w_ba, w_bab, b_ba, b_bab, w_bb, w_bbb, b_bb, b_bbb, w_d2a, w_d2ab, b_d2a, b_d2ab, w_d2b, w_d2bb, b_d2b, b_d2bb, w_d1a, w_d1ab, b_d1a, b_d1ab, w_d1b, w_d1bb, b_d1b, b_d1bb, w_out, w_outb, b_out, b_outb, xpad0, xpad0b, t_e1, t_e1b, t_e1pad, t_e1padb, skip1, skip1b, p1, p1b, p1pad, p1padb, t_e2, t_e2b, t_e2pad, t_e2padb, skip2, skip2b, p2, p2b, p2pad, p2padb, t_b, t_bb, t_bpad, t_bpadb, bott, bottb, u2, u2b, cat2, cat2b, cat2pad, cat2padb, t_d2, t_d2b, t_d2pad, t_d2padb, dec2out, dec2outb, u1, u1b, cat1, cat1b, cat1pad, cat1padb, t_d1, t_d1b, t_d1pad, t_d1padb, dec1out, dec1outb, y, yb, t_e1_stack, skip1_stack, t_e2_stack, skip2_stack, t_b_stack, bott_stack, t_d2_stack, dec2out_stack, t_d1_stack, dec1out_stack)
    a11 = 0.0
    a12 = 0.0
    a21 = 0.0
    a22 = 0.0
    m1 = 0.0
    m2 = 0.0
    s = 0.0
    a11b = 0.0
    a12b = 0.0
    a21b = 0.0
    a22b = 0.0
    m1b = 0.0
    m2b = 0.0
    sb = 0.0
    two = 2
    four = 4
    h2 = div(h, two)
    w2 = div(w, two)
    h4 = div(h, four)
    w4 = div(w, four)
    hp1 = h + 2
    wp1 = w + 2
    hp2 = h2 + 2
    wp2 = w2 + 2
    hp4 = h4 + 2
    wp4 = w4 + 2
    pad = 1
    kh = 3
    kw = 3
    khkw = kh * kw
    kh_out = 1
    kw_out = 1
    khkw_out = kh_out * kw_out
    scale = 2
    zero_val = 0.0
    c32 = c3 + c2
    c21 = c2 + c1
    hw = h * w
    hw2 = h2 * w2
    hw4 = h4 * w4
    n_xpad0 = c_in * hp1 * wp1
    __icse_0 = c1 * hw
    n_e1_mid = __icse_0
    __icse_1 = c1 * hp1 * wp1
    n_e1_midpad = __icse_1
    n_e1_out = __icse_0
    n_p1pad = c1 * hp2 * wp2
    __icse_2 = c2 * hw2
    n_e2_mid = __icse_2
    __icse_3 = c2 * hp2 * wp2
    n_e2_midpad = __icse_3
    n_e2_out = __icse_2
    n_p2pad = c2 * hp4 * wp4
    __icse_4 = c3 * hw4
    n_b_mid = __icse_4
    n_b_midpad = c3 * hp4 * wp4
    n_b_out = __icse_4
    n_cat2pad = c32 * hp2 * wp2
    n_d2_mid = __icse_2
    n_d2_midpad = __icse_3
    n_d2_out = __icse_2
    n_cat1pad = c21 * hp1 * wp1
    n_d1_mid = __icse_0
    n_d1_midpad = __icse_1
    n_d1_out = __icse_0
    for i = 1:n_xpad0
        xpad0[i] = zero_val
    end
    for idx = 1:c_in * hw
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
        xpad0[yi] = x[idx]
    end
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        s = 0.0
        for i_k = 1:c_in * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_5 = ci - 1
            xi = __icse_5 * hp1 * wp1 + (row - 1) * wp1 + col
            wi = (((co - 1) * c_in + __icse_5) * kh + (ki - 1)) * kw + kj
            s = s + xpad0[xi] * w_e1a[wi]
        end
        t_e1[idx] = s + b_e1a[co]
    end
    for i = 1:n_e1_mid
        __idx_t_e1_stack_0 = (i - 1) + 1
        __cse_6 = t_e1[i]
        t_e1_stack[__idx_t_e1_stack_0] = __cse_6
        t_e1[i] = max(__cse_6, zero_val)
    end
    for i = 1:n_e1_midpad
        t_e1pad[i] = zero_val
    end
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
        t_e1pad[yi] = t_e1[idx]
    end
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        s = 0.0
        for i_k = 1:c1 * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_7 = ci - 1
            xi = __icse_7 * hp1 * wp1 + (row - 1) * wp1 + col
            wi = (((co - 1) * c1 + __icse_7) * kh + (ki - 1)) * kw + kj
            s = s + t_e1pad[xi] * w_e1b[wi]
        end
        skip1[idx] = s + b_e1b[co]
    end
    for i = 1:n_e1_out
        __idx_skip1_stack_0 = (i - 1) + 1
        __cse_8 = skip1[i]
        skip1_stack[__idx_skip1_stack_0] = __cse_8
        skip1[i] = max(__cse_8, zero_val)
    end
    for idx = 1:c1 * hw2
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        oi = div(rem, w2) + 1
        oj = mod(rem, w2) + 1
        i0 = 2oi - 1
        j0 = 2oj - 1
        a11 = skip1[(ci - 1) * hw + (i0 - 1) * w + j0]
        a12 = skip1[(ci - 1) * hw + (i0 - 1) * w + j0 + 1]
        a21 = skip1[(ci - 1) * hw + i0 * w + j0]
        a22 = skip1[(ci - 1) * hw + i0 * w + j0 + 1]
        m1 = max(a11, a12)
        m2 = max(a21, a22)
        p1[idx] = max(m1, m2)
    end
    for i = 1:n_p1pad
        p1pad[i] = zero_val
    end
    for idx = 1:c1 * hw2
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
        p1pad[yi] = p1[idx]
    end
    for idx = 1:c2 * hw2
        idxm1 = idx - 1
        co = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        s = 0.0
        for i_k = 1:c1 * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_9 = ci - 1
            xi = __icse_9 * hp2 * wp2 + (row - 1) * wp2 + col
            wi = (((co - 1) * c1 + __icse_9) * kh + (ki - 1)) * kw + kj
            s = s + p1pad[xi] * w_e2a[wi]
        end
        t_e2[idx] = s + b_e2a[co]
    end
    for i = 1:n_e2_mid
        __idx_t_e2_stack_0 = (i - 1) + 1
        __cse_10 = t_e2[i]
        t_e2_stack[__idx_t_e2_stack_0] = __cse_10
        t_e2[i] = max(__cse_10, zero_val)
    end
    for i = 1:n_e2_midpad
        t_e2pad[i] = zero_val
    end
    for idx = 1:c2 * hw2
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
        t_e2pad[yi] = t_e2[idx]
    end
    for idx = 1:c2 * hw2
        idxm1 = idx - 1
        co = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        s = 0.0
        for i_k = 1:c2 * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_11 = ci - 1
            xi = __icse_11 * hp2 * wp2 + (row - 1) * wp2 + col
            wi = (((co - 1) * c2 + __icse_11) * kh + (ki - 1)) * kw + kj
            s = s + t_e2pad[xi] * w_e2b[wi]
        end
        skip2[idx] = s + b_e2b[co]
    end
    for i = 1:n_e2_out
        __idx_skip2_stack_0 = (i - 1) + 1
        __cse_12 = skip2[i]
        skip2_stack[__idx_skip2_stack_0] = __cse_12
        skip2[i] = max(__cse_12, zero_val)
    end
    for idx = 1:c2 * hw4
        idxm1 = idx - 1
        ci = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        oi = div(rem, w4) + 1
        oj = mod(rem, w4) + 1
        i0 = 2oi - 1
        j0 = 2oj - 1
        a11 = skip2[(ci - 1) * hw2 + (i0 - 1) * w2 + j0]
        a12 = skip2[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1]
        a21 = skip2[(ci - 1) * hw2 + i0 * w2 + j0]
        a22 = skip2[(ci - 1) * hw2 + i0 * w2 + j0 + 1]
        m1 = max(a11, a12)
        m2 = max(a21, a22)
        p2[idx] = max(m1, m2)
    end
    for i = 1:n_p2pad
        p2pad[i] = zero_val
    end
    for idx = 1:c2 * hw4
        idxm1 = idx - 1
        ci = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        i = div(rem, w4) + 1
        j = mod(rem, w4) + 1
        yi = (ci - 1) * hp4 * wp4 + ((i + pad) - 1) * wp4 + (j + pad)
        p2pad[yi] = p2[idx]
    end
    for idx = 1:c3 * hw4
        idxm1 = idx - 1
        co = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        i = div(rem, w4) + 1
        j = mod(rem, w4) + 1
        s = 0.0
        for i_k = 1:c2 * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_13 = ci - 1
            xi = __icse_13 * hp4 * wp4 + (row - 1) * wp4 + col
            wi = (((co - 1) * c2 + __icse_13) * kh + (ki - 1)) * kw + kj
            s = s + p2pad[xi] * w_ba[wi]
        end
        t_b[idx] = s + b_ba[co]
    end
    for i = 1:n_b_mid
        __idx_t_b_stack_0 = (i - 1) + 1
        __cse_14 = t_b[i]
        t_b_stack[__idx_t_b_stack_0] = __cse_14
        t_b[i] = max(__cse_14, zero_val)
    end
    for i = 1:n_b_midpad
        t_bpad[i] = zero_val
    end
    for idx = 1:c3 * hw4
        idxm1 = idx - 1
        ci = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        i = div(rem, w4) + 1
        j = mod(rem, w4) + 1
        yi = (ci - 1) * hp4 * wp4 + ((i + pad) - 1) * wp4 + (j + pad)
        t_bpad[yi] = t_b[idx]
    end
    for idx = 1:c3 * hw4
        idxm1 = idx - 1
        co = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        i = div(rem, w4) + 1
        j = mod(rem, w4) + 1
        s = 0.0
        for i_k = 1:c3 * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_15 = ci - 1
            xi = __icse_15 * hp4 * wp4 + (row - 1) * wp4 + col
            wi = (((co - 1) * c3 + __icse_15) * kh + (ki - 1)) * kw + kj
            s = s + t_bpad[xi] * w_bb[wi]
        end
        bott[idx] = s + b_bb[co]
    end
    for i = 1:n_b_out
        __idx_bott_stack_0 = (i - 1) + 1
        __cse_16 = bott[i]
        bott_stack[__idx_bott_stack_0] = __cse_16
        bott[i] = max(__cse_16, zero_val)
    end
    for idx = 1:c3 * hw2
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        oi = div(rem, w2) + 1
        oj = mod(rem, w2) + 1
        oim1 = oi - 1
        ojm1 = oj - 1
        i = div(oim1, scale) + 1
        j = div(ojm1, scale) + 1
        xi = (ci - 1) * hw4 + (i - 1) * w4 + j
        u2[idx] = bott[xi]
    end
    for idx = 1:c3 * hw2
        cat2[idx] = u2[idx]
    end
    for idx = 1:c2 * hw2
        cat2[c3 * hw2 + idx] = skip2[idx]
    end
    for i = 1:n_cat2pad
        cat2pad[i] = zero_val
    end
    for idx = 1:c32 * hw2
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
        cat2pad[yi] = cat2[idx]
    end
    for idx = 1:c2 * hw2
        idxm1 = idx - 1
        co = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        s = 0.0
        for i_k = 1:c32 * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_17 = ci - 1
            xi = __icse_17 * hp2 * wp2 + (row - 1) * wp2 + col
            wi = (((co - 1) * c32 + __icse_17) * kh + (ki - 1)) * kw + kj
            s = s + cat2pad[xi] * w_d2a[wi]
        end
        t_d2[idx] = s + b_d2a[co]
    end
    for i = 1:n_d2_mid
        __idx_t_d2_stack_0 = (i - 1) + 1
        __cse_18 = t_d2[i]
        t_d2_stack[__idx_t_d2_stack_0] = __cse_18
        t_d2[i] = max(__cse_18, zero_val)
    end
    for i = 1:n_d2_midpad
        t_d2pad[i] = zero_val
    end
    for idx = 1:c2 * hw2
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
        t_d2pad[yi] = t_d2[idx]
    end
    for idx = 1:c2 * hw2
        idxm1 = idx - 1
        co = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        s = 0.0
        for i_k = 1:c2 * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_19 = ci - 1
            xi = __icse_19 * hp2 * wp2 + (row - 1) * wp2 + col
            wi = (((co - 1) * c2 + __icse_19) * kh + (ki - 1)) * kw + kj
            s = s + t_d2pad[xi] * w_d2b[wi]
        end
        dec2out[idx] = s + b_d2b[co]
    end
    for i = 1:n_d2_out
        __idx_dec2out_stack_0 = (i - 1) + 1
        __cse_20 = dec2out[i]
        dec2out_stack[__idx_dec2out_stack_0] = __cse_20
        dec2out[i] = max(__cse_20, zero_val)
    end
    for idx = 1:c2 * hw
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        oi = div(rem, w) + 1
        oj = mod(rem, w) + 1
        oim1 = oi - 1
        ojm1 = oj - 1
        i = div(oim1, scale) + 1
        j = div(ojm1, scale) + 1
        xi = (ci - 1) * hw2 + (i - 1) * w2 + j
        u1[idx] = dec2out[xi]
    end
    for idx = 1:c2 * hw
        cat1[idx] = u1[idx]
    end
    for idx = 1:c1 * hw
        cat1[c2 * hw + idx] = skip1[idx]
    end
    for i = 1:n_cat1pad
        cat1pad[i] = zero_val
    end
    for idx = 1:c21 * hw
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
        cat1pad[yi] = cat1[idx]
    end
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        s = 0.0
        for i_k = 1:c21 * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_21 = ci - 1
            xi = __icse_21 * hp1 * wp1 + (row - 1) * wp1 + col
            wi = (((co - 1) * c21 + __icse_21) * kh + (ki - 1)) * kw + kj
            s = s + cat1pad[xi] * w_d1a[wi]
        end
        t_d1[idx] = s + b_d1a[co]
    end
    for i = 1:n_d1_mid
        __idx_t_d1_stack_0 = (i - 1) + 1
        __cse_22 = t_d1[i]
        t_d1_stack[__idx_t_d1_stack_0] = __cse_22
        t_d1[i] = max(__cse_22, zero_val)
    end
    for i = 1:n_d1_midpad
        t_d1pad[i] = zero_val
    end
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
        t_d1pad[yi] = t_d1[idx]
    end
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        s = 0.0
        for i_k = 1:c1 * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_23 = ci - 1
            xi = __icse_23 * hp1 * wp1 + (row - 1) * wp1 + col
            wi = (((co - 1) * c1 + __icse_23) * kh + (ki - 1)) * kw + kj
            s = s + t_d1pad[xi] * w_d1b[wi]
        end
        dec1out[idx] = s + b_d1b[co]
    end
    for i = 1:n_d1_out
        __idx_dec1out_stack_0 = (i - 1) + 1
        __cse_24 = dec1out[i]
        dec1out_stack[__idx_dec1out_stack_0] = __cse_24
        dec1out[i] = max(__cse_24, zero_val)
    end
    for idx = 1:c_out * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        s = 0.0
        for i_k = 1:c1 * khkw_out
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw_out) + 1
            rem2 = mod(kseqm1, khkw_out)
            ki = div(rem2, kw_out) + 1
            kj = mod(rem2, kw_out) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_25 = ci - 1
            xi = __icse_25 * hw + (row - 1) * w + col
            wi = (((co - 1) * c1 + __icse_25) * kh_out + (ki - 1)) * kw_out + kj
            s = s + dec1out[xi] * w_out[wi]
        end
        y[idx] = s + b_out[co]
    end
    two = 2
    four = 4
    h2 = div(h, two)
    w2 = div(w, two)
    h4 = div(h, four)
    w4 = div(w, four)
    hp1 = h + 2
    wp1 = w + 2
    hp2 = h2 + 2
    wp2 = w2 + 2
    hp4 = h4 + 2
    wp4 = w4 + 2
    pad = 1
    kh = 3
    kw = 3
    khkw = kh * kw
    kh_out = 1
    kw_out = 1
    khkw_out = kh_out * kw_out
    scale = 2
    c32 = c3 + c2
    c21 = c2 + c1
    hw = h * w
    hw2 = h2 * w2
    hw4 = h4 * w4
    n_xpad0 = c_in * hp1 * wp1
    __icse_26 = c1 * hw
    n_e1_mid = __icse_26
    __icse_27 = c1 * hp1 * wp1
    n_e1_midpad = __icse_27
    n_e1_out = __icse_26
    n_p1pad = c1 * hp2 * wp2
    __icse_28 = c2 * hw2
    n_e2_mid = __icse_28
    __icse_29 = c2 * hp2 * wp2
    n_e2_midpad = __icse_29
    n_e2_out = __icse_28
    n_p2pad = c2 * hp4 * wp4
    __icse_30 = c3 * hw4
    n_b_mid = __icse_30
    n_b_midpad = c3 * hp4 * wp4
    n_b_out = __icse_30
    n_cat2pad = c32 * hp2 * wp2
    n_d2_mid = __icse_28
    n_d2_midpad = __icse_29
    n_d2_out = __icse_28
    n_cat1pad = c21 * hp1 * wp1
    n_d1_mid = __icse_26
    n_d1_midpad = __icse_27
    n_d1_out = __icse_26
    for idx = c_out * hw:-1:1
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        __oldb_0 = yb[idx]
        yb[idx] = 0.0
        sb = sb + __oldb_0
        b_outb[co] = b_outb[co] + __oldb_0
        for i_k = c1 * khkw_out:-1:1
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw_out) + 1
            rem2 = mod(kseqm1, khkw_out)
            ki = div(rem2, kw_out) + 1
            kj = mod(rem2, kw_out) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_31 = ci - 1
            xi = __icse_31 * hw + (row - 1) * w + col
            wi = (((co - 1) * c1 + __icse_31) * kh_out + (ki - 1)) * kw_out + kj
            dec1outb[xi] = dec1outb[xi] + w_out[wi] * sb
            w_outb[wi] = w_outb[wi] + dec1out[xi] * sb
        end
        sb = 0.0
    end
    for i = n_d1_out:-1:1
        __idx_dec1out_stack_0 = (i - 1) + 1
        dec1out[i] = dec1out_stack[__idx_dec1out_stack_0]
        __oldb_2 = dec1outb[i]
        dec1outb[i] = 0.0
        dec1outb[i] = dec1outb[i] + (0.5 * (1.0 + sign(dec1out[i] - zero_val))) * __oldb_2
    end
    for idx = c1 * hw:-1:1
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        __oldb_0 = dec1outb[idx]
        dec1outb[idx] = 0.0
        sb = sb + __oldb_0
        b_d1bb[co] = b_d1bb[co] + __oldb_0
        for i_k = c1 * khkw:-1:1
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_32 = ci - 1
            xi = __icse_32 * hp1 * wp1 + (row - 1) * wp1 + col
            wi = (((co - 1) * c1 + __icse_32) * kh + (ki - 1)) * kw + kj
            t_d1padb[xi] = t_d1padb[xi] + w_d1b[wi] * sb
            w_d1bb[wi] = w_d1bb[wi] + t_d1pad[xi] * sb
        end
        sb = 0.0
    end
    for idx = c1 * hw:-1:1
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
        __oldb_0 = t_d1padb[yi]
        t_d1padb[yi] = 0.0
        t_d1b[idx] = t_d1b[idx] + __oldb_0
    end
    for i = n_d1_midpad:-1:1
        t_d1padb[i] = 0.0
    end
    for i = n_d1_mid:-1:1
        __idx_t_d1_stack_0 = (i - 1) + 1
        t_d1[i] = t_d1_stack[__idx_t_d1_stack_0]
        __oldb_2 = t_d1b[i]
        t_d1b[i] = 0.0
        t_d1b[i] = t_d1b[i] + (0.5 * (1.0 + sign(t_d1[i] - zero_val))) * __oldb_2
    end
    for idx = c1 * hw:-1:1
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        __oldb_0 = t_d1b[idx]
        t_d1b[idx] = 0.0
        sb = sb + __oldb_0
        b_d1ab[co] = b_d1ab[co] + __oldb_0
        for i_k = c21 * khkw:-1:1
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_33 = ci - 1
            xi = __icse_33 * hp1 * wp1 + (row - 1) * wp1 + col
            wi = (((co - 1) * c21 + __icse_33) * kh + (ki - 1)) * kw + kj
            cat1padb[xi] = cat1padb[xi] + w_d1a[wi] * sb
            w_d1ab[wi] = w_d1ab[wi] + cat1pad[xi] * sb
        end
        sb = 0.0
    end
    for idx = c21 * hw:-1:1
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
        __oldb_0 = cat1padb[yi]
        cat1padb[yi] = 0.0
        cat1b[idx] = cat1b[idx] + __oldb_0
    end
    for i = n_cat1pad:-1:1
        cat1padb[i] = 0.0
    end
    for idx = c1 * hw:-1:1
        __oldb_0 = cat1b[c2 * hw + idx]
        cat1b[c2 * hw + idx] = 0.0
        skip1b[idx] = skip1b[idx] + __oldb_0
    end
    for idx = c2 * hw:-1:1
        __oldb_0 = cat1b[idx]
        cat1b[idx] = 0.0
        u1b[idx] = u1b[idx] + __oldb_0
    end
    for idx = c2 * hw:-1:1
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        oi = div(rem, w) + 1
        oj = mod(rem, w) + 1
        oim1 = oi - 1
        ojm1 = oj - 1
        i = div(oim1, scale) + 1
        j = div(ojm1, scale) + 1
        xi = (ci - 1) * hw2 + (i - 1) * w2 + j
        __oldb_0 = u1b[idx]
        u1b[idx] = 0.0
        dec2outb[xi] = dec2outb[xi] + __oldb_0
    end
    for i = n_d2_out:-1:1
        __idx_dec2out_stack_0 = (i - 1) + 1
        dec2out[i] = dec2out_stack[__idx_dec2out_stack_0]
        __oldb_2 = dec2outb[i]
        dec2outb[i] = 0.0
        dec2outb[i] = dec2outb[i] + (0.5 * (1.0 + sign(dec2out[i] - zero_val))) * __oldb_2
    end
    for idx = c2 * hw2:-1:1
        idxm1 = idx - 1
        co = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        __oldb_0 = dec2outb[idx]
        dec2outb[idx] = 0.0
        sb = sb + __oldb_0
        b_d2bb[co] = b_d2bb[co] + __oldb_0
        for i_k = c2 * khkw:-1:1
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_34 = ci - 1
            xi = __icse_34 * hp2 * wp2 + (row - 1) * wp2 + col
            wi = (((co - 1) * c2 + __icse_34) * kh + (ki - 1)) * kw + kj
            t_d2padb[xi] = t_d2padb[xi] + w_d2b[wi] * sb
            w_d2bb[wi] = w_d2bb[wi] + t_d2pad[xi] * sb
        end
        sb = 0.0
    end
    for idx = c2 * hw2:-1:1
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
        __oldb_0 = t_d2padb[yi]
        t_d2padb[yi] = 0.0
        t_d2b[idx] = t_d2b[idx] + __oldb_0
    end
    for i = n_d2_midpad:-1:1
        t_d2padb[i] = 0.0
    end
    for i = n_d2_mid:-1:1
        __idx_t_d2_stack_0 = (i - 1) + 1
        t_d2[i] = t_d2_stack[__idx_t_d2_stack_0]
        __oldb_2 = t_d2b[i]
        t_d2b[i] = 0.0
        t_d2b[i] = t_d2b[i] + (0.5 * (1.0 + sign(t_d2[i] - zero_val))) * __oldb_2
    end
    for idx = c2 * hw2:-1:1
        idxm1 = idx - 1
        co = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        __oldb_0 = t_d2b[idx]
        t_d2b[idx] = 0.0
        sb = sb + __oldb_0
        b_d2ab[co] = b_d2ab[co] + __oldb_0
        for i_k = c32 * khkw:-1:1
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_35 = ci - 1
            xi = __icse_35 * hp2 * wp2 + (row - 1) * wp2 + col
            wi = (((co - 1) * c32 + __icse_35) * kh + (ki - 1)) * kw + kj
            cat2padb[xi] = cat2padb[xi] + w_d2a[wi] * sb
            w_d2ab[wi] = w_d2ab[wi] + cat2pad[xi] * sb
        end
        sb = 0.0
    end
    for idx = c32 * hw2:-1:1
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
        __oldb_0 = cat2padb[yi]
        cat2padb[yi] = 0.0
        cat2b[idx] = cat2b[idx] + __oldb_0
    end
    for i = n_cat2pad:-1:1
        cat2padb[i] = 0.0
    end
    for idx = c2 * hw2:-1:1
        __oldb_0 = cat2b[c3 * hw2 + idx]
        cat2b[c3 * hw2 + idx] = 0.0
        skip2b[idx] = skip2b[idx] + __oldb_0
    end
    for idx = c3 * hw2:-1:1
        __oldb_0 = cat2b[idx]
        cat2b[idx] = 0.0
        u2b[idx] = u2b[idx] + __oldb_0
    end
    for idx = c3 * hw2:-1:1
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        oi = div(rem, w2) + 1
        oj = mod(rem, w2) + 1
        oim1 = oi - 1
        ojm1 = oj - 1
        i = div(oim1, scale) + 1
        j = div(ojm1, scale) + 1
        xi = (ci - 1) * hw4 + (i - 1) * w4 + j
        __oldb_0 = u2b[idx]
        u2b[idx] = 0.0
        bottb[xi] = bottb[xi] + __oldb_0
    end
    for i = n_b_out:-1:1
        __idx_bott_stack_0 = (i - 1) + 1
        bott[i] = bott_stack[__idx_bott_stack_0]
        __oldb_2 = bottb[i]
        bottb[i] = 0.0
        bottb[i] = bottb[i] + (0.5 * (1.0 + sign(bott[i] - zero_val))) * __oldb_2
    end
    for idx = c3 * hw4:-1:1
        idxm1 = idx - 1
        co = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        i = div(rem, w4) + 1
        j = mod(rem, w4) + 1
        __oldb_0 = bottb[idx]
        bottb[idx] = 0.0
        sb = sb + __oldb_0
        b_bbb[co] = b_bbb[co] + __oldb_0
        for i_k = c3 * khkw:-1:1
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_36 = ci - 1
            xi = __icse_36 * hp4 * wp4 + (row - 1) * wp4 + col
            wi = (((co - 1) * c3 + __icse_36) * kh + (ki - 1)) * kw + kj
            t_bpadb[xi] = t_bpadb[xi] + w_bb[wi] * sb
            w_bbb[wi] = w_bbb[wi] + t_bpad[xi] * sb
        end
        sb = 0.0
    end
    for idx = c3 * hw4:-1:1
        idxm1 = idx - 1
        ci = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        i = div(rem, w4) + 1
        j = mod(rem, w4) + 1
        yi = (ci - 1) * hp4 * wp4 + ((i + pad) - 1) * wp4 + (j + pad)
        __oldb_0 = t_bpadb[yi]
        t_bpadb[yi] = 0.0
        t_bb[idx] = t_bb[idx] + __oldb_0
    end
    for i = n_b_midpad:-1:1
        t_bpadb[i] = 0.0
    end
    for i = n_b_mid:-1:1
        __idx_t_b_stack_0 = (i - 1) + 1
        t_b[i] = t_b_stack[__idx_t_b_stack_0]
        __oldb_2 = t_bb[i]
        t_bb[i] = 0.0
        t_bb[i] = t_bb[i] + (0.5 * (1.0 + sign(t_b[i] - zero_val))) * __oldb_2
    end
    for idx = c3 * hw4:-1:1
        idxm1 = idx - 1
        co = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        i = div(rem, w4) + 1
        j = mod(rem, w4) + 1
        __oldb_0 = t_bb[idx]
        t_bb[idx] = 0.0
        sb = sb + __oldb_0
        b_bab[co] = b_bab[co] + __oldb_0
        for i_k = c2 * khkw:-1:1
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_37 = ci - 1
            xi = __icse_37 * hp4 * wp4 + (row - 1) * wp4 + col
            wi = (((co - 1) * c2 + __icse_37) * kh + (ki - 1)) * kw + kj
            p2padb[xi] = p2padb[xi] + w_ba[wi] * sb
            w_bab[wi] = w_bab[wi] + p2pad[xi] * sb
        end
        sb = 0.0
    end
    for idx = c2 * hw4:-1:1
        idxm1 = idx - 1
        ci = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        i = div(rem, w4) + 1
        j = mod(rem, w4) + 1
        yi = (ci - 1) * hp4 * wp4 + ((i + pad) - 1) * wp4 + (j + pad)
        __oldb_0 = p2padb[yi]
        p2padb[yi] = 0.0
        p2b[idx] = p2b[idx] + __oldb_0
    end
    for i = n_p2pad:-1:1
        p2padb[i] = 0.0
    end
    for idx = 1:c2 * hw4
        __icse_38 = idx - 1
        idxm1 = __icse_38
        ci = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        oi = div(rem, w4) + 1
        oj = mod(rem, w4) + 1
        i0 = 2oi - 1
        j0 = 2oj - 1
        a11 = skip2[(ci - 1) * hw2 + (i0 - 1) * w2 + j0]
        a12 = skip2[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1]
        a21 = skip2[(ci - 1) * hw2 + i0 * w2 + j0]
        a22 = skip2[(ci - 1) * hw2 + i0 * w2 + j0 + 1]
        m1 = max(a11, a12)
        m2 = max(a21, a22)
        idxm1 = __icse_38
        ci = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        oi = div(rem, w4) + 1
        oj = mod(rem, w4) + 1
        i0 = 2oi - 1
        j0 = 2oj - 1
        __oldb_0 = p2b[idx]
        p2b[idx] = 0.0
        m1b = m1b + (0.5 * (1.0 + sign(m1 - m2))) * __oldb_0
        m2b = m2b + (0.5 * (1.0 + sign(m2 - m1))) * __oldb_0
        __oldb_0 = m2b
        m2b = 0.0
        a21b = a21b + (0.5 * (1.0 + sign(a21 - a22))) * __oldb_0
        a22b = a22b + (0.5 * (1.0 + sign(a22 - a21))) * __oldb_0
        __oldb_0 = m1b
        m1b = 0.0
        a11b = a11b + (0.5 * (1.0 + sign(a11 - a12))) * __oldb_0
        a12b = a12b + (0.5 * (1.0 + sign(a12 - a11))) * __oldb_0
        __oldb_0 = a22b
        a22b = 0.0
        skip2b[(ci - 1) * hw2 + i0 * w2 + j0 + 1] = skip2b[(ci - 1) * hw2 + i0 * w2 + j0 + 1] + __oldb_0
        __oldb_0 = a21b
        a21b = 0.0
        skip2b[(ci - 1) * hw2 + i0 * w2 + j0] = skip2b[(ci - 1) * hw2 + i0 * w2 + j0] + __oldb_0
        __oldb_0 = a12b
        a12b = 0.0
        skip2b[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1] = skip2b[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1] + __oldb_0
        __oldb_0 = a11b
        a11b = 0.0
        skip2b[(ci - 1) * hw2 + (i0 - 1) * w2 + j0] = skip2b[(ci - 1) * hw2 + (i0 - 1) * w2 + j0] + __oldb_0
    end
    for i = n_e2_out:-1:1
        __idx_skip2_stack_0 = (i - 1) + 1
        skip2[i] = skip2_stack[__idx_skip2_stack_0]
        __oldb_2 = skip2b[i]
        skip2b[i] = 0.0
        skip2b[i] = skip2b[i] + (0.5 * (1.0 + sign(skip2[i] - zero_val))) * __oldb_2
    end
    for idx = c2 * hw2:-1:1
        idxm1 = idx - 1
        co = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        __oldb_0 = skip2b[idx]
        skip2b[idx] = 0.0
        sb = sb + __oldb_0
        b_e2bb[co] = b_e2bb[co] + __oldb_0
        for i_k = c2 * khkw:-1:1
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_39 = ci - 1
            xi = __icse_39 * hp2 * wp2 + (row - 1) * wp2 + col
            wi = (((co - 1) * c2 + __icse_39) * kh + (ki - 1)) * kw + kj
            t_e2padb[xi] = t_e2padb[xi] + w_e2b[wi] * sb
            w_e2bb[wi] = w_e2bb[wi] + t_e2pad[xi] * sb
        end
        sb = 0.0
    end
    for idx = c2 * hw2:-1:1
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
        __oldb_0 = t_e2padb[yi]
        t_e2padb[yi] = 0.0
        t_e2b[idx] = t_e2b[idx] + __oldb_0
    end
    for i = n_e2_midpad:-1:1
        t_e2padb[i] = 0.0
    end
    for i = n_e2_mid:-1:1
        __idx_t_e2_stack_0 = (i - 1) + 1
        t_e2[i] = t_e2_stack[__idx_t_e2_stack_0]
        __oldb_2 = t_e2b[i]
        t_e2b[i] = 0.0
        t_e2b[i] = t_e2b[i] + (0.5 * (1.0 + sign(t_e2[i] - zero_val))) * __oldb_2
    end
    for idx = c2 * hw2:-1:1
        idxm1 = idx - 1
        co = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        __oldb_0 = t_e2b[idx]
        t_e2b[idx] = 0.0
        sb = sb + __oldb_0
        b_e2ab[co] = b_e2ab[co] + __oldb_0
        for i_k = c1 * khkw:-1:1
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_40 = ci - 1
            xi = __icse_40 * hp2 * wp2 + (row - 1) * wp2 + col
            wi = (((co - 1) * c1 + __icse_40) * kh + (ki - 1)) * kw + kj
            p1padb[xi] = p1padb[xi] + w_e2a[wi] * sb
            w_e2ab[wi] = w_e2ab[wi] + p1pad[xi] * sb
        end
        sb = 0.0
    end
    for idx = c1 * hw2:-1:1
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
        __oldb_0 = p1padb[yi]
        p1padb[yi] = 0.0
        p1b[idx] = p1b[idx] + __oldb_0
    end
    for i = n_p1pad:-1:1
        p1padb[i] = 0.0
    end
    for idx = 1:c1 * hw2
        __icse_41 = idx - 1
        idxm1 = __icse_41
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        oi = div(rem, w2) + 1
        oj = mod(rem, w2) + 1
        i0 = 2oi - 1
        j0 = 2oj - 1
        a11 = skip1[(ci - 1) * hw + (i0 - 1) * w + j0]
        a12 = skip1[(ci - 1) * hw + (i0 - 1) * w + j0 + 1]
        a21 = skip1[(ci - 1) * hw + i0 * w + j0]
        a22 = skip1[(ci - 1) * hw + i0 * w + j0 + 1]
        m1 = max(a11, a12)
        m2 = max(a21, a22)
        idxm1 = __icse_41
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        oi = div(rem, w2) + 1
        oj = mod(rem, w2) + 1
        i0 = 2oi - 1
        j0 = 2oj - 1
        __oldb_0 = p1b[idx]
        p1b[idx] = 0.0
        m1b = m1b + (0.5 * (1.0 + sign(m1 - m2))) * __oldb_0
        m2b = m2b + (0.5 * (1.0 + sign(m2 - m1))) * __oldb_0
        __oldb_0 = m2b
        m2b = 0.0
        a21b = a21b + (0.5 * (1.0 + sign(a21 - a22))) * __oldb_0
        a22b = a22b + (0.5 * (1.0 + sign(a22 - a21))) * __oldb_0
        __oldb_0 = m1b
        m1b = 0.0
        a11b = a11b + (0.5 * (1.0 + sign(a11 - a12))) * __oldb_0
        a12b = a12b + (0.5 * (1.0 + sign(a12 - a11))) * __oldb_0
        __oldb_0 = a22b
        a22b = 0.0
        skip1b[(ci - 1) * hw + i0 * w + j0 + 1] = skip1b[(ci - 1) * hw + i0 * w + j0 + 1] + __oldb_0
        __oldb_0 = a21b
        a21b = 0.0
        skip1b[(ci - 1) * hw + i0 * w + j0] = skip1b[(ci - 1) * hw + i0 * w + j0] + __oldb_0
        __oldb_0 = a12b
        a12b = 0.0
        skip1b[(ci - 1) * hw + (i0 - 1) * w + j0 + 1] = skip1b[(ci - 1) * hw + (i0 - 1) * w + j0 + 1] + __oldb_0
        __oldb_0 = a11b
        a11b = 0.0
        skip1b[(ci - 1) * hw + (i0 - 1) * w + j0] = skip1b[(ci - 1) * hw + (i0 - 1) * w + j0] + __oldb_0
    end
    for i = n_e1_out:-1:1
        __idx_skip1_stack_0 = (i - 1) + 1
        skip1[i] = skip1_stack[__idx_skip1_stack_0]
        __oldb_2 = skip1b[i]
        skip1b[i] = 0.0
        skip1b[i] = skip1b[i] + (0.5 * (1.0 + sign(skip1[i] - zero_val))) * __oldb_2
    end
    for idx = c1 * hw:-1:1
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        __oldb_0 = skip1b[idx]
        skip1b[idx] = 0.0
        sb = sb + __oldb_0
        b_e1bb[co] = b_e1bb[co] + __oldb_0
        for i_k = c1 * khkw:-1:1
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_42 = ci - 1
            xi = __icse_42 * hp1 * wp1 + (row - 1) * wp1 + col
            wi = (((co - 1) * c1 + __icse_42) * kh + (ki - 1)) * kw + kj
            t_e1padb[xi] = t_e1padb[xi] + w_e1b[wi] * sb
            w_e1bb[wi] = w_e1bb[wi] + t_e1pad[xi] * sb
        end
        sb = 0.0
    end
    for idx = c1 * hw:-1:1
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
        __oldb_0 = t_e1padb[yi]
        t_e1padb[yi] = 0.0
        t_e1b[idx] = t_e1b[idx] + __oldb_0
    end
    for i = n_e1_midpad:-1:1
        t_e1padb[i] = 0.0
    end
    for i = n_e1_mid:-1:1
        __idx_t_e1_stack_0 = (i - 1) + 1
        t_e1[i] = t_e1_stack[__idx_t_e1_stack_0]
        __oldb_2 = t_e1b[i]
        t_e1b[i] = 0.0
        t_e1b[i] = t_e1b[i] + (0.5 * (1.0 + sign(t_e1[i] - zero_val))) * __oldb_2
    end
    for idx = c1 * hw:-1:1
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        __oldb_0 = t_e1b[idx]
        t_e1b[idx] = 0.0
        sb = sb + __oldb_0
        b_e1ab[co] = b_e1ab[co] + __oldb_0
        for i_k = c_in * khkw:-1:1
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            __icse_43 = ci - 1
            xi = __icse_43 * hp1 * wp1 + (row - 1) * wp1 + col
            wi = (((co - 1) * c_in + __icse_43) * kh + (ki - 1)) * kw + kj
            xpad0b[xi] = xpad0b[xi] + w_e1a[wi] * sb
            w_e1ab[wi] = w_e1ab[wi] + xpad0[xi] * sb
        end
        sb = 0.0
    end
    for idx = c_in * hw:-1:1
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
        __oldb_0 = xpad0b[yi]
        xpad0b[yi] = 0.0
        xb[idx] = xb[idx] + __oldb_0
    end
    for i = n_xpad0:-1:1
        xpad0b[i] = 0.0
    end
    zero_valb = 0.0
    return nothing
end

function unet(x, h, w, c_in, c1, c2, c3, c_out, w_e1a, b_e1a, w_e1b, b_e1b, w_e2a, b_e2a, w_e2b, b_e2b, w_ba, b_ba, w_bb, b_bb, w_d2a, b_d2a, w_d2b, b_d2b, w_d1a, b_d1a, w_d1b, b_d1b, w_out, b_out, xpad0, t_e1, t_e1pad, skip1, p1, p1pad, t_e2, t_e2pad, skip2, p2, p2pad, t_b, t_bpad, bott, u2, cat2, cat2pad, t_d2, t_d2pad, dec2out, u1, cat1, cat1pad, t_d1, t_d1pad, dec1out, y)
    two = 2
    four = 4
    h2 = div(h, two)
    w2 = div(w, two)
    h4 = div(h, four)
    w4 = div(w, four)
    hp1 = h + 2
    wp1 = w + 2
    hp2 = h2 + 2
    wp2 = w2 + 2
    hp4 = h4 + 2
    wp4 = w4 + 2
    pad = 1
    kh = 3
    kw = 3
    khkw = kh * kw
    kh_out = 1
    kw_out = 1
    khkw_out = kh_out * kw_out
    scale = 2
    zero_val = 0.0
    c32 = c3 + c2
    c21 = c2 + c1
    hw = h * w
    hw2 = h2 * w2
    hw4 = h4 * w4
    n_xpad0 = c_in * hp1 * wp1
    n_e1_mid = c1 * hw
    n_e1_midpad = c1 * hp1 * wp1
    n_e1_out = c1 * hw
    n_p1pad = c1 * hp2 * wp2
    n_e2_mid = c2 * hw2
    n_e2_midpad = c2 * hp2 * wp2
    n_e2_out = c2 * hw2
    n_p2pad = c2 * hp4 * wp4
    n_b_mid = c3 * hw4
    n_b_midpad = c3 * hp4 * wp4
    n_b_out = c3 * hw4
    n_cat2pad = c32 * hp2 * wp2
    n_d2_mid = c2 * hw2
    n_d2_midpad = c2 * hp2 * wp2
    n_d2_out = c2 * hw2
    n_cat1pad = c21 * hp1 * wp1
    n_d1_mid = c1 * hw
    n_d1_midpad = c1 * hp1 * wp1
    n_d1_out = c1 * hw
    for i = 1:n_xpad0
        xpad0[i] = zero_val
    end
    for idx = 1:c_in * hw
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
        xpad0[yi] = x[idx]
    end
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        s = 0.0
        for i_k = 1:c_in * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp1 * wp1 + (row - 1) * wp1 + col
            wi = (((co - 1) * c_in + (ci - 1)) * kh + (ki - 1)) * kw + kj
            s = s + xpad0[xi] * w_e1a[wi]
        end
        t_e1[idx] = s + b_e1a[co]
    end
    for i = 1:n_e1_mid
        t_e1[i] = max(t_e1[i], zero_val)
    end
    for i = 1:n_e1_midpad
        t_e1pad[i] = zero_val
    end
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
        t_e1pad[yi] = t_e1[idx]
    end
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        s = 0.0
        for i_k = 1:c1 * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp1 * wp1 + (row - 1) * wp1 + col
            wi = (((co - 1) * c1 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            s = s + t_e1pad[xi] * w_e1b[wi]
        end
        skip1[idx] = s + b_e1b[co]
    end
    for i = 1:n_e1_out
        skip1[i] = max(skip1[i], zero_val)
    end
    for idx = 1:c1 * hw2
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        oi = div(rem, w2) + 1
        oj = mod(rem, w2) + 1
        i0 = 2oi - 1
        j0 = 2oj - 1
        a11 = skip1[(ci - 1) * hw + (i0 - 1) * w + j0]
        a12 = skip1[(ci - 1) * hw + (i0 - 1) * w + j0 + 1]
        a21 = skip1[(ci - 1) * hw + i0 * w + j0]
        a22 = skip1[(ci - 1) * hw + i0 * w + j0 + 1]
        m1 = max(a11, a12)
        m2 = max(a21, a22)
        p1[idx] = max(m1, m2)
    end
    for i = 1:n_p1pad
        p1pad[i] = zero_val
    end
    for idx = 1:c1 * hw2
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
        p1pad[yi] = p1[idx]
    end
    for idx = 1:c2 * hw2
        idxm1 = idx - 1
        co = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        s = 0.0
        for i_k = 1:c1 * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp2 * wp2 + (row - 1) * wp2 + col
            wi = (((co - 1) * c1 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            s = s + p1pad[xi] * w_e2a[wi]
        end
        t_e2[idx] = s + b_e2a[co]
    end
    for i = 1:n_e2_mid
        t_e2[i] = max(t_e2[i], zero_val)
    end
    for i = 1:n_e2_midpad
        t_e2pad[i] = zero_val
    end
    for idx = 1:c2 * hw2
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
        t_e2pad[yi] = t_e2[idx]
    end
    for idx = 1:c2 * hw2
        idxm1 = idx - 1
        co = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        s = 0.0
        for i_k = 1:c2 * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp2 * wp2 + (row - 1) * wp2 + col
            wi = (((co - 1) * c2 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            s = s + t_e2pad[xi] * w_e2b[wi]
        end
        skip2[idx] = s + b_e2b[co]
    end
    for i = 1:n_e2_out
        skip2[i] = max(skip2[i], zero_val)
    end
    for idx = 1:c2 * hw4
        idxm1 = idx - 1
        ci = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        oi = div(rem, w4) + 1
        oj = mod(rem, w4) + 1
        i0 = 2oi - 1
        j0 = 2oj - 1
        a11 = skip2[(ci - 1) * hw2 + (i0 - 1) * w2 + j0]
        a12 = skip2[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1]
        a21 = skip2[(ci - 1) * hw2 + i0 * w2 + j0]
        a22 = skip2[(ci - 1) * hw2 + i0 * w2 + j0 + 1]
        m1 = max(a11, a12)
        m2 = max(a21, a22)
        p2[idx] = max(m1, m2)
    end
    for i = 1:n_p2pad
        p2pad[i] = zero_val
    end
    for idx = 1:c2 * hw4
        idxm1 = idx - 1
        ci = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        i = div(rem, w4) + 1
        j = mod(rem, w4) + 1
        yi = (ci - 1) * hp4 * wp4 + ((i + pad) - 1) * wp4 + (j + pad)
        p2pad[yi] = p2[idx]
    end
    for idx = 1:c3 * hw4
        idxm1 = idx - 1
        co = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        i = div(rem, w4) + 1
        j = mod(rem, w4) + 1
        s = 0.0
        for i_k = 1:c2 * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp4 * wp4 + (row - 1) * wp4 + col
            wi = (((co - 1) * c2 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            s = s + p2pad[xi] * w_ba[wi]
        end
        t_b[idx] = s + b_ba[co]
    end
    for i = 1:n_b_mid
        t_b[i] = max(t_b[i], zero_val)
    end
    for i = 1:n_b_midpad
        t_bpad[i] = zero_val
    end
    for idx = 1:c3 * hw4
        idxm1 = idx - 1
        ci = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        i = div(rem, w4) + 1
        j = mod(rem, w4) + 1
        yi = (ci - 1) * hp4 * wp4 + ((i + pad) - 1) * wp4 + (j + pad)
        t_bpad[yi] = t_b[idx]
    end
    for idx = 1:c3 * hw4
        idxm1 = idx - 1
        co = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        i = div(rem, w4) + 1
        j = mod(rem, w4) + 1
        s = 0.0
        for i_k = 1:c3 * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp4 * wp4 + (row - 1) * wp4 + col
            wi = (((co - 1) * c3 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            s = s + t_bpad[xi] * w_bb[wi]
        end
        bott[idx] = s + b_bb[co]
    end
    for i = 1:n_b_out
        bott[i] = max(bott[i], zero_val)
    end
    for idx = 1:c3 * hw2
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        oi = div(rem, w2) + 1
        oj = mod(rem, w2) + 1
        oim1 = oi - 1
        ojm1 = oj - 1
        i = div(oim1, scale) + 1
        j = div(ojm1, scale) + 1
        xi = (ci - 1) * hw4 + (i - 1) * w4 + j
        u2[idx] = bott[xi]
    end
    for idx = 1:c3 * hw2
        cat2[idx] = u2[idx]
    end
    for idx = 1:c2 * hw2
        cat2[c3 * hw2 + idx] = skip2[idx]
    end
    for i = 1:n_cat2pad
        cat2pad[i] = zero_val
    end
    for idx = 1:c32 * hw2
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
        cat2pad[yi] = cat2[idx]
    end
    for idx = 1:c2 * hw2
        idxm1 = idx - 1
        co = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        s = 0.0
        for i_k = 1:c32 * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp2 * wp2 + (row - 1) * wp2 + col
            wi = (((co - 1) * c32 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            s = s + cat2pad[xi] * w_d2a[wi]
        end
        t_d2[idx] = s + b_d2a[co]
    end
    for i = 1:n_d2_mid
        t_d2[i] = max(t_d2[i], zero_val)
    end
    for i = 1:n_d2_midpad
        t_d2pad[i] = zero_val
    end
    for idx = 1:c2 * hw2
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
        t_d2pad[yi] = t_d2[idx]
    end
    for idx = 1:c2 * hw2
        idxm1 = idx - 1
        co = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        s = 0.0
        for i_k = 1:c2 * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp2 * wp2 + (row - 1) * wp2 + col
            wi = (((co - 1) * c2 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            s = s + t_d2pad[xi] * w_d2b[wi]
        end
        dec2out[idx] = s + b_d2b[co]
    end
    for i = 1:n_d2_out
        dec2out[i] = max(dec2out[i], zero_val)
    end
    for idx = 1:c2 * hw
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        oi = div(rem, w) + 1
        oj = mod(rem, w) + 1
        oim1 = oi - 1
        ojm1 = oj - 1
        i = div(oim1, scale) + 1
        j = div(ojm1, scale) + 1
        xi = (ci - 1) * hw2 + (i - 1) * w2 + j
        u1[idx] = dec2out[xi]
    end
    for idx = 1:c2 * hw
        cat1[idx] = u1[idx]
    end
    for idx = 1:c1 * hw
        cat1[c2 * hw + idx] = skip1[idx]
    end
    for i = 1:n_cat1pad
        cat1pad[i] = zero_val
    end
    for idx = 1:c21 * hw
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
        cat1pad[yi] = cat1[idx]
    end
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        s = 0.0
        for i_k = 1:c21 * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp1 * wp1 + (row - 1) * wp1 + col
            wi = (((co - 1) * c21 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            s = s + cat1pad[xi] * w_d1a[wi]
        end
        t_d1[idx] = s + b_d1a[co]
    end
    for i = 1:n_d1_mid
        t_d1[i] = max(t_d1[i], zero_val)
    end
    for i = 1:n_d1_midpad
        t_d1pad[i] = zero_val
    end
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
        t_d1pad[yi] = t_d1[idx]
    end
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        s = 0.0
        for i_k = 1:c1 * khkw
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp1 * wp1 + (row - 1) * wp1 + col
            wi = (((co - 1) * c1 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            s = s + t_d1pad[xi] * w_d1b[wi]
        end
        dec1out[idx] = s + b_d1b[co]
    end
    for i = 1:n_d1_out
        dec1out[i] = max(dec1out[i], zero_val)
    end
    for idx = 1:c_out * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        s = 0.0
        for i_k = 1:c1 * khkw_out
            kseqm1 = i_k - 1
            ci = div(kseqm1, khkw_out) + 1
            rem2 = mod(kseqm1, khkw_out)
            ki = div(rem2, kw_out) + 1
            kj = mod(rem2, kw_out) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hw + (row - 1) * w + col
            wi = (((co - 1) * c1 + (ci - 1)) * kh_out + (ki - 1)) * kw_out + kj
            s = s + dec1out[xi] * w_out[wi]
        end
        y[idx] = s + b_out[co]
    end
end
