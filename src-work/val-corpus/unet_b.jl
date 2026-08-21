function initstacks_unet_b(c1, c2)
    t_e1_stack = Vector{Float64}(undef, div(n_e1_mid - 1, 1) + 1)
    skip1_stack = Vector{Float64}(undef, div(n_e1_out - 1, 1) + 1)
    a11_stack = Vector{Float64}(undef, ((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1)
    a12_stack = Vector{Float64}(undef, ((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1)
    a21_stack = Vector{Float64}(undef, ((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1)
    a22_stack = Vector{Float64}(undef, ((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1)
    m1_stack = Vector{Float64}(undef, ((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1)
    m2_stack = Vector{Float64}(undef, ((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1)
    t_e2_stack = Vector{Float64}(undef, div(n_e2_mid - 1, 1) + 1)
    skip2_stack = Vector{Float64}(undef, div(n_e2_out - 1, 1) + 1)
    t_b_stack = Vector{Float64}(undef, div(n_b_mid - 1, 1) + 1)
    bott_stack = Vector{Float64}(undef, div(n_b_out - 1, 1) + 1)
    t_d2_stack = Vector{Float64}(undef, div(n_d2_mid - 1, 1) + 1)
    dec2out_stack = Vector{Float64}(undef, div(n_d2_out - 1, 1) + 1)
    t_d1_stack = Vector{Float64}(undef, div(n_d1_mid - 1, 1) + 1)
    dec1out_stack = Vector{Float64}(undef, div(n_d1_out - 1, 1) + 1)
    return (t_e1_stack, skip1_stack, a11_stack, a12_stack, a21_stack, a22_stack, m1_stack, m2_stack, t_e2_stack, skip2_stack, t_b_stack, bott_stack, t_d2_stack, dec2out_stack, t_d1_stack, dec1out_stack)
end

function unet_b(x, xb, h, w, c_in, c1, c2, c3, c_out, w_e1a, w_e1ab, b_e1a, b_e1ab, w_e1b, w_e1bb, b_e1b, b_e1bb, w_e2a, w_e2ab, b_e2a, b_e2ab, w_e2b, w_e2bb, b_e2b, b_e2bb, w_ba, w_bab, b_ba, b_bab, w_bb, w_bbb, b_bb, b_bbb, w_d2a, w_d2ab, b_d2a, b_d2ab, w_d2b, w_d2bb, b_d2b, b_d2bb, w_d1a, w_d1ab, b_d1a, b_d1ab, w_d1b, w_d1bb, b_d1b, b_d1bb, w_out, w_outb, b_out, b_outb, xpad0, xpad0b, t_e1, t_e1b, t_e1pad, t_e1padb, skip1, skip1b, p1, p1b, p1pad, p1padb, t_e2, t_e2b, t_e2pad, t_e2padb, skip2, skip2b, p2, p2b, p2pad, p2padb, t_b, t_bb, t_bpad, t_bpadb, bott, bottb, u2, u2b, cat2, cat2b, cat2pad, cat2padb, t_d2, t_d2b, t_d2pad, t_d2padb, dec2out, dec2outb, u1, u1b, cat1, cat1b, cat1pad, cat1padb, t_d1, t_d1b, t_d1pad, t_d1padb, dec1out, dec1outb, y, yb, t_e1_stack, skip1_stack, a11_stack, a12_stack, a21_stack, a22_stack, m1_stack, m2_stack, t_e2_stack, skip2_stack, t_b_stack, bott_stack, t_d2_stack, dec2out_stack, t_d1_stack, dec1out_stack)
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
        for i_seq_k = 1:c_in * khkw
            kseqm1 = i_seq_k - 1
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
        t_e1_stack[(i - 1) + 1] = t_e1[i]
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
        for i_seq_k = 1:c1 * khkw
            kseqm1 = i_seq_k - 1
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
        skip1_stack[(i - 1) + 1] = skip1[i]
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
        for i_seq_k = 1:c1 * khkw
            kseqm1 = i_seq_k - 1
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
        t_e2_stack[(i - 1) + 1] = t_e2[i]
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
        for i_seq_k = 1:c2 * khkw
            kseqm1 = i_seq_k - 1
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
        skip2_stack[(i - 1) + 1] = skip2[i]
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
        for i_seq_k = 1:c2 * khkw
            kseqm1 = i_seq_k - 1
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
        t_b_stack[(i - 1) + 1] = t_b[i]
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
        for i_seq_k = 1:c3 * khkw
            kseqm1 = i_seq_k - 1
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
        bott_stack[(i - 1) + 1] = bott[i]
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
        for i_seq_k = 1:c32 * khkw
            kseqm1 = i_seq_k - 1
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
        t_d2_stack[(i - 1) + 1] = t_d2[i]
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
        for i_seq_k = 1:c2 * khkw
            kseqm1 = i_seq_k - 1
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
        dec2out_stack[(i - 1) + 1] = dec2out[i]
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
        for i_seq_k = 1:c21 * khkw
            kseqm1 = i_seq_k - 1
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
        t_d1_stack[(i - 1) + 1] = t_d1[i]
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
        for i_seq_k = 1:c1 * khkw
            kseqm1 = i_seq_k - 1
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
        dec1out_stack[(i - 1) + 1] = dec1out[i]
        dec1out[i] = max(dec1out[i], zero_val)
    end
    for idx = 1:c_out * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        s = 0.0
        for i_seq_k = 1:c1 * khkw_out
            kseqm1 = i_seq_k - 1
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
    for idx = 1:c_out * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        sb = sb + yb[idx]
        b_outb[co] = b_outb[co] + yb[idx]
        yb[idx] = 0.0
        for i_seq_k = c1 * khkw_out:-1:1
            kseqm1 = i_seq_k - 1
            ci = div(kseqm1, khkw_out) + 1
            rem2 = mod(kseqm1, khkw_out)
            ki = div(rem2, kw_out) + 1
            kj = mod(rem2, kw_out) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hw + (row - 1) * w + col
            wi = (((co - 1) * c1 + (ci - 1)) * kh_out + (ki - 1)) * kw_out + kj
            dec1outb[xi] = dec1outb[xi] + w_out[wi] * sb
            w_outb[wi] = w_outb[wi] + dec1out[xi] * sb
        end
        sb = 0.0
    end
    for i = n_d1_out:-1:1
        dec1out[i] = dec1out_stack[(i - 1) + 1]
        dec1outb[i] = (0.5 * (1.0 + sign(dec1out[i] - zero_val))) * dec1outb[i]
    end
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        sb = sb + dec1outb[idx]
        b_d1bb[co] = b_d1bb[co] + dec1outb[idx]
        dec1outb[idx] = 0.0
        for i_seq_k = c1 * khkw:-1:1
            kseqm1 = i_seq_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp1 * wp1 + (row - 1) * wp1 + col
            wi = (((co - 1) * c1 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            t_d1padb[xi] = t_d1padb[xi] + w_d1b[wi] * sb
            w_d1bb[wi] = w_d1bb[wi] + t_d1pad[xi] * sb
        end
        sb = 0.0
    end
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
        t_d1b[idx] = t_d1b[idx] + t_d1padb[yi]
        t_d1padb[yi] = 0.0
    end
    for i = 1:n_d1_midpad
        t_d1padb[i] = 0.0
    end
    for i = n_d1_mid:-1:1
        t_d1[i] = t_d1_stack[(i - 1) + 1]
        t_d1b[i] = (0.5 * (1.0 + sign(t_d1[i] - zero_val))) * t_d1b[i]
    end
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        sb = sb + t_d1b[idx]
        b_d1ab[co] = b_d1ab[co] + t_d1b[idx]
        t_d1b[idx] = 0.0
        for i_seq_k = c21 * khkw:-1:1
            kseqm1 = i_seq_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp1 * wp1 + (row - 1) * wp1 + col
            wi = (((co - 1) * c21 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            cat1padb[xi] = cat1padb[xi] + w_d1a[wi] * sb
            w_d1ab[wi] = w_d1ab[wi] + cat1pad[xi] * sb
        end
        sb = 0.0
    end
    for idx = 1:c21 * hw
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
        cat1b[idx] = cat1b[idx] + cat1padb[yi]
        cat1padb[yi] = 0.0
    end
    for i = 1:n_cat1pad
        cat1padb[i] = 0.0
    end
    for idx = 1:c1 * hw
        skip1b[idx] = skip1b[idx] + cat1b[c2 * hw + idx]
        cat1b[c2 * hw + idx] = 0.0
    end
    for idx = 1:c2 * hw
        u1b[idx] = u1b[idx] + cat1b[idx]
        cat1b[idx] = 0.0
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
        dec2outb[xi] = dec2outb[xi] + u1b[idx]
        u1b[idx] = 0.0
    end
    for i = n_d2_out:-1:1
        dec2out[i] = dec2out_stack[(i - 1) + 1]
        dec2outb[i] = (0.5 * (1.0 + sign(dec2out[i] - zero_val))) * dec2outb[i]
    end
    for idx = 1:c2 * hw2
        idxm1 = idx - 1
        co = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        sb = sb + dec2outb[idx]
        b_d2bb[co] = b_d2bb[co] + dec2outb[idx]
        dec2outb[idx] = 0.0
        for i_seq_k = c2 * khkw:-1:1
            kseqm1 = i_seq_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp2 * wp2 + (row - 1) * wp2 + col
            wi = (((co - 1) * c2 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            t_d2padb[xi] = t_d2padb[xi] + w_d2b[wi] * sb
            w_d2bb[wi] = w_d2bb[wi] + t_d2pad[xi] * sb
        end
        sb = 0.0
    end
    for idx = 1:c2 * hw2
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
        t_d2b[idx] = t_d2b[idx] + t_d2padb[yi]
        t_d2padb[yi] = 0.0
    end
    for i = 1:n_d2_midpad
        t_d2padb[i] = 0.0
    end
    for i = n_d2_mid:-1:1
        t_d2[i] = t_d2_stack[(i - 1) + 1]
        t_d2b[i] = (0.5 * (1.0 + sign(t_d2[i] - zero_val))) * t_d2b[i]
    end
    for idx = 1:c2 * hw2
        idxm1 = idx - 1
        co = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        sb = sb + t_d2b[idx]
        b_d2ab[co] = b_d2ab[co] + t_d2b[idx]
        t_d2b[idx] = 0.0
        for i_seq_k = c32 * khkw:-1:1
            kseqm1 = i_seq_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp2 * wp2 + (row - 1) * wp2 + col
            wi = (((co - 1) * c32 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            cat2padb[xi] = cat2padb[xi] + w_d2a[wi] * sb
            w_d2ab[wi] = w_d2ab[wi] + cat2pad[xi] * sb
        end
        sb = 0.0
    end
    for idx = 1:c32 * hw2
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
        cat2b[idx] = cat2b[idx] + cat2padb[yi]
        cat2padb[yi] = 0.0
    end
    for i = 1:n_cat2pad
        cat2padb[i] = 0.0
    end
    for idx = 1:c2 * hw2
        skip2b[idx] = skip2b[idx] + cat2b[c3 * hw2 + idx]
        cat2b[c3 * hw2 + idx] = 0.0
    end
    for idx = 1:c3 * hw2
        u2b[idx] = u2b[idx] + cat2b[idx]
        cat2b[idx] = 0.0
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
        bottb[xi] = bottb[xi] + u2b[idx]
        u2b[idx] = 0.0
    end
    for i = n_b_out:-1:1
        bott[i] = bott_stack[(i - 1) + 1]
        bottb[i] = (0.5 * (1.0 + sign(bott[i] - zero_val))) * bottb[i]
    end
    for idx = 1:c3 * hw4
        idxm1 = idx - 1
        co = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        i = div(rem, w4) + 1
        j = mod(rem, w4) + 1
        sb = sb + bottb[idx]
        b_bbb[co] = b_bbb[co] + bottb[idx]
        bottb[idx] = 0.0
        for i_seq_k = c3 * khkw:-1:1
            kseqm1 = i_seq_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp4 * wp4 + (row - 1) * wp4 + col
            wi = (((co - 1) * c3 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            t_bpadb[xi] = t_bpadb[xi] + w_bb[wi] * sb
            w_bbb[wi] = w_bbb[wi] + t_bpad[xi] * sb
        end
        sb = 0.0
    end
    for idx = 1:c3 * hw4
        idxm1 = idx - 1
        ci = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        i = div(rem, w4) + 1
        j = mod(rem, w4) + 1
        yi = (ci - 1) * hp4 * wp4 + ((i + pad) - 1) * wp4 + (j + pad)
        t_bb[idx] = t_bb[idx] + t_bpadb[yi]
        t_bpadb[yi] = 0.0
    end
    for i = 1:n_b_midpad
        t_bpadb[i] = 0.0
    end
    for i = n_b_mid:-1:1
        t_b[i] = t_b_stack[(i - 1) + 1]
        t_bb[i] = (0.5 * (1.0 + sign(t_b[i] - zero_val))) * t_bb[i]
    end
    for idx = 1:c3 * hw4
        idxm1 = idx - 1
        co = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        i = div(rem, w4) + 1
        j = mod(rem, w4) + 1
        sb = sb + t_bb[idx]
        b_bab[co] = b_bab[co] + t_bb[idx]
        t_bb[idx] = 0.0
        for i_seq_k = c2 * khkw:-1:1
            kseqm1 = i_seq_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp4 * wp4 + (row - 1) * wp4 + col
            wi = (((co - 1) * c2 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            p2padb[xi] = p2padb[xi] + w_ba[wi] * sb
            w_bab[wi] = w_bab[wi] + p2pad[xi] * sb
        end
        sb = 0.0
    end
    for idx = 1:c2 * hw4
        idxm1 = idx - 1
        ci = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        i = div(rem, w4) + 1
        j = mod(rem, w4) + 1
        yi = (ci - 1) * hp4 * wp4 + ((i + pad) - 1) * wp4 + (j + pad)
        p2b[idx] = p2b[idx] + p2padb[yi]
        p2padb[yi] = 0.0
    end
    for i = 1:n_p2pad
        p2padb[i] = 0.0
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
        idxm1 = idx - 1
        ci = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        oi = div(rem, w4) + 1
        oj = mod(rem, w4) + 1
        i0 = 2oi - 1
        j0 = 2oj - 1
        m1b = m1b + (0.5 * (1.0 + sign(m1 - m2))) * p2b[idx]
        m2b = m2b + (0.5 * (1.0 + sign(m2 - m1))) * p2b[idx]
        p2b[idx] = 0.0
        a21b = a21b + (0.5 * (1.0 + sign(a21 - a22))) * m2b
        a22b = a22b + (0.5 * (1.0 + sign(a22 - a21))) * m2b
        m2b = 0.0
        a11b = a11b + (0.5 * (1.0 + sign(a11 - a12))) * m1b
        a12b = a12b + (0.5 * (1.0 + sign(a12 - a11))) * m1b
        m1b = 0.0
        skip2b[(ci - 1) * hw2 + i0 * w2 + j0 + 1] = skip2b[(ci - 1) * hw2 + i0 * w2 + j0 + 1] + a22b
        a22b = 0.0
        skip2b[(ci - 1) * hw2 + i0 * w2 + j0] = skip2b[(ci - 1) * hw2 + i0 * w2 + j0] + a21b
        a21b = 0.0
        skip2b[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1] = skip2b[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1] + a12b
        a12b = 0.0
        skip2b[(ci - 1) * hw2 + (i0 - 1) * w2 + j0] = skip2b[(ci - 1) * hw2 + (i0 - 1) * w2 + j0] + a11b
        a11b = 0.0
    end
    for i = n_e2_out:-1:1
        skip2[i] = skip2_stack[(i - 1) + 1]
        skip2b[i] = (0.5 * (1.0 + sign(skip2[i] - zero_val))) * skip2b[i]
    end
    for idx = 1:c2 * hw2
        idxm1 = idx - 1
        co = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        sb = sb + skip2b[idx]
        b_e2bb[co] = b_e2bb[co] + skip2b[idx]
        skip2b[idx] = 0.0
        for i_seq_k = c2 * khkw:-1:1
            kseqm1 = i_seq_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp2 * wp2 + (row - 1) * wp2 + col
            wi = (((co - 1) * c2 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            t_e2padb[xi] = t_e2padb[xi] + w_e2b[wi] * sb
            w_e2bb[wi] = w_e2bb[wi] + t_e2pad[xi] * sb
        end
        sb = 0.0
    end
    for idx = 1:c2 * hw2
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
        t_e2b[idx] = t_e2b[idx] + t_e2padb[yi]
        t_e2padb[yi] = 0.0
    end
    for i = 1:n_e2_midpad
        t_e2padb[i] = 0.0
    end
    for i = n_e2_mid:-1:1
        t_e2[i] = t_e2_stack[(i - 1) + 1]
        t_e2b[i] = (0.5 * (1.0 + sign(t_e2[i] - zero_val))) * t_e2b[i]
    end
    for idx = 1:c2 * hw2
        idxm1 = idx - 1
        co = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        sb = sb + t_e2b[idx]
        b_e2ab[co] = b_e2ab[co] + t_e2b[idx]
        t_e2b[idx] = 0.0
        for i_seq_k = c1 * khkw:-1:1
            kseqm1 = i_seq_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp2 * wp2 + (row - 1) * wp2 + col
            wi = (((co - 1) * c1 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            p1padb[xi] = p1padb[xi] + w_e2a[wi] * sb
            w_e2ab[wi] = w_e2ab[wi] + p1pad[xi] * sb
        end
        sb = 0.0
    end
    for idx = 1:c1 * hw2
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
        p1b[idx] = p1b[idx] + p1padb[yi]
        p1padb[yi] = 0.0
    end
    for i = 1:n_p1pad
        p1padb[i] = 0.0
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
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        oi = div(rem, w2) + 1
        oj = mod(rem, w2) + 1
        i0 = 2oi - 1
        j0 = 2oj - 1
        m1b = m1b + (0.5 * (1.0 + sign(m1 - m2))) * p1b[idx]
        m2b = m2b + (0.5 * (1.0 + sign(m2 - m1))) * p1b[idx]
        p1b[idx] = 0.0
        a21b = a21b + (0.5 * (1.0 + sign(a21 - a22))) * m2b
        a22b = a22b + (0.5 * (1.0 + sign(a22 - a21))) * m2b
        m2b = 0.0
        a11b = a11b + (0.5 * (1.0 + sign(a11 - a12))) * m1b
        a12b = a12b + (0.5 * (1.0 + sign(a12 - a11))) * m1b
        m1b = 0.0
        skip1b[(ci - 1) * hw + i0 * w + j0 + 1] = skip1b[(ci - 1) * hw + i0 * w + j0 + 1] + a22b
        a22b = 0.0
        skip1b[(ci - 1) * hw + i0 * w + j0] = skip1b[(ci - 1) * hw + i0 * w + j0] + a21b
        a21b = 0.0
        skip1b[(ci - 1) * hw + (i0 - 1) * w + j0 + 1] = skip1b[(ci - 1) * hw + (i0 - 1) * w + j0 + 1] + a12b
        a12b = 0.0
        skip1b[(ci - 1) * hw + (i0 - 1) * w + j0] = skip1b[(ci - 1) * hw + (i0 - 1) * w + j0] + a11b
        a11b = 0.0
    end
    for i = n_e1_out:-1:1
        skip1[i] = skip1_stack[(i - 1) + 1]
        skip1b[i] = (0.5 * (1.0 + sign(skip1[i] - zero_val))) * skip1b[i]
    end
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        sb = sb + skip1b[idx]
        b_e1bb[co] = b_e1bb[co] + skip1b[idx]
        skip1b[idx] = 0.0
        for i_seq_k = c1 * khkw:-1:1
            kseqm1 = i_seq_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp1 * wp1 + (row - 1) * wp1 + col
            wi = (((co - 1) * c1 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            t_e1padb[xi] = t_e1padb[xi] + w_e1b[wi] * sb
            w_e1bb[wi] = w_e1bb[wi] + t_e1pad[xi] * sb
        end
        sb = 0.0
    end
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
        t_e1b[idx] = t_e1b[idx] + t_e1padb[yi]
        t_e1padb[yi] = 0.0
    end
    for i = 1:n_e1_midpad
        t_e1padb[i] = 0.0
    end
    for i = n_e1_mid:-1:1
        t_e1[i] = t_e1_stack[(i - 1) + 1]
        t_e1b[i] = (0.5 * (1.0 + sign(t_e1[i] - zero_val))) * t_e1b[i]
    end
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        sb = sb + t_e1b[idx]
        b_e1ab[co] = b_e1ab[co] + t_e1b[idx]
        t_e1b[idx] = 0.0
        for i_seq_k = c_in * khkw:-1:1
            kseqm1 = i_seq_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = (i + ki) - 1
            col = (j + kj) - 1
            xi = (ci - 1) * hp1 * wp1 + (row - 1) * wp1 + col
            wi = (((co - 1) * c_in + (ci - 1)) * kh + (ki - 1)) * kw + kj
            xpad0b[xi] = xpad0b[xi] + w_e1a[wi] * sb
            w_e1ab[wi] = w_e1ab[wi] + xpad0[xi] * sb
        end
        sb = 0.0
    end
    for idx = 1:c_in * hw
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
        xb[idx] = xb[idx] + xpad0b[yi]
        xpad0b[yi] = 0.0
    end
    for i = 1:n_xpad0
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
        for i_seq_k = 1:c_in * khkw
            kseqm1 = i_seq_k - 1
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
        for i_seq_k = 1:c1 * khkw
            kseqm1 = i_seq_k - 1
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
        for i_seq_k = 1:c1 * khkw
            kseqm1 = i_seq_k - 1
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
        for i_seq_k = 1:c2 * khkw
            kseqm1 = i_seq_k - 1
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
        for i_seq_k = 1:c2 * khkw
            kseqm1 = i_seq_k - 1
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
        for i_seq_k = 1:c3 * khkw
            kseqm1 = i_seq_k - 1
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
        for i_seq_k = 1:c32 * khkw
            kseqm1 = i_seq_k - 1
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
        for i_seq_k = 1:c2 * khkw
            kseqm1 = i_seq_k - 1
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
        for i_seq_k = 1:c21 * khkw
            kseqm1 = i_seq_k - 1
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
        for i_seq_k = 1:c1 * khkw
            kseqm1 = i_seq_k - 1
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
        for i_seq_k = 1:c1 * khkw_out
            kseqm1 = i_seq_k - 1
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
