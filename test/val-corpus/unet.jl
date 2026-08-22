# unet(x, h, w, c_in, c1, c2, c3, c_out,
#      w_e1a, b_e1a, w_e1b, b_e1b, w_e2a, b_e2a, w_e2b, b_e2b,
#      w_ba, b_ba, w_bb, b_bb,
#      w_d2a, b_d2a, w_d2b, b_d2b, w_d1a, b_d1a, w_d1b, b_d1b,
#      w_out, b_out,
#      xpad0, t_e1, t_e1pad, skip1, p1, p1pad,
#      t_e2, t_e2pad, skip2, p2, p2pad,
#      t_b, t_bpad, bott,
#      u2, cat2, cat2pad, t_d2, t_d2pad, dec2out,
#      u1, cat1, cat1pad, t_d1, t_d1pad, dec1out,
#      y)
#
# A depth-2 U-Net: two encoder double-conv stages (each followed by 2x2 max
# pooling), a bottleneck double-conv stage, two decoder double-conv stages
# (each preceded by 2x nearest-neighbor upsampling and a skip-connection
# concatenation), and a final 1-by-1 convolution to c_out channels. All
# convolutions are 3-by-3, stride 1, "same" (except the final 1-by-1
# output convolution). h and w must both be divisible by 4. Every array
# argument below is allocated by the caller. Every stage that used to be
# its own kernel (zero-fill, zero-pad-and-copy, convolution, ReLU, max
# pool, upsample, channel concat) is inlined directly into this one
# function instead of being called out to a helper.
#
# x: input image, length c_in * h * w, layout (channel, row, col)
# h: input height
# w: input width
# c_in: number of input channels
# c1: encoder/decoder channel count at the finest resolution (h, w)
# c2: encoder/decoder channel count at the half resolution (h/2, w/2)
# c3: bottleneck channel count at the quarter resolution (h/4, w/4)
# c_out: number of output channels
# w_e1a, b_e1a: encoder-1 first conv weights/bias (c1, c_in, 3, 3) / (c1)
# w_e1b, b_e1b: encoder-1 second conv weights/bias (c1, c1, 3, 3) / (c1)
# w_e2a, b_e2a: encoder-2 first conv weights/bias (c2, c1, 3, 3) / (c2)
# w_e2b, b_e2b: encoder-2 second conv weights/bias (c2, c2, 3, 3) / (c2)
# w_ba, b_ba: bottleneck first conv weights/bias (c3, c2, 3, 3) / (c3)
# w_bb, b_bb: bottleneck second conv weights/bias (c3, c3, 3, 3) / (c3)
# w_d2a, b_d2a: decoder-2 first conv weights/bias (c2, c3 + c2, 3, 3) / (c2)
# w_d2b, b_d2b: decoder-2 second conv weights/bias (c2, c2, 3, 3) / (c2)
# w_d1a, b_d1a: decoder-1 first conv weights/bias (c1, c2 + c1, 3, 3) / (c1)
# w_d1b, b_d1b: decoder-1 second conv weights/bias (c1, c1, 3, 3) / (c1)
# w_out, b_out: output 1-by-1 conv weights/bias (c_out, c1, 1, 1) / (c_out)
# xpad0: scratch, padded input, length c_in * (h + 2) * (w + 2)
# t_e1: scratch, length c1 * h * w
# t_e1pad: scratch, length c1 * (h + 2) * (w + 2)
# skip1: encoder-1 output, length c1 * h * w
# p1: pooled encoder-1 output, length c1 * (h / 2) * (w / 2)
# p1pad: scratch, length c1 * (h / 2 + 2) * (w / 2 + 2)
# t_e2: scratch, length c2 * (h / 2) * (w / 2)
# t_e2pad: scratch, length c2 * (h / 2 + 2) * (w / 2 + 2)
# skip2: encoder-2 output, length c2 * (h / 2) * (w / 2)
# p2: pooled encoder-2 output, length c2 * (h / 4) * (w / 4)
# p2pad: scratch, length c2 * (h / 4 + 2) * (w / 4 + 2)
# t_b: scratch, length c3 * (h / 4) * (w / 4)
# t_bpad: scratch, length c3 * (h / 4 + 2) * (w / 4 + 2)
# bott: bottleneck output, length c3 * (h / 4) * (w / 4)
# u2: upsampled bottleneck output, length c3 * (h / 2) * (w / 2)
# cat2: skip2 concatenated with u2, length (c3 + c2) * (h / 2) * (w / 2)
# cat2pad: scratch, length (c3 + c2) * (h / 2 + 2) * (w / 2 + 2)
# t_d2: scratch, length c2 * (h / 2) * (w / 2)
# t_d2pad: scratch, length c2 * (h / 2 + 2) * (w / 2 + 2)
# dec2out: decoder-2 output, length c2 * (h / 2) * (w / 2)
# u1: upsampled decoder-2 output, length c2 * h * w
# cat1: skip1 concatenated with u1, length (c2 + c1) * h * w
# cat1pad: scratch, length (c2 + c1) * (h + 2) * (w + 2)
# t_d1: scratch, length c1 * h * w
# t_d1pad: scratch, length c1 * (h + 2) * (w + 2)
# dec1out: decoder-1 output, length c1 * h * w
# y: final output, length c_out * h * w, filled in place
function unet(x, h, w, c_in, c1, c2, c3, c_out,
              w_e1a, b_e1a, w_e1b, b_e1b, w_e2a, b_e2a, w_e2b, b_e2b,
              w_ba, b_ba, w_bb, b_bb,
              w_d2a, b_d2a, w_d2b, b_d2b, w_d1a, b_d1a, w_d1b, b_d1b,
              w_out, b_out,
              xpad0, t_e1, t_e1pad, skip1, p1, p1pad,
              t_e2, t_e2pad, skip2, p2, p2pad,
              t_b, t_bpad, bott,
              u2, cat2, cat2pad, t_d2, t_d2pad, dec2out,
              u1, cat1, cat1pad, t_d1, t_d1pad, dec1out,
              y)
    # resolution at each of the three scales
    two = 2
    four = 4
    h2 = div(h, two)
    w2 = div(w, two)
    h4 = div(h, four)
    w4 = div(w, four)
    # padded dimensions (pad = 1 for every 3-by-3 "same" conv)
    hp1 = h + 2
    wp1 = w + 2
    hp2 = h2 + 2
    wp2 = w2 + 2
    hp4 = h4 + 2
    wp4 = w4 + 2
    # padding width used by every 3-by-3 "same" convolution
    pad = 1
    # 3-by-3 kernel size shared by every double-conv stage below
    kh = 3
    kw = 3
    khkw = kh * kw
    # 1-by-1, unpadded kernel used by the final output projection
    kh_out = 1
    kw_out = 1
    khkw_out = kh_out * kw_out
    # nearest-neighbor upsampling factor
    scale = 2
    zero_val = 0.0
    # channel count at each of the two concatenation points
    c32 = c3 + c2
    c21 = c2 + c1
    # per-channel spatial sizes at each of the three scales, computed once
    hw = h * w
    hw2 = h2 * w2
    hw4 = h4 * w4
    # flattened-buffer sizes needed for the zero-fill / ReLU sweeps below,
    # all computed once so every loop bound is a bare symbol
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

    # ================= encoder stage 1, at full resolution =================
    # zero xpad0, then scatter x into its zero-padded interior; every input
    # element maps to exactly one interior element, independently of every
    # other element
    for i = 1:n_xpad0
        xpad0[i] = zero_val
    end
    for idx = 1:c_in * hw
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp1 * wp1 + (i + pad - 1) * wp1 + (j + pad)
        xpad0[yi] = x[idx]
    end
    # encoder-1 first conv: xpad0 (c_in channels) -> t_e1 (c1 channels).
    # output channel/row/col are independent, so they form one flattened
    # loop; the sum over input channel and kernel offsets is a genuine
    # reduction, so that inner loop is sequential
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
            row = i + ki - 1
            col = j + kj - 1
            xi = (ci - 1) * hp1 * wp1 + (row - 1) * wp1 + col
            wi = (((co - 1) * c_in + (ci - 1)) * kh + (ki - 1)) * kw + kj
            s = s + xpad0[xi] * w_e1a[wi]
        end
        t_e1[idx] = s + b_e1a[co]
    end
    # encoder-1 first conv's ReLU, in place
    for i = 1:n_e1_mid
        t_e1[i] = max(t_e1[i], zero_val)
    end
    # re-pad t_e1 before the encoder-1 second conv
    for i = 1:n_e1_midpad
        t_e1pad[i] = zero_val
    end
    for idx = 1:c1 * hw
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp1 * wp1 + (i + pad - 1) * wp1 + (j + pad)
        t_e1pad[yi] = t_e1[idx]
    end
    # encoder-1 second conv: t_e1pad (c1 channels) -> skip1 (c1 channels)
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
            row = i + ki - 1
            col = j + kj - 1
            xi = (ci - 1) * hp1 * wp1 + (row - 1) * wp1 + col
            wi = (((co - 1) * c1 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            s = s + t_e1pad[xi] * w_e1b[wi]
        end
        skip1[idx] = s + b_e1b[co]
    end
    # encoder-1 second conv's ReLU, in place; skip1 is both this stage's
    # output and the skip connection consumed later by decoder-1
    for i = 1:n_e1_out
        skip1[i] = max(skip1[i], zero_val)
    end
    # 2x2 max pool, stride 2: every output element reduces a disjoint
    # 2-by-2 window, independently of every other output element
    for idx = 1:c1 * hw2
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        oi = div(rem, w2) + 1
        oj = mod(rem, w2) + 1
        i0 = 2 * oi - 1
        j0 = 2 * oj - 1
        a11 = skip1[(ci - 1) * hw + (i0 - 1) * w + j0]
        a12 = skip1[(ci - 1) * hw + (i0 - 1) * w + j0 + 1]
        a21 = skip1[(ci - 1) * hw + i0 * w + j0]
        a22 = skip1[(ci - 1) * hw + i0 * w + j0 + 1]
        m1 = max(a11, a12)
        m2 = max(a21, a22)
        p1[idx] = max(m1, m2)
    end

    # ================= encoder stage 2, at half resolution =================
    for i = 1:n_p1pad
        p1pad[i] = zero_val
    end
    for idx = 1:c1 * hw2
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        yi = (ci - 1) * hp2 * wp2 + (i + pad - 1) * wp2 + (j + pad)
        p1pad[yi] = p1[idx]
    end
    # encoder-2 first conv: p1pad (c1 channels) -> t_e2 (c2 channels)
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
            row = i + ki - 1
            col = j + kj - 1
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
        yi = (ci - 1) * hp2 * wp2 + (i + pad - 1) * wp2 + (j + pad)
        t_e2pad[yi] = t_e2[idx]
    end
    # encoder-2 second conv: t_e2pad (c2 channels) -> skip2 (c2 channels)
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
            row = i + ki - 1
            col = j + kj - 1
            xi = (ci - 1) * hp2 * wp2 + (row - 1) * wp2 + col
            wi = (((co - 1) * c2 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            s = s + t_e2pad[xi] * w_e2b[wi]
        end
        skip2[idx] = s + b_e2b[co]
    end
    for i = 1:n_e2_out
        skip2[i] = max(skip2[i], zero_val)
    end
    # 2x2 max pool, stride 2, halving the spatial resolution again
    for idx = 1:c2 * hw4
        idxm1 = idx - 1
        ci = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        oi = div(rem, w4) + 1
        oj = mod(rem, w4) + 1
        i0 = 2 * oi - 1
        j0 = 2 * oj - 1
        a11 = skip2[(ci - 1) * hw2 + (i0 - 1) * w2 + j0]
        a12 = skip2[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1]
        a21 = skip2[(ci - 1) * hw2 + i0 * w2 + j0]
        a22 = skip2[(ci - 1) * hw2 + i0 * w2 + j0 + 1]
        m1 = max(a11, a12)
        m2 = max(a21, a22)
        p2[idx] = max(m1, m2)
    end

    # ===================== bottleneck, at quarter resolution =====================
    for i = 1:n_p2pad
        p2pad[i] = zero_val
    end
    for idx = 1:c2 * hw4
        idxm1 = idx - 1
        ci = div(idxm1, hw4) + 1
        rem = mod(idxm1, hw4)
        i = div(rem, w4) + 1
        j = mod(rem, w4) + 1
        yi = (ci - 1) * hp4 * wp4 + (i + pad - 1) * wp4 + (j + pad)
        p2pad[yi] = p2[idx]
    end
    # bottleneck first conv: p2pad (c2 channels) -> t_b (c3 channels)
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
            row = i + ki - 1
            col = j + kj - 1
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
        yi = (ci - 1) * hp4 * wp4 + (i + pad - 1) * wp4 + (j + pad)
        t_bpad[yi] = t_b[idx]
    end
    # bottleneck second conv: t_bpad (c3 channels) -> bott (c3 channels)
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
            row = i + ki - 1
            col = j + kj - 1
            xi = (ci - 1) * hp4 * wp4 + (row - 1) * wp4 + col
            wi = (((co - 1) * c3 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            s = s + t_bpad[xi] * w_bb[wi]
        end
        bott[idx] = s + b_bb[co]
    end
    for i = 1:n_b_out
        bott[i] = max(bott[i], zero_val)
    end

    # ============== decoder stage 2: upsample, concat, double conv ==============
    # 2x nearest-neighbor upsample: bott (c3, h4-by-w4) -> u2 (c3, h2-by-w2);
    # every output element reads exactly one input element, independently
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
    # concat u2 (c3 channels) with skip2 (c2 channels) along channels: u2's
    # channels come first, skip2's channels are appended after
    for idx = 1:c3 * hw2
        cat2[idx] = u2[idx]
    end
    for idx = 1:c2 * hw2
        cat2[c3 * hw2 + idx] = skip2[idx]
    end
    # zero-pad cat2 before the decoder-2 double conv
    for i = 1:n_cat2pad
        cat2pad[i] = zero_val
    end
    for idx = 1:c32 * hw2
        idxm1 = idx - 1
        ci = div(idxm1, hw2) + 1
        rem = mod(idxm1, hw2)
        i = div(rem, w2) + 1
        j = mod(rem, w2) + 1
        yi = (ci - 1) * hp2 * wp2 + (i + pad - 1) * wp2 + (j + pad)
        cat2pad[yi] = cat2[idx]
    end
    # decoder-2 first conv: cat2pad (c32 channels) -> t_d2 (c2 channels)
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
            row = i + ki - 1
            col = j + kj - 1
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
        yi = (ci - 1) * hp2 * wp2 + (i + pad - 1) * wp2 + (j + pad)
        t_d2pad[yi] = t_d2[idx]
    end
    # decoder-2 second conv: t_d2pad (c2 channels) -> dec2out (c2 channels)
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
            row = i + ki - 1
            col = j + kj - 1
            xi = (ci - 1) * hp2 * wp2 + (row - 1) * wp2 + col
            wi = (((co - 1) * c2 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            s = s + t_d2pad[xi] * w_d2b[wi]
        end
        dec2out[idx] = s + b_d2b[co]
    end
    for i = 1:n_d2_out
        dec2out[i] = max(dec2out[i], zero_val)
    end

    # ============== decoder stage 1: upsample, concat, double conv ==============
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
        yi = (ci - 1) * hp1 * wp1 + (i + pad - 1) * wp1 + (j + pad)
        cat1pad[yi] = cat1[idx]
    end
    # decoder-1 first conv: cat1pad (c21 channels) -> t_d1 (c1 channels)
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
            row = i + ki - 1
            col = j + kj - 1
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
        yi = (ci - 1) * hp1 * wp1 + (i + pad - 1) * wp1 + (j + pad)
        t_d1pad[yi] = t_d1[idx]
    end
    # decoder-1 second conv: t_d1pad (c1 channels) -> dec1out (c1 channels)
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
            row = i + ki - 1
            col = j + kj - 1
            xi = (ci - 1) * hp1 * wp1 + (row - 1) * wp1 + col
            wi = (((co - 1) * c1 + (ci - 1)) * kh + (ki - 1)) * kw + kj
            s = s + t_d1pad[xi] * w_d1b[wi]
        end
        dec1out[idx] = s + b_d1b[co]
    end
    for i = 1:n_d1_out
        dec1out[i] = max(dec1out[i], zero_val)
    end

    # ===================== output projection: 1-by-1 conv =====================
    # no padding needed (pad = 0, so hp = h, wp = w for this conv only)
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
            row = i + ki - 1
            col = j + kj - 1
            xi = (ci - 1) * hw + (row - 1) * w + col
            wi = (((co - 1) * c1 + (ci - 1)) * kh_out + (ki - 1)) * kw_out + kj
            s = s + dec1out[xi] * w_out[wi]
        end
        y[idx] = s + b_out[co]
    end
    return nothing
end