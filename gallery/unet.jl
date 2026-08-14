# zero_array(y, n)
#
# Fills y with zeros.
#
# y: output array of length n, filled in place
# n: number of elements to zero
function zero_array(y, n)
    for i = 1:n
        y[i] = 0.0
    end
    return nothing
end

# pad_channels(x, y, c, h, w, pad, hp, wp)
#
# Copies a c-channel, h-by-w feature map x into the zero-padded interior
# of y, a c-channel, hp-by-wp feature map (hp = h + 2*pad, wp = w + 2*pad).
# y must already be zero-filled (see zero_array) before calling this.
#
# x: input feature map, length c * h * w, layout (channel, row, col)
# y: output padded feature map, length c * hp * wp, same layout, pre-zeroed
# c: number of channels
# h: input height
# w: input width
# pad: padding width on each side
# hp: padded height, h + 2 * pad
# wp: padded width, w + 2 * pad
function pad_channels(x, y, c, h, w, pad, hp, wp)
    # every input element maps to exactly one interior element of y,
    # independently of every other element
    hw = h * w
    for idx = 1:c * h * w
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp * wp + (i + pad - 1) * wp + (j + pad)
        y[yi] = x[idx]
    end
    return nothing
end

# conv2d(xpad, wgt, bias, y, c_in, c_out, h, w, kh, kw, pad, hp, wp)
#
# 2-D convolution, stride 1, on an already zero-padded input, producing an
# h-by-w output with c_out channels ("same" convolution when
# pad = div(kh - 1, 2) = div(kw - 1, 2)).
#
# xpad: zero-padded input, length c_in * hp * wp, layout (channel, row, col)
# wgt: filter weights, length c_out * c_in * kh * kw,
#      layout (out channel, in channel, kernel row, kernel col)
# bias: bias per output channel, length c_out
# y: output feature map, length c_out * h * w, filled in place
# c_in: number of input channels
# c_out: number of output channels
# h: output height
# w: output width
# kh: kernel height
# kw: kernel width
# pad: padding width that was used to build xpad
# hp: padded input height, h + 2 * pad
# wp: padded input width, w + 2 * pad
function conv2d(xpad, wgt, bias, y, c_in, c_out, h, w, kh, kw, pad, hp, wp)
    # output channel, row, and column are all independent of one another and
    # the ranges are fixed, so the three loops fuse into one flattened loop
    hw = h * w
    khkw = kh * kw
    for idx = 1:c_out * h * w
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        s = 0.0
        # the convolution sum is a genuine reduction over input channel and
        # kernel offsets, so this inner loop is sequential
        for i_seq_k = 1:c_in * khkw
            kseqm1 = i_seq_k - 1
            ci = div(kseqm1, khkw) + 1
            rem2 = mod(kseqm1, khkw)
            ki = div(rem2, kw) + 1
            kj = mod(rem2, kw) + 1
            row = i + ki - 1
            col = j + kj - 1
            xi = (ci - 1) * hp * wp + (row - 1) * wp + col
            wi = (((co - 1) * c_in + (ci - 1)) * kh + (ki - 1)) * kw + kj
            s = s + xpad[xi] * wgt[wi]
        end
        y[idx] = s + bias[co]
    end
    return nothing
end

# relu(x, y, n)
#
# Elementwise rectified linear unit, y[i] = max(x[i], 0.0). x and y may be
# the same array.
#
# x: input array of length n
# y: output array of length n, filled in place
# n: number of elements
function relu(x, y, n)
    zero_val = 0.0
    for i = 1:n
        y[i] = max(x[i], zero_val)
    end
    return nothing
end

# maxpool2x2(x, y, c, h, w, oh, ow)
#
# 2-by-2 max pooling with stride 2. h and w must be even; oh = div(h, 2),
# ow = div(w, 2).
#
# x: input feature map, length c * h * w, layout (channel, row, col)
# y: output feature map, length c * oh * ow, filled in place, same layout
# c: number of channels
# h: input height
# w: input width
# oh: output height, div(h, 2)
# ow: output width, div(w, 2)
function maxpool2x2(x, y, c, h, w, oh, ow)
    # every output element reduces a disjoint 2-by-2 window, independently
    # of every other output element
    ohow = oh * ow
    for idx = 1:c * oh * ow
        idxm1 = idx - 1
        ci = div(idxm1, ohow) + 1
        rem = mod(idxm1, ohow)
        oi = div(rem, ow) + 1
        oj = mod(rem, ow) + 1
        i0 = 2 * oi - 1
        j0 = 2 * oj - 1
        a11 = x[(ci - 1) * h * w + (i0 - 1) * w + j0]
        a12 = x[(ci - 1) * h * w + (i0 - 1) * w + j0 + 1]
        a21 = x[(ci - 1) * h * w + i0 * w + j0]
        a22 = x[(ci - 1) * h * w + i0 * w + j0 + 1]
        m1 = max(a11, a12)
        m2 = max(a21, a22)
        y[idx] = max(m1, m2)
    end
    return nothing
end

# upsample2x(x, y, c, h, w, oh, ow)
#
# 2x nearest-neighbor spatial upsampling. oh = 2 * h, ow = 2 * w.
#
# x: input feature map, length c * h * w, layout (channel, row, col)
# y: output feature map, length c * oh * ow, filled in place, same layout
# c: number of channels
# h: input height
# w: input width
# oh: output height, 2 * h
# ow: output width, 2 * w
function upsample2x(x, y, c, h, w, oh, ow)
    # every output element reads exactly one input element, independently
    # of every other output element
    ohow = oh * ow
    scale = 2
    for idx = 1:c * oh * ow
        idxm1 = idx - 1
        ci = div(idxm1, ohow) + 1
        rem = mod(idxm1, ohow)
        oi = div(rem, ow) + 1
        oj = mod(rem, ow) + 1
        oim1 = oi - 1
        ojm1 = oj - 1
        i = div(oim1, scale) + 1
        j = div(ojm1, scale) + 1
        xi = (ci - 1) * h * w + (i - 1) * w + j
        y[idx] = x[xi]
    end
    return nothing
end

# concat_channels(x1, x2, y, c1, c2, h, w)
#
# Concatenates two feature maps along the channel dimension: the first c1
# channels of y come from x1, the remaining c2 channels come from x2.
#
# x1: first input feature map, length c1 * h * w, layout (channel, row, col)
# x2: second input feature map, length c2 * h * w, same layout
# y: output feature map, length (c1 + c2) * h * w, filled in place
# c1: number of channels in x1
# c2: number of channels in x2
# h: height shared by x1, x2, and y
# w: width shared by x1, x2, and y
function concat_channels(x1, x2, y, c1, c2, h, w)
    # copy x1's channels into the head of y
    for idx = 1:c1 * h * w
        y[idx] = x1[idx]
    end
    # copy x2's channels into the tail of y
    for idx = 1:c2 * h * w
        y[c1 * h * w + idx] = x2[idx]
    end
    return nothing
end

# double_conv_block(xpad, w1, b1, w2, b2, t1, t1pad, y, c_in, c_mid, c_out, h, w, kh, kw, pad, hp, wp)
#
# The U-Net "double convolution" building block: 3-by-3 same-convolution,
# ReLU, 3-by-3 same-convolution, ReLU. Input must already be zero-padded;
# output is not padded.
#
# xpad: zero-padded input, length c_in * hp * wp
# w1: first conv's weights, length c_mid * c_in * kh * kw
# b1: first conv's bias, length c_mid
# w2: second conv's weights, length c_out * c_mid * kh * kw
# b2: second conv's bias, length c_out
# t1: scratch buffer for the first conv's (unpadded) output, length c_mid * h * w
# t1pad: scratch buffer for the zero-padded version of t1, length c_mid * hp * wp
# y: output feature map, length c_out * h * w, filled in place
# c_in: number of input channels
# c_mid: number of channels after the first convolution
# c_out: number of channels after the second convolution
# h: feature map height (unchanged by either convolution)
# w: feature map width (unchanged by either convolution)
# kh: kernel height
# kw: kernel width
# pad: padding width, div(kh - 1, 2)
# hp: padded height, h + 2 * pad
# wp: padded width, w + 2 * pad
function double_conv_block(xpad, w1, b1, w2, b2, t1, t1pad, y, c_in, c_mid, c_out, h, w, kh, kw, pad, hp, wp)
    # sizes needed below, computed once so every call below takes a bare
    # symbol rather than an inline expression
    n_mid = c_mid * h * w
    n_mid_pad = c_mid * hp * wp
    n_out = c_out * h * w
    # first conv + relu
    conv2d(xpad, w1, b1, t1, c_in, c_mid, h, w, kh, kw, pad, hp, wp)
    relu(t1, t1, n_mid)
    # re-pad before the second conv
    zero_array(t1pad, n_mid_pad)
    pad_channels(t1, t1pad, c_mid, h, w, pad, hp, wp)
    # second conv + relu
    conv2d(t1pad, w2, b2, y, c_mid, c_out, h, w, kh, kw, pad, hp, wp)
    relu(y, y, n_out)
    return nothing
end

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
# argument below is allocated by the caller.
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
    # 3-by-3 kernel size shared by every double_conv_block call
    kh = 3
    kw = 3
    # 1-by-1, unpadded kernel used by the final output projection
    kh_out = 1
    kw_out = 1
    pad_out = 0
    # channel counts at the two concatenation points, and the padded-buffer
    # sizes below, all computed once so every call takes a bare symbol
    # rather than an inline expression
    c32 = c3 + c2
    c21 = c2 + c1
    n_xpad0 = c_in * hp1 * wp1
    n_p1pad = c1 * hp2 * wp2
    n_p2pad = c2 * hp4 * wp4
    n_cat2pad = c32 * hp2 * wp2
    n_cat1pad = c21 * hp1 * wp1

    # encoder stage 1, at full resolution
    zero_array(xpad0, n_xpad0)
    pad_channels(x, xpad0, c_in, h, w, pad, hp1, wp1)
    double_conv_block(xpad0, w_e1a, b_e1a, w_e1b, b_e1b, t_e1, t_e1pad, skip1, c_in, c1, c1, h, w, kh, kw, pad, hp1, wp1)
    maxpool2x2(skip1, p1, c1, h, w, h2, w2)

    # encoder stage 2, at half resolution
    zero_array(p1pad, n_p1pad)
    pad_channels(p1, p1pad, c1, h2, w2, pad, hp2, wp2)
    double_conv_block(p1pad, w_e2a, b_e2a, w_e2b, b_e2b, t_e2, t_e2pad, skip2, c1, c2, c2, h2, w2, kh, kw, pad, hp2, wp2)
    maxpool2x2(skip2, p2, c2, h2, w2, h4, w4)

    # bottleneck, at quarter resolution
    zero_array(p2pad, n_p2pad)
    pad_channels(p2, p2pad, c2, h4, w4, pad, hp4, wp4)
    double_conv_block(p2pad, w_ba, b_ba, w_bb, b_bb, t_b, t_bpad, bott, c2, c3, c3, h4, w4, kh, kw, pad, hp4, wp4)

    # decoder stage 2: upsample to half resolution, concat with skip2
    upsample2x(bott, u2, c3, h4, w4, h2, w2)
    concat_channels(u2, skip2, cat2, c3, c2, h2, w2)
    zero_array(cat2pad, n_cat2pad)
    pad_channels(cat2, cat2pad, c32, h2, w2, pad, hp2, wp2)
    double_conv_block(cat2pad, w_d2a, b_d2a, w_d2b, b_d2b, t_d2, t_d2pad, dec2out, c32, c2, c2, h2, w2, kh, kw, pad, hp2, wp2)

    # decoder stage 1: upsample to full resolution, concat with skip1
    upsample2x(dec2out, u1, c2, h2, w2, h, w)
    concat_channels(u1, skip1, cat1, c2, c1, h, w)
    zero_array(cat1pad, n_cat1pad)
    pad_channels(cat1, cat1pad, c21, h, w, pad, hp1, wp1)
    double_conv_block(cat1pad, w_d1a, b_d1a, w_d1b, b_d1b, t_d1, t_d1pad, dec1out, c21, c1, c1, h, w, kh, kw, pad, hp1, wp1)

    # output projection: 1-by-1 conv, no padding needed (pad = 0, hp = h, wp = w)
    conv2d(dec1out, w_out, b_out, y, c1, c_out, h, w, kh_out, kw_out, pad_out, h, w)
    return nothing
end