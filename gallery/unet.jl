# zerofill(buf, n)
#
# Fills the first n elements of buf with zero.
#
# buf: array to clear, length at least n
# n: number of elements to clear
function zerofill(buf, n)
    for i = 1:n
        buf[i] = 0.0
    end
    return nothing
end

# pad_copy(src, dst, c, h, w, pad)
#
# Copies an unpadded (channel, row, col) array into the interior of an
# already-zeroed padded array, leaving a border of width pad on every
# side. Every input element maps to exactly one interior element,
# independently of every other element.
#
# src: input array, length c*h*w, layout (channel, row, col)
# dst: padded output array, length c*(h+2*pad)*(w+2*pad); must already
#      be zero-filled (zerofill) before this call
# c: number of channels
# h: input height
# w: input width
# pad: padding width added on every side
function pad_copy(src, dst, c, h, w, pad)
    hw = h * w
    hp = h + 2 * pad
    wp = w + 2 * pad
    for idx = 1:c * hw
        idxm1 = idx - 1
        ci = div(idxm1, hw) + 1
        rem = mod(idxm1, hw)
        i = div(rem, w) + 1
        j = mod(rem, w) + 1
        yi = (ci - 1) * hp * wp + (i + pad - 1) * wp + (j + pad)
        dst[yi] = src[idx]
    end
    return nothing
end

# conv3x3(src_pad, weight, bias, dst, c_in, c_out, h, w, pad)
#
# 3-by-3, stride-1, "same" convolution from a zero-padded input to an
# unpadded output. Output channel/row/col are independent, so they form
# one flattened loop; the sum over input channel and kernel offsets is
# a genuine reduction, so that inner loop is sequential.
#
# src_pad: padded input array, length c_in*(h+2*pad)*(w+2*pad)
# weight: kernel weights, length c_out*c_in*3*3
# bias: kernel bias, length c_out
# dst: output array, length c_out*h*w, filled in place
# c_in: number of input channels
# c_out: number of output channels
# h: output height
# w: output width
# pad: padding width used on src_pad (1 for a 3-by-3 "same" convolution)
function conv3x3(src_pad, weight, bias, dst, c_in, c_out, h, w, pad)
    hw = h * w
    hp = h + 2 * pad
    wp = w + 2 * pad
    kh = 3
    kw = 3
    khkw = kh * kw
    for idx = 1:c_out * hw
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
            row = i + ki - 1
            col = j + kj - 1
            xi = (ci - 1) * hp * wp + (row - 1) * wp + col
            wi = (((co - 1) * c_in + (ci - 1)) * kh + (ki - 1)) * kw + kj
            s = s + src_pad[xi] * weight[wi]
        end
        dst[idx] = s + bias[co]
    end
    return nothing
end

# relu(buf, n)
#
# Rectified linear unit, applied in place: every negative entry becomes
# zero, every non-negative entry is unchanged.
#
# buf: array to rectify, length at least n
# n: number of elements to rectify
function relu(buf, n)
    for i = 1:n
        buf[i] = max(buf[i], 0.0)
    end
    return nothing
end

# double_conv3x3(src_pad, wa, ba, wb, bb, mid, midpad, dst,
#                 c_in, c_mid, h, w, pad)
#
# Two chained 3-by-3 "same" convolutions with a ReLU after each: the
# first maps c_in channels to c_mid channels, the second keeps c_mid
# channels. Both convolutions read a zero-padded input and write an
# unpadded output; midpad is zero-filled and re-populated here between
# the two convolutions.
#
# src_pad: padded input array, length c_in*(h+2*pad)*(w+2*pad)
# wa, ba: first convolution's weights (c_mid*c_in*3*3) and bias (c_mid)
# wb, bb: second convolution's weights (c_mid*c_mid*3*3) and bias (c_mid)
# mid: scratch array, length c_mid*h*w, for the first convolution's output
# midpad: scratch array, length c_mid*(h+2*pad)*(w+2*pad), the re-padded
#         copy of mid consumed by the second convolution
# dst: output array, length c_mid*h*w, filled in place
# c_in: number of input channels
# c_mid: number of channels after each of the two convolutions
# h: output height
# w: output width
# pad: padding width used by both convolutions (1 for 3-by-3 "same")
function double_conv3x3(src_pad, wa, ba, wb, bb, mid, midpad, dst, c_in, c_mid, h, w, pad)
    n_mid = c_mid * h * w
    n_midpad = c_mid * (h + 2 * pad) * (w + 2 * pad)
    conv3x3(src_pad, wa, ba, mid, c_in, c_mid, h, w, pad)
    relu(mid, n_mid)
    zerofill(midpad, n_midpad)
    pad_copy(mid, midpad, c_mid, h, w, pad)
    conv3x3(midpad, wb, bb, dst, c_mid, c_mid, h, w, pad)
    relu(dst, n_mid)
    return nothing
end

# maxpool2x2(src, dst, c, ho, wo)
#
# 2-by-2 max pooling, stride 2: every output pixel reduces a disjoint
# 2-by-2 window of the input, independently of every other output pixel.
#
# src: input array, length c*(2*ho)*(2*wo), layout (channel, row, col)
# dst: output array, length c*ho*wo, filled in place
# c: number of channels
# ho: output height
# wo: output width
function maxpool2x2(src, dst, c, ho, wo)
    hi = 2 * ho
    wi = 2 * wo
    hwi = hi * wi
    hwo = ho * wo
    for idx = 1:c * hwo
        idxm1 = idx - 1
        ci = div(idxm1, hwo) + 1
        rem = mod(idxm1, hwo)
        oi = div(rem, wo) + 1
        oj = mod(rem, wo) + 1
        i0 = 2 * oi - 1
        j0 = 2 * oj - 1
        a11 = src[(ci - 1) * hwi + (i0 - 1) * wi + j0]
        a12 = src[(ci - 1) * hwi + (i0 - 1) * wi + j0 + 1]
        a21 = src[(ci - 1) * hwi + i0 * wi + j0]
        a22 = src[(ci - 1) * hwi + i0 * wi + j0 + 1]
        m1 = max(a11, a12)
        m2 = max(a21, a22)
        dst[idx] = max(m1, m2)
    end
    return nothing
end

# upsample2x(src, dst, c, hi, wi)
#
# 2x nearest-neighbor upsampling: every output pixel reads exactly one
# input pixel, independently of every other output pixel.
#
# src: input array, length c*hi*wi, layout (channel, row, col)
# dst: output array, length c*(2*hi)*(2*wi), filled in place
# c: number of channels
# hi: input height
# wi: input width
function upsample2x(src, dst, c, hi, wi)
    ho = 2 * hi
    wo = 2 * wi
    hwi = hi * wi
    hwo = ho * wo
    scale = 2
    for idx = 1:c * hwo
        idxm1 = idx - 1
        ci = div(idxm1, hwo) + 1
        rem = mod(idxm1, hwo)
        oi = div(rem, wo) + 1
        oj = mod(rem, wo) + 1
        oim1 = oi - 1
        ojm1 = oj - 1
        i = div(oim1, scale) + 1
        j = div(ojm1, scale) + 1
        xi = (ci - 1) * hwi + (i - 1) * wi + j
        dst[idx] = src[xi]
    end
    return nothing
end

# concat_channels(a, b, dst, ca, cb, hw)
#
# Concatenates two channel-major arrays along the channel dimension: a's
# channels come first, b's channels are appended after.
#
# a: first input array, length ca*hw
# b: second input array, length cb*hw
# dst: output array, length (ca+cb)*hw, filled in place
# ca: number of channels in a
# cb: number of channels in b
# hw: number of pixels per channel (rows times columns)
function concat_channels(a, b, dst, ca, cb, hw)
    for idx = 1:ca * hw
        dst[idx] = a[idx]
    end
    for idx = 1:cb * hw
        dst[ca * hw + idx] = b[idx]
    end
    return nothing
end

# conv1x1(src, weight, bias, dst, c_in, c_out, hw)
#
# 1-by-1 convolution (a per-pixel dense layer across channels); no
# padding is needed since a 1-by-1 kernel never reads outside its own
# pixel.
#
# src: input array, length c_in*hw, layout (channel, pixel)
# weight: kernel weights, length c_out*c_in
# bias: kernel bias, length c_out
# dst: output array, length c_out*hw, filled in place
# c_in: number of input channels
# c_out: number of output channels
# hw: number of pixels (rows times columns)
function conv1x1(src, weight, bias, dst, c_in, c_out, hw)
    for idx = 1:c_out * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        p = mod(idxm1, hw) + 1
        s = 0.0
        for i_ci = 1:c_in
            xi = (i_ci - 1) * hw + p
            wi = (co - 1) * c_in + i_ci
            s = s + src[xi] * weight[wi]
        end
        dst[idx] = s + bias[co]
    end
    return nothing
end

# unet(x, h, w, c_in, c1, c2, c3, c_out,
#            w_e1a, b_e1a, w_e1b, b_e1b, w_e2a, b_e2a, w_e2b, b_e2b,
#            w_ba, b_ba, w_bb, b_bb,
#            w_d2a, b_d2a, w_d2b, b_d2b, w_d1a, b_d1a, w_d1b, b_d1b,
#            w_out, b_out,
#            xpad0, t_e1, t_e1pad, skip1, p1, p1pad,
#            t_e2, t_e2pad, skip2, p2, p2pad,
#            t_b, t_bpad, bott,
#            u2, cat2, cat2pad, t_d2, t_d2pad, dec2out,
#            u1, cat1, cat1pad, t_d1, t_d1pad, dec1out,
#            y)
#
# A depth-2 U-Net: two encoder double-conv stages (each followed by 2x2
# max pooling), a bottleneck double-conv stage, two decoder double-conv
# stages (each preceded by 2x nearest-neighbor upsampling and a skip-
# connection concatenation), and a final 1-by-1 convolution to c_out
# channels. All convolutions are 3-by-3, stride 1, "same" (except the
# final 1-by-1 output convolution). h and w must both be divisible by
# 4. Every array argument below is allocated by the caller. Every stage
# (zero-fill, zero-pad-and-copy, double convolution, max pool, upsample,
# channel concatenation) is its own kernel, called in sequence below.
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
function unet(x, h, w, c_in, c1, c2, c3, c_out, w_e1a, b_e1a, w_e1b, b_e1b, w_e2a, b_e2a, w_e2b, b_e2b, w_ba, b_ba, w_bb, b_bb, w_d2a, b_d2a, w_d2b, b_d2b, w_d1a, b_d1a, w_d1b, b_d1b, w_out, b_out, xpad0, t_e1, t_e1pad, skip1, p1, p1pad, t_e2, t_e2pad, skip2, p2, p2pad, t_b, t_bpad, bott, u2, cat2, cat2pad, t_d2, t_d2pad, dec2out, u1, cat1, cat1pad, t_d1, t_d1pad, dec1out, y)
    # resolution at each of the three scales
    h2 = div(h, 2)
    w2 = div(w, 2)
    h4 = div(h, 4)
    w4 = div(w, 4)
    # padding width used by every 3-by-3 "same" convolution
    pad = 1
    # channel count at each of the two concatenation points
    c32 = c3 + c2
    c21 = c2 + c1
    # per-channel spatial sizes at each of the three scales, computed
    # once so every zero-fill size below is a bare symbol
    hw = h * w
    hw2 = h2 * w2
    n_xpad0 = c_in * (h + 2 * pad) * (w + 2 * pad)
    n_p1pad = c1 * (h2 + 2 * pad) * (w2 + 2 * pad)
    n_p2pad = c2 * (h4 + 2 * pad) * (w4 + 2 * pad)
    n_cat2pad = c32 * (h2 + 2 * pad) * (w2 + 2 * pad)
    n_cat1pad = c21 * (h + 2 * pad) * (w + 2 * pad)

    # ================= encoder stage 1, at full resolution =================
    zerofill(xpad0, n_xpad0)
    pad_copy(x, xpad0, c_in, h, w, pad)
    double_conv3x3(xpad0, w_e1a, b_e1a, w_e1b, b_e1b, t_e1, t_e1pad, skip1, c_in, c1, h, w, pad)
    maxpool2x2(skip1, p1, c1, h2, w2)

    # ================= encoder stage 2, at half resolution =================
    zerofill(p1pad, n_p1pad)
    pad_copy(p1, p1pad, c1, h2, w2, pad)
    double_conv3x3(p1pad, w_e2a, b_e2a, w_e2b, b_e2b, t_e2, t_e2pad, skip2, c1, c2, h2, w2, pad)
    maxpool2x2(skip2, p2, c2, h4, w4)

    # ===================== bottleneck, at quarter resolution =====================
    zerofill(p2pad, n_p2pad)
    pad_copy(p2, p2pad, c2, h4, w4, pad)
    double_conv3x3(p2pad, w_ba, b_ba, w_bb, b_bb, t_b, t_bpad, bott, c2, c3, h4, w4, pad)

    # ============== decoder stage 2: upsample, concat, double conv ==============
    upsample2x(bott, u2, c3, h4, w4)
    concat_channels(u2, skip2, cat2, c3, c2, hw2)
    zerofill(cat2pad, n_cat2pad)
    pad_copy(cat2, cat2pad, c32, h2, w2, pad)
    double_conv3x3(cat2pad, w_d2a, b_d2a, w_d2b, b_d2b, t_d2, t_d2pad, dec2out, c32, c2, h2, w2, pad)

    # ============== decoder stage 1: upsample, concat, double conv ==============
    upsample2x(dec2out, u1, c2, h2, w2)
    concat_channels(u1, skip1, cat1, c2, c1, hw)
    zerofill(cat1pad, n_cat1pad)
    pad_copy(cat1, cat1pad, c21, h, w, pad)
    double_conv3x3(cat1pad, w_d1a, b_d1a, w_d1b, b_d1b, t_d1, t_d1pad, dec1out, c21, c1, h, w, pad)

    # ===================== output projection: 1-by-1 conv =====================
    conv1x1(dec1out, w_out, b_out, y, c1, c_out, hw)
    return nothing
end