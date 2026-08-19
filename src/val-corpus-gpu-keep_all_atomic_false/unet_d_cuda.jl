import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_unet_d_1!(n_xpad0, xpad0, xpad0d, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_xpad0 - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    xpad0d[i] = 0.0
    xpad0[i] = zero_val
    return nothing
end

function cuda_kernel_unet_d_2!(c_in, hp1, hw, pad, w, wp1, x, xd, xpad0, xpad0d)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c_in * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cid = 0.0
    ci = div(idxm1, hw) + 1
    remd = 0.0
    rem = mod(idxm1, hw)
    id = 0.0
    i = div(rem, w) + 1
    jd = 0.0
    j = mod(rem, w) + 1
    yid = 0.0
    yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
    xpad0d[yi] = xd[idx]
    xpad0[yi] = x[idx]
    return nothing
end

function cuda_kernel_unet_d_3!(b_e1a, b_e1ad, c1, c_in, hp1, hw, kh, khkw, kw, t_e1, t_e1d, w, w_e1a, w_e1ad, wp1, xpad0, xpad0d)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cod = 0.0
    co = div(idxm1, hw) + 1
    remd = 0.0
    rem = mod(idxm1, hw)
    id = 0.0
    i = div(rem, w) + 1
    jd = 0.0
    j = mod(rem, w) + 1
    sd = 0.0
    s = 0.0
    for i_seq_k = 1:c_in * khkw
        kseqm1d = 0.0
        kseqm1 = i_seq_k - 1
        cid = 0.0
        ci = div(kseqm1, khkw) + 1
        rem2d = 0.0
        rem2 = mod(kseqm1, khkw)
        kid = 0.0
        ki = div(rem2, kw) + 1
        kjd = 0.0
        kj = mod(rem2, kw) + 1
        rowd = 0.0
        row = (i + ki) - 1
        cold = 0.0
        col = (j + kj) - 1
        xid = 0.0
        xi = (ci - 1) * hp1 * wp1 + (row - 1) * wp1 + col
        wid = 0.0
        wi = (((co - 1) * c_in + (ci - 1)) * kh + (ki - 1)) * kw + kj
        sd = sd + (w_e1a[wi] * xpad0d[xi] + xpad0[xi] * w_e1ad[wi])
        s = s + xpad0[xi] * w_e1a[wi]
    end
    t_e1d[idx] = sd + b_e1ad[co]
    t_e1[idx] = s + b_e1a[co]
    return nothing
end

function cuda_kernel_unet_d_4!(n_e1_mid, t_e1, t_e1d, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e1_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e1d[i] = (0.5 * (1.0 + sign(t_e1[i] - zero_val))) * t_e1d[i]
    t_e1[i] = max(t_e1[i], zero_val)
    return nothing
end

function cuda_kernel_unet_d_5!(n_e1_midpad, t_e1pad, t_e1padd, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e1_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e1padd[i] = 0.0
    t_e1pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_d_6!(c1, hp1, hw, pad, t_e1, t_e1d, t_e1pad, t_e1padd, w, wp1)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cid = 0.0
    ci = div(idxm1, hw) + 1
    remd = 0.0
    rem = mod(idxm1, hw)
    id = 0.0
    i = div(rem, w) + 1
    jd = 0.0
    j = mod(rem, w) + 1
    yid = 0.0
    yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
    t_e1padd[yi] = t_e1d[idx]
    t_e1pad[yi] = t_e1[idx]
    return nothing
end

function cuda_kernel_unet_d_7!(b_e1b, b_e1bd, c1, hp1, hw, kh, khkw, kw, skip1, skip1d, t_e1pad, t_e1padd, w, w_e1b, w_e1bd, wp1)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cod = 0.0
    co = div(idxm1, hw) + 1
    remd = 0.0
    rem = mod(idxm1, hw)
    id = 0.0
    i = div(rem, w) + 1
    jd = 0.0
    j = mod(rem, w) + 1
    sd = 0.0
    s = 0.0
    for i_seq_k = 1:c1 * khkw
        kseqm1d = 0.0
        kseqm1 = i_seq_k - 1
        cid = 0.0
        ci = div(kseqm1, khkw) + 1
        rem2d = 0.0
        rem2 = mod(kseqm1, khkw)
        kid = 0.0
        ki = div(rem2, kw) + 1
        kjd = 0.0
        kj = mod(rem2, kw) + 1
        rowd = 0.0
        row = (i + ki) - 1
        cold = 0.0
        col = (j + kj) - 1
        xid = 0.0
        xi = (ci - 1) * hp1 * wp1 + (row - 1) * wp1 + col
        wid = 0.0
        wi = (((co - 1) * c1 + (ci - 1)) * kh + (ki - 1)) * kw + kj
        sd = sd + (w_e1b[wi] * t_e1padd[xi] + t_e1pad[xi] * w_e1bd[wi])
        s = s + t_e1pad[xi] * w_e1b[wi]
    end
    skip1d[idx] = sd + b_e1bd[co]
    skip1[idx] = s + b_e1b[co]
    return nothing
end

function cuda_kernel_unet_d_8!(n_e1_out, skip1, skip1d, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e1_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    skip1d[i] = (0.5 * (1.0 + sign(skip1[i] - zero_val))) * skip1d[i]
    skip1[i] = max(skip1[i], zero_val)
    return nothing
end

function cuda_kernel_unet_d_9!(c1, hw, hw2, p1, p1d, skip1, skip1d, w, w2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cid = 0.0
    ci = div(idxm1, hw2) + 1
    remd = 0.0
    rem = mod(idxm1, hw2)
    oid = 0.0
    oi = div(rem, w2) + 1
    ojd = 0.0
    oj = mod(rem, w2) + 1
    i0d = 0.0
    i0 = 2oi - 1
    j0d = 0.0
    j0 = 2oj - 1
    a11d = skip1d[(ci - 1) * hw + (i0 - 1) * w + j0]
    a11 = skip1[(ci - 1) * hw + (i0 - 1) * w + j0]
    a12d = skip1d[(ci - 1) * hw + (i0 - 1) * w + j0 + 1]
    a12 = skip1[(ci - 1) * hw + (i0 - 1) * w + j0 + 1]
    a21d = skip1d[(ci - 1) * hw + i0 * w + j0]
    a21 = skip1[(ci - 1) * hw + i0 * w + j0]
    a22d = skip1d[(ci - 1) * hw + i0 * w + j0 + 1]
    a22 = skip1[(ci - 1) * hw + i0 * w + j0 + 1]
    m1d = (0.5 * (1.0 + sign(a11 - a12))) * a11d + (0.5 * (1.0 + sign(a12 - a11))) * a12d
    m1 = max(a11, a12)
    m2d = (0.5 * (1.0 + sign(a21 - a22))) * a21d + (0.5 * (1.0 + sign(a22 - a21))) * a22d
    m2 = max(a21, a22)
    p1d[idx] = (0.5 * (1.0 + sign(m1 - m2))) * m1d + (0.5 * (1.0 + sign(m2 - m1))) * m2d
    p1[idx] = max(m1, m2)
    return nothing
end

function cuda_kernel_unet_d_10!(n_p1pad, p1pad, p1padd, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_p1pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    p1padd[i] = 0.0
    p1pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_d_11!(c1, hp2, hw2, p1, p1d, p1pad, p1padd, pad, w2, wp2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cid = 0.0
    ci = div(idxm1, hw2) + 1
    remd = 0.0
    rem = mod(idxm1, hw2)
    id = 0.0
    i = div(rem, w2) + 1
    jd = 0.0
    j = mod(rem, w2) + 1
    yid = 0.0
    yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
    p1padd[yi] = p1d[idx]
    p1pad[yi] = p1[idx]
    return nothing
end

function cuda_kernel_unet_d_12!(b_e2a, b_e2ad, c1, c2, hp2, hw2, kh, khkw, kw, p1pad, p1padd, t_e2, t_e2d, w2, w_e2a, w_e2ad, wp2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cod = 0.0
    co = div(idxm1, hw2) + 1
    remd = 0.0
    rem = mod(idxm1, hw2)
    id = 0.0
    i = div(rem, w2) + 1
    jd = 0.0
    j = mod(rem, w2) + 1
    sd = 0.0
    s = 0.0
    for i_seq_k = 1:c1 * khkw
        kseqm1d = 0.0
        kseqm1 = i_seq_k - 1
        cid = 0.0
        ci = div(kseqm1, khkw) + 1
        rem2d = 0.0
        rem2 = mod(kseqm1, khkw)
        kid = 0.0
        ki = div(rem2, kw) + 1
        kjd = 0.0
        kj = mod(rem2, kw) + 1
        rowd = 0.0
        row = (i + ki) - 1
        cold = 0.0
        col = (j + kj) - 1
        xid = 0.0
        xi = (ci - 1) * hp2 * wp2 + (row - 1) * wp2 + col
        wid = 0.0
        wi = (((co - 1) * c1 + (ci - 1)) * kh + (ki - 1)) * kw + kj
        sd = sd + (w_e2a[wi] * p1padd[xi] + p1pad[xi] * w_e2ad[wi])
        s = s + p1pad[xi] * w_e2a[wi]
    end
    t_e2d[idx] = sd + b_e2ad[co]
    t_e2[idx] = s + b_e2a[co]
    return nothing
end

function cuda_kernel_unet_d_13!(n_e2_mid, t_e2, t_e2d, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e2_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e2d[i] = (0.5 * (1.0 + sign(t_e2[i] - zero_val))) * t_e2d[i]
    t_e2[i] = max(t_e2[i], zero_val)
    return nothing
end

function cuda_kernel_unet_d_14!(n_e2_midpad, t_e2pad, t_e2padd, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e2_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e2padd[i] = 0.0
    t_e2pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_d_15!(c2, hp2, hw2, pad, t_e2, t_e2d, t_e2pad, t_e2padd, w2, wp2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cid = 0.0
    ci = div(idxm1, hw2) + 1
    remd = 0.0
    rem = mod(idxm1, hw2)
    id = 0.0
    i = div(rem, w2) + 1
    jd = 0.0
    j = mod(rem, w2) + 1
    yid = 0.0
    yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
    t_e2padd[yi] = t_e2d[idx]
    t_e2pad[yi] = t_e2[idx]
    return nothing
end

function cuda_kernel_unet_d_16!(b_e2b, b_e2bd, c2, hp2, hw2, kh, khkw, kw, skip2, skip2d, t_e2pad, t_e2padd, w2, w_e2b, w_e2bd, wp2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cod = 0.0
    co = div(idxm1, hw2) + 1
    remd = 0.0
    rem = mod(idxm1, hw2)
    id = 0.0
    i = div(rem, w2) + 1
    jd = 0.0
    j = mod(rem, w2) + 1
    sd = 0.0
    s = 0.0
    for i_seq_k = 1:c2 * khkw
        kseqm1d = 0.0
        kseqm1 = i_seq_k - 1
        cid = 0.0
        ci = div(kseqm1, khkw) + 1
        rem2d = 0.0
        rem2 = mod(kseqm1, khkw)
        kid = 0.0
        ki = div(rem2, kw) + 1
        kjd = 0.0
        kj = mod(rem2, kw) + 1
        rowd = 0.0
        row = (i + ki) - 1
        cold = 0.0
        col = (j + kj) - 1
        xid = 0.0
        xi = (ci - 1) * hp2 * wp2 + (row - 1) * wp2 + col
        wid = 0.0
        wi = (((co - 1) * c2 + (ci - 1)) * kh + (ki - 1)) * kw + kj
        sd = sd + (w_e2b[wi] * t_e2padd[xi] + t_e2pad[xi] * w_e2bd[wi])
        s = s + t_e2pad[xi] * w_e2b[wi]
    end
    skip2d[idx] = sd + b_e2bd[co]
    skip2[idx] = s + b_e2b[co]
    return nothing
end

function cuda_kernel_unet_d_17!(n_e2_out, skip2, skip2d, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e2_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    skip2d[i] = (0.5 * (1.0 + sign(skip2[i] - zero_val))) * skip2d[i]
    skip2[i] = max(skip2[i], zero_val)
    return nothing
end

function cuda_kernel_unet_d_18!(c2, hw2, hw4, p2, p2d, skip2, skip2d, w2, w4)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw4 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cid = 0.0
    ci = div(idxm1, hw4) + 1
    remd = 0.0
    rem = mod(idxm1, hw4)
    oid = 0.0
    oi = div(rem, w4) + 1
    ojd = 0.0
    oj = mod(rem, w4) + 1
    i0d = 0.0
    i0 = 2oi - 1
    j0d = 0.0
    j0 = 2oj - 1
    a11d = skip2d[(ci - 1) * hw2 + (i0 - 1) * w2 + j0]
    a11 = skip2[(ci - 1) * hw2 + (i0 - 1) * w2 + j0]
    a12d = skip2d[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1]
    a12 = skip2[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1]
    a21d = skip2d[(ci - 1) * hw2 + i0 * w2 + j0]
    a21 = skip2[(ci - 1) * hw2 + i0 * w2 + j0]
    a22d = skip2d[(ci - 1) * hw2 + i0 * w2 + j0 + 1]
    a22 = skip2[(ci - 1) * hw2 + i0 * w2 + j0 + 1]
    m1d = (0.5 * (1.0 + sign(a11 - a12))) * a11d + (0.5 * (1.0 + sign(a12 - a11))) * a12d
    m1 = max(a11, a12)
    m2d = (0.5 * (1.0 + sign(a21 - a22))) * a21d + (0.5 * (1.0 + sign(a22 - a21))) * a22d
    m2 = max(a21, a22)
    p2d[idx] = (0.5 * (1.0 + sign(m1 - m2))) * m1d + (0.5 * (1.0 + sign(m2 - m1))) * m2d
    p2[idx] = max(m1, m2)
    return nothing
end

function cuda_kernel_unet_d_19!(n_p2pad, p2pad, p2padd, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_p2pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    p2padd[i] = 0.0
    p2pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_d_20!(c2, hp4, hw4, p2, p2d, p2pad, p2padd, pad, w4, wp4)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw4 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cid = 0.0
    ci = div(idxm1, hw4) + 1
    remd = 0.0
    rem = mod(idxm1, hw4)
    id = 0.0
    i = div(rem, w4) + 1
    jd = 0.0
    j = mod(rem, w4) + 1
    yid = 0.0
    yi = (ci - 1) * hp4 * wp4 + ((i + pad) - 1) * wp4 + (j + pad)
    p2padd[yi] = p2d[idx]
    p2pad[yi] = p2[idx]
    return nothing
end

function cuda_kernel_unet_d_21!(b_ba, b_bad, c2, c3, hp4, hw4, kh, khkw, kw, p2pad, p2padd, t_b, t_bd, w4, w_ba, w_bad, wp4)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c3 * hw4 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cod = 0.0
    co = div(idxm1, hw4) + 1
    remd = 0.0
    rem = mod(idxm1, hw4)
    id = 0.0
    i = div(rem, w4) + 1
    jd = 0.0
    j = mod(rem, w4) + 1
    sd = 0.0
    s = 0.0
    for i_seq_k = 1:c2 * khkw
        kseqm1d = 0.0
        kseqm1 = i_seq_k - 1
        cid = 0.0
        ci = div(kseqm1, khkw) + 1
        rem2d = 0.0
        rem2 = mod(kseqm1, khkw)
        kid = 0.0
        ki = div(rem2, kw) + 1
        kjd = 0.0
        kj = mod(rem2, kw) + 1
        rowd = 0.0
        row = (i + ki) - 1
        cold = 0.0
        col = (j + kj) - 1
        xid = 0.0
        xi = (ci - 1) * hp4 * wp4 + (row - 1) * wp4 + col
        wid = 0.0
        wi = (((co - 1) * c2 + (ci - 1)) * kh + (ki - 1)) * kw + kj
        sd = sd + (w_ba[wi] * p2padd[xi] + p2pad[xi] * w_bad[wi])
        s = s + p2pad[xi] * w_ba[wi]
    end
    t_bd[idx] = sd + b_bad[co]
    t_b[idx] = s + b_ba[co]
    return nothing
end

function cuda_kernel_unet_d_22!(n_b_mid, t_b, t_bd, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_b_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_bd[i] = (0.5 * (1.0 + sign(t_b[i] - zero_val))) * t_bd[i]
    t_b[i] = max(t_b[i], zero_val)
    return nothing
end

function cuda_kernel_unet_d_23!(n_b_midpad, t_bpad, t_bpadd, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_b_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_bpadd[i] = 0.0
    t_bpad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_d_24!(c3, hp4, hw4, pad, t_b, t_bd, t_bpad, t_bpadd, w4, wp4)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c3 * hw4 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cid = 0.0
    ci = div(idxm1, hw4) + 1
    remd = 0.0
    rem = mod(idxm1, hw4)
    id = 0.0
    i = div(rem, w4) + 1
    jd = 0.0
    j = mod(rem, w4) + 1
    yid = 0.0
    yi = (ci - 1) * hp4 * wp4 + ((i + pad) - 1) * wp4 + (j + pad)
    t_bpadd[yi] = t_bd[idx]
    t_bpad[yi] = t_b[idx]
    return nothing
end

function cuda_kernel_unet_d_25!(b_bb, b_bbd, bott, bottd, c3, hp4, hw4, kh, khkw, kw, t_bpad, t_bpadd, w4, w_bb, w_bbd, wp4)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c3 * hw4 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cod = 0.0
    co = div(idxm1, hw4) + 1
    remd = 0.0
    rem = mod(idxm1, hw4)
    id = 0.0
    i = div(rem, w4) + 1
    jd = 0.0
    j = mod(rem, w4) + 1
    sd = 0.0
    s = 0.0
    for i_seq_k = 1:c3 * khkw
        kseqm1d = 0.0
        kseqm1 = i_seq_k - 1
        cid = 0.0
        ci = div(kseqm1, khkw) + 1
        rem2d = 0.0
        rem2 = mod(kseqm1, khkw)
        kid = 0.0
        ki = div(rem2, kw) + 1
        kjd = 0.0
        kj = mod(rem2, kw) + 1
        rowd = 0.0
        row = (i + ki) - 1
        cold = 0.0
        col = (j + kj) - 1
        xid = 0.0
        xi = (ci - 1) * hp4 * wp4 + (row - 1) * wp4 + col
        wid = 0.0
        wi = (((co - 1) * c3 + (ci - 1)) * kh + (ki - 1)) * kw + kj
        sd = sd + (w_bb[wi] * t_bpadd[xi] + t_bpad[xi] * w_bbd[wi])
        s = s + t_bpad[xi] * w_bb[wi]
    end
    bottd[idx] = sd + b_bbd[co]
    bott[idx] = s + b_bb[co]
    return nothing
end

function cuda_kernel_unet_d_26!(bott, bottd, n_b_out, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_b_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    bottd[i] = (0.5 * (1.0 + sign(bott[i] - zero_val))) * bottd[i]
    bott[i] = max(bott[i], zero_val)
    return nothing
end

function cuda_kernel_unet_d_27!(bott, bottd, c3, hw2, hw4, scale, u2, u2d, w2, w4)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c3 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cid = 0.0
    ci = div(idxm1, hw2) + 1
    remd = 0.0
    rem = mod(idxm1, hw2)
    oid = 0.0
    oi = div(rem, w2) + 1
    ojd = 0.0
    oj = mod(rem, w2) + 1
    oim1d = 0.0
    oim1 = oi - 1
    ojm1d = 0.0
    ojm1 = oj - 1
    id = 0.0
    i = div(oim1, scale) + 1
    jd = 0.0
    j = div(ojm1, scale) + 1
    xid = 0.0
    xi = (ci - 1) * hw4 + (i - 1) * w4 + j
    u2d[idx] = bottd[xi]
    u2[idx] = bott[xi]
    return nothing
end

function cuda_kernel_unet_d_28!(c3, cat2, cat2d, hw2, u2, u2d)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c3 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    cat2d[idx] = u2d[idx]
    cat2[idx] = u2[idx]
    return nothing
end

function cuda_kernel_unet_d_29!(c2, c3, cat2, cat2d, hw2, skip2, skip2d)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    cat2d[c3 * hw2 + idx] = skip2d[idx]
    cat2[c3 * hw2 + idx] = skip2[idx]
    return nothing
end

function cuda_kernel_unet_d_30!(cat2pad, cat2padd, n_cat2pad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_cat2pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    cat2padd[i] = 0.0
    cat2pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_d_31!(c32, cat2, cat2d, cat2pad, cat2padd, hp2, hw2, pad, w2, wp2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c32 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cid = 0.0
    ci = div(idxm1, hw2) + 1
    remd = 0.0
    rem = mod(idxm1, hw2)
    id = 0.0
    i = div(rem, w2) + 1
    jd = 0.0
    j = mod(rem, w2) + 1
    yid = 0.0
    yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
    cat2padd[yi] = cat2d[idx]
    cat2pad[yi] = cat2[idx]
    return nothing
end

function cuda_kernel_unet_d_32!(b_d2a, b_d2ad, c2, c32, cat2pad, cat2padd, hp2, hw2, kh, khkw, kw, t_d2, t_d2d, w2, w_d2a, w_d2ad, wp2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cod = 0.0
    co = div(idxm1, hw2) + 1
    remd = 0.0
    rem = mod(idxm1, hw2)
    id = 0.0
    i = div(rem, w2) + 1
    jd = 0.0
    j = mod(rem, w2) + 1
    sd = 0.0
    s = 0.0
    for i_seq_k = 1:c32 * khkw
        kseqm1d = 0.0
        kseqm1 = i_seq_k - 1
        cid = 0.0
        ci = div(kseqm1, khkw) + 1
        rem2d = 0.0
        rem2 = mod(kseqm1, khkw)
        kid = 0.0
        ki = div(rem2, kw) + 1
        kjd = 0.0
        kj = mod(rem2, kw) + 1
        rowd = 0.0
        row = (i + ki) - 1
        cold = 0.0
        col = (j + kj) - 1
        xid = 0.0
        xi = (ci - 1) * hp2 * wp2 + (row - 1) * wp2 + col
        wid = 0.0
        wi = (((co - 1) * c32 + (ci - 1)) * kh + (ki - 1)) * kw + kj
        sd = sd + (w_d2a[wi] * cat2padd[xi] + cat2pad[xi] * w_d2ad[wi])
        s = s + cat2pad[xi] * w_d2a[wi]
    end
    t_d2d[idx] = sd + b_d2ad[co]
    t_d2[idx] = s + b_d2a[co]
    return nothing
end

function cuda_kernel_unet_d_33!(n_d2_mid, t_d2, t_d2d, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d2_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d2d[i] = (0.5 * (1.0 + sign(t_d2[i] - zero_val))) * t_d2d[i]
    t_d2[i] = max(t_d2[i], zero_val)
    return nothing
end

function cuda_kernel_unet_d_34!(n_d2_midpad, t_d2pad, t_d2padd, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d2_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d2padd[i] = 0.0
    t_d2pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_d_35!(c2, hp2, hw2, pad, t_d2, t_d2d, t_d2pad, t_d2padd, w2, wp2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cid = 0.0
    ci = div(idxm1, hw2) + 1
    remd = 0.0
    rem = mod(idxm1, hw2)
    id = 0.0
    i = div(rem, w2) + 1
    jd = 0.0
    j = mod(rem, w2) + 1
    yid = 0.0
    yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
    t_d2padd[yi] = t_d2d[idx]
    t_d2pad[yi] = t_d2[idx]
    return nothing
end

function cuda_kernel_unet_d_36!(b_d2b, b_d2bd, c2, dec2out, dec2outd, hp2, hw2, kh, khkw, kw, t_d2pad, t_d2padd, w2, w_d2b, w_d2bd, wp2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cod = 0.0
    co = div(idxm1, hw2) + 1
    remd = 0.0
    rem = mod(idxm1, hw2)
    id = 0.0
    i = div(rem, w2) + 1
    jd = 0.0
    j = mod(rem, w2) + 1
    sd = 0.0
    s = 0.0
    for i_seq_k = 1:c2 * khkw
        kseqm1d = 0.0
        kseqm1 = i_seq_k - 1
        cid = 0.0
        ci = div(kseqm1, khkw) + 1
        rem2d = 0.0
        rem2 = mod(kseqm1, khkw)
        kid = 0.0
        ki = div(rem2, kw) + 1
        kjd = 0.0
        kj = mod(rem2, kw) + 1
        rowd = 0.0
        row = (i + ki) - 1
        cold = 0.0
        col = (j + kj) - 1
        xid = 0.0
        xi = (ci - 1) * hp2 * wp2 + (row - 1) * wp2 + col
        wid = 0.0
        wi = (((co - 1) * c2 + (ci - 1)) * kh + (ki - 1)) * kw + kj
        sd = sd + (w_d2b[wi] * t_d2padd[xi] + t_d2pad[xi] * w_d2bd[wi])
        s = s + t_d2pad[xi] * w_d2b[wi]
    end
    dec2outd[idx] = sd + b_d2bd[co]
    dec2out[idx] = s + b_d2b[co]
    return nothing
end

function cuda_kernel_unet_d_37!(dec2out, dec2outd, n_d2_out, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d2_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    dec2outd[i] = (0.5 * (1.0 + sign(dec2out[i] - zero_val))) * dec2outd[i]
    dec2out[i] = max(dec2out[i], zero_val)
    return nothing
end

function cuda_kernel_unet_d_38!(c2, dec2out, dec2outd, hw, hw2, scale, u1, u1d, w, w2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cid = 0.0
    ci = div(idxm1, hw) + 1
    remd = 0.0
    rem = mod(idxm1, hw)
    oid = 0.0
    oi = div(rem, w) + 1
    ojd = 0.0
    oj = mod(rem, w) + 1
    oim1d = 0.0
    oim1 = oi - 1
    ojm1d = 0.0
    ojm1 = oj - 1
    id = 0.0
    i = div(oim1, scale) + 1
    jd = 0.0
    j = div(ojm1, scale) + 1
    xid = 0.0
    xi = (ci - 1) * hw2 + (i - 1) * w2 + j
    u1d[idx] = dec2outd[xi]
    u1[idx] = dec2out[xi]
    return nothing
end

function cuda_kernel_unet_d_39!(c2, cat1, cat1d, hw, u1, u1d)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    cat1d[idx] = u1d[idx]
    cat1[idx] = u1[idx]
    return nothing
end

function cuda_kernel_unet_d_40!(c1, c2, cat1, cat1d, hw, skip1, skip1d)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    cat1d[c2 * hw + idx] = skip1d[idx]
    cat1[c2 * hw + idx] = skip1[idx]
    return nothing
end

function cuda_kernel_unet_d_41!(cat1pad, cat1padd, n_cat1pad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_cat1pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    cat1padd[i] = 0.0
    cat1pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_d_42!(c21, cat1, cat1d, cat1pad, cat1padd, hp1, hw, pad, w, wp1)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c21 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cid = 0.0
    ci = div(idxm1, hw) + 1
    remd = 0.0
    rem = mod(idxm1, hw)
    id = 0.0
    i = div(rem, w) + 1
    jd = 0.0
    j = mod(rem, w) + 1
    yid = 0.0
    yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
    cat1padd[yi] = cat1d[idx]
    cat1pad[yi] = cat1[idx]
    return nothing
end

function cuda_kernel_unet_d_43!(b_d1a, b_d1ad, c1, c21, cat1pad, cat1padd, hp1, hw, kh, khkw, kw, t_d1, t_d1d, w, w_d1a, w_d1ad, wp1)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cod = 0.0
    co = div(idxm1, hw) + 1
    remd = 0.0
    rem = mod(idxm1, hw)
    id = 0.0
    i = div(rem, w) + 1
    jd = 0.0
    j = mod(rem, w) + 1
    sd = 0.0
    s = 0.0
    for i_seq_k = 1:c21 * khkw
        kseqm1d = 0.0
        kseqm1 = i_seq_k - 1
        cid = 0.0
        ci = div(kseqm1, khkw) + 1
        rem2d = 0.0
        rem2 = mod(kseqm1, khkw)
        kid = 0.0
        ki = div(rem2, kw) + 1
        kjd = 0.0
        kj = mod(rem2, kw) + 1
        rowd = 0.0
        row = (i + ki) - 1
        cold = 0.0
        col = (j + kj) - 1
        xid = 0.0
        xi = (ci - 1) * hp1 * wp1 + (row - 1) * wp1 + col
        wid = 0.0
        wi = (((co - 1) * c21 + (ci - 1)) * kh + (ki - 1)) * kw + kj
        sd = sd + (w_d1a[wi] * cat1padd[xi] + cat1pad[xi] * w_d1ad[wi])
        s = s + cat1pad[xi] * w_d1a[wi]
    end
    t_d1d[idx] = sd + b_d1ad[co]
    t_d1[idx] = s + b_d1a[co]
    return nothing
end

function cuda_kernel_unet_d_44!(n_d1_mid, t_d1, t_d1d, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d1_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d1d[i] = (0.5 * (1.0 + sign(t_d1[i] - zero_val))) * t_d1d[i]
    t_d1[i] = max(t_d1[i], zero_val)
    return nothing
end

function cuda_kernel_unet_d_45!(n_d1_midpad, t_d1pad, t_d1padd, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d1_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d1padd[i] = 0.0
    t_d1pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_d_46!(c1, hp1, hw, pad, t_d1, t_d1d, t_d1pad, t_d1padd, w, wp1)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cid = 0.0
    ci = div(idxm1, hw) + 1
    remd = 0.0
    rem = mod(idxm1, hw)
    id = 0.0
    i = div(rem, w) + 1
    jd = 0.0
    j = mod(rem, w) + 1
    yid = 0.0
    yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
    t_d1padd[yi] = t_d1d[idx]
    t_d1pad[yi] = t_d1[idx]
    return nothing
end

function cuda_kernel_unet_d_47!(b_d1b, b_d1bd, c1, dec1out, dec1outd, hp1, hw, kh, khkw, kw, t_d1pad, t_d1padd, w, w_d1b, w_d1bd, wp1)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cod = 0.0
    co = div(idxm1, hw) + 1
    remd = 0.0
    rem = mod(idxm1, hw)
    id = 0.0
    i = div(rem, w) + 1
    jd = 0.0
    j = mod(rem, w) + 1
    sd = 0.0
    s = 0.0
    for i_seq_k = 1:c1 * khkw
        kseqm1d = 0.0
        kseqm1 = i_seq_k - 1
        cid = 0.0
        ci = div(kseqm1, khkw) + 1
        rem2d = 0.0
        rem2 = mod(kseqm1, khkw)
        kid = 0.0
        ki = div(rem2, kw) + 1
        kjd = 0.0
        kj = mod(rem2, kw) + 1
        rowd = 0.0
        row = (i + ki) - 1
        cold = 0.0
        col = (j + kj) - 1
        xid = 0.0
        xi = (ci - 1) * hp1 * wp1 + (row - 1) * wp1 + col
        wid = 0.0
        wi = (((co - 1) * c1 + (ci - 1)) * kh + (ki - 1)) * kw + kj
        sd = sd + (w_d1b[wi] * t_d1padd[xi] + t_d1pad[xi] * w_d1bd[wi])
        s = s + t_d1pad[xi] * w_d1b[wi]
    end
    dec1outd[idx] = sd + b_d1bd[co]
    dec1out[idx] = s + b_d1b[co]
    return nothing
end

function cuda_kernel_unet_d_48!(dec1out, dec1outd, n_d1_out, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d1_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    dec1outd[i] = (0.5 * (1.0 + sign(dec1out[i] - zero_val))) * dec1outd[i]
    dec1out[i] = max(dec1out[i], zero_val)
    return nothing
end

function cuda_kernel_unet_d_49!(b_out, b_outd, c1, c_out, dec1out, dec1outd, hw, kh_out, khkw_out, kw_out, w, w_out, w_outd, y, yd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c_out * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1d = 0.0
    idxm1 = idx - 1
    cod = 0.0
    co = div(idxm1, hw) + 1
    remd = 0.0
    rem = mod(idxm1, hw)
    id = 0.0
    i = div(rem, w) + 1
    jd = 0.0
    j = mod(rem, w) + 1
    sd = 0.0
    s = 0.0
    for i_seq_k = 1:c1 * khkw_out
        kseqm1d = 0.0
        kseqm1 = i_seq_k - 1
        cid = 0.0
        ci = div(kseqm1, khkw_out) + 1
        rem2d = 0.0
        rem2 = mod(kseqm1, khkw_out)
        kid = 0.0
        ki = div(rem2, kw_out) + 1
        kjd = 0.0
        kj = mod(rem2, kw_out) + 1
        rowd = 0.0
        row = (i + ki) - 1
        cold = 0.0
        col = (j + kj) - 1
        xid = 0.0
        xi = (ci - 1) * hw + (row - 1) * w + col
        wid = 0.0
        wi = (((co - 1) * c1 + (ci - 1)) * kh_out + (ki - 1)) * kw_out + kj
        sd = sd + (w_out[wi] * dec1outd[xi] + dec1out[xi] * w_outd[wi])
        s = s + dec1out[xi] * w_out[wi]
    end
    yd[idx] = sd + b_outd[co]
    y[idx] = s + b_out[co]
    return nothing
end

function cuda_kernel_unet_1!(n_xpad0, xpad0, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_xpad0 - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    xpad0[i] = zero_val
    return nothing
end

function cuda_kernel_unet_2!(c_in, hp1, hw, pad, w, wp1, x, xpad0)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c_in * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw) + 1
    rem = mod(idxm1, hw)
    i = div(rem, w) + 1
    j = mod(rem, w) + 1
    yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
    xpad0[yi] = x[idx]
    return nothing
end

function cuda_kernel_unet_3!(b_e1a, c1, c_in, hp1, hw, kh, khkw, kw, t_e1, w, w_e1a, wp1, xpad0)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
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
    return nothing
end

function cuda_kernel_unet_4!(n_e1_mid, t_e1, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e1_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e1[i] = max(t_e1[i], zero_val)
    return nothing
end

function cuda_kernel_unet_5!(n_e1_midpad, t_e1pad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e1_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e1pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_6!(c1, hp1, hw, pad, t_e1, t_e1pad, w, wp1)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw) + 1
    rem = mod(idxm1, hw)
    i = div(rem, w) + 1
    j = mod(rem, w) + 1
    yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
    t_e1pad[yi] = t_e1[idx]
    return nothing
end

function cuda_kernel_unet_7!(b_e1b, c1, hp1, hw, kh, khkw, kw, skip1, t_e1pad, w, w_e1b, wp1)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
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
    return nothing
end

function cuda_kernel_unet_8!(n_e1_out, skip1, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e1_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    skip1[i] = max(skip1[i], zero_val)
    return nothing
end

function cuda_kernel_unet_9!(c1, hw, hw2, p1, skip1, w, w2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
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
    return nothing
end

function cuda_kernel_unet_10!(n_p1pad, p1pad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_p1pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    p1pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_11!(c1, hp2, hw2, p1, p1pad, pad, w2, wp2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw2) + 1
    rem = mod(idxm1, hw2)
    i = div(rem, w2) + 1
    j = mod(rem, w2) + 1
    yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
    p1pad[yi] = p1[idx]
    return nothing
end

function cuda_kernel_unet_12!(b_e2a, c1, c2, hp2, hw2, kh, khkw, kw, p1pad, t_e2, w2, w_e2a, wp2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
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
    return nothing
end

function cuda_kernel_unet_13!(n_e2_mid, t_e2, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e2_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e2[i] = max(t_e2[i], zero_val)
    return nothing
end

function cuda_kernel_unet_14!(n_e2_midpad, t_e2pad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e2_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e2pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_15!(c2, hp2, hw2, pad, t_e2, t_e2pad, w2, wp2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw2) + 1
    rem = mod(idxm1, hw2)
    i = div(rem, w2) + 1
    j = mod(rem, w2) + 1
    yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
    t_e2pad[yi] = t_e2[idx]
    return nothing
end

function cuda_kernel_unet_16!(b_e2b, c2, hp2, hw2, kh, khkw, kw, skip2, t_e2pad, w2, w_e2b, wp2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
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
    return nothing
end

function cuda_kernel_unet_17!(n_e2_out, skip2, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e2_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    skip2[i] = max(skip2[i], zero_val)
    return nothing
end

function cuda_kernel_unet_18!(c2, hw2, hw4, p2, skip2, w2, w4)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw4 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
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
    return nothing
end

function cuda_kernel_unet_19!(n_p2pad, p2pad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_p2pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    p2pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_20!(c2, hp4, hw4, p2, p2pad, pad, w4, wp4)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw4 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw4) + 1
    rem = mod(idxm1, hw4)
    i = div(rem, w4) + 1
    j = mod(rem, w4) + 1
    yi = (ci - 1) * hp4 * wp4 + ((i + pad) - 1) * wp4 + (j + pad)
    p2pad[yi] = p2[idx]
    return nothing
end

function cuda_kernel_unet_21!(b_ba, c2, c3, hp4, hw4, kh, khkw, kw, p2pad, t_b, w4, w_ba, wp4)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c3 * hw4 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
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
    return nothing
end

function cuda_kernel_unet_22!(n_b_mid, t_b, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_b_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_b[i] = max(t_b[i], zero_val)
    return nothing
end

function cuda_kernel_unet_23!(n_b_midpad, t_bpad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_b_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_bpad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_24!(c3, hp4, hw4, pad, t_b, t_bpad, w4, wp4)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c3 * hw4 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw4) + 1
    rem = mod(idxm1, hw4)
    i = div(rem, w4) + 1
    j = mod(rem, w4) + 1
    yi = (ci - 1) * hp4 * wp4 + ((i + pad) - 1) * wp4 + (j + pad)
    t_bpad[yi] = t_b[idx]
    return nothing
end

function cuda_kernel_unet_25!(b_bb, bott, c3, hp4, hw4, kh, khkw, kw, t_bpad, w4, w_bb, wp4)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c3 * hw4 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
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
    return nothing
end

function cuda_kernel_unet_26!(bott, n_b_out, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_b_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    bott[i] = max(bott[i], zero_val)
    return nothing
end

function cuda_kernel_unet_27!(bott, c3, hw2, hw4, scale, u2, w2, w4)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c3 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
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
    return nothing
end

function cuda_kernel_unet_28!(c3, cat2, hw2, u2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c3 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    cat2[idx] = u2[idx]
    return nothing
end

function cuda_kernel_unet_29!(c2, c3, cat2, hw2, skip2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    cat2[c3 * hw2 + idx] = skip2[idx]
    return nothing
end

function cuda_kernel_unet_30!(cat2pad, n_cat2pad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_cat2pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    cat2pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_31!(c32, cat2, cat2pad, hp2, hw2, pad, w2, wp2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c32 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw2) + 1
    rem = mod(idxm1, hw2)
    i = div(rem, w2) + 1
    j = mod(rem, w2) + 1
    yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
    cat2pad[yi] = cat2[idx]
    return nothing
end

function cuda_kernel_unet_32!(b_d2a, c2, c32, cat2pad, hp2, hw2, kh, khkw, kw, t_d2, w2, w_d2a, wp2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
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
    return nothing
end

function cuda_kernel_unet_33!(n_d2_mid, t_d2, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d2_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d2[i] = max(t_d2[i], zero_val)
    return nothing
end

function cuda_kernel_unet_34!(n_d2_midpad, t_d2pad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d2_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d2pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_35!(c2, hp2, hw2, pad, t_d2, t_d2pad, w2, wp2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw2) + 1
    rem = mod(idxm1, hw2)
    i = div(rem, w2) + 1
    j = mod(rem, w2) + 1
    yi = (ci - 1) * hp2 * wp2 + ((i + pad) - 1) * wp2 + (j + pad)
    t_d2pad[yi] = t_d2[idx]
    return nothing
end

function cuda_kernel_unet_36!(b_d2b, c2, dec2out, hp2, hw2, kh, khkw, kw, t_d2pad, w2, w_d2b, wp2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
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
    return nothing
end

function cuda_kernel_unet_37!(dec2out, n_d2_out, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d2_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    dec2out[i] = max(dec2out[i], zero_val)
    return nothing
end

function cuda_kernel_unet_38!(c2, dec2out, hw, hw2, scale, u1, w, w2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
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
    return nothing
end

function cuda_kernel_unet_39!(c2, cat1, hw, u1)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    cat1[idx] = u1[idx]
    return nothing
end

function cuda_kernel_unet_40!(c1, c2, cat1, hw, skip1)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    cat1[c2 * hw + idx] = skip1[idx]
    return nothing
end

function cuda_kernel_unet_41!(cat1pad, n_cat1pad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_cat1pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    cat1pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_42!(c21, cat1, cat1pad, hp1, hw, pad, w, wp1)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c21 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw) + 1
    rem = mod(idxm1, hw)
    i = div(rem, w) + 1
    j = mod(rem, w) + 1
    yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
    cat1pad[yi] = cat1[idx]
    return nothing
end

function cuda_kernel_unet_43!(b_d1a, c1, c21, cat1pad, hp1, hw, kh, khkw, kw, t_d1, w, w_d1a, wp1)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
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
    return nothing
end

function cuda_kernel_unet_44!(n_d1_mid, t_d1, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d1_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d1[i] = max(t_d1[i], zero_val)
    return nothing
end

function cuda_kernel_unet_45!(n_d1_midpad, t_d1pad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d1_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d1pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_46!(c1, hp1, hw, pad, t_d1, t_d1pad, w, wp1)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    idxm1 = idx - 1
    ci = div(idxm1, hw) + 1
    rem = mod(idxm1, hw)
    i = div(rem, w) + 1
    j = mod(rem, w) + 1
    yi = (ci - 1) * hp1 * wp1 + ((i + pad) - 1) * wp1 + (j + pad)
    t_d1pad[yi] = t_d1[idx]
    return nothing
end

function cuda_kernel_unet_47!(b_d1b, c1, dec1out, hp1, hw, kh, khkw, kw, t_d1pad, w, w_d1b, wp1)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
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
    return nothing
end

function cuda_kernel_unet_48!(dec1out, n_d1_out, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d1_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    dec1out[i] = max(dec1out[i], zero_val)
    return nothing
end

function cuda_kernel_unet_49!(b_out, c1, c_out, dec1out, hw, kh_out, khkw_out, kw_out, w, w_out, y)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c_out * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
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
    return nothing
end

function unet_d_cuda(x, xd, h, w, c_in, c1, c2, c3, c_out, w_e1a, w_e1ad, b_e1a, b_e1ad, w_e1b, w_e1bd, b_e1b, b_e1bd, w_e2a, w_e2ad, b_e2a, b_e2ad, w_e2b, w_e2bd, b_e2b, b_e2bd, w_ba, w_bad, b_ba, b_bad, w_bb, w_bbd, b_bb, b_bbd, w_d2a, w_d2ad, b_d2a, b_d2ad, w_d2b, w_d2bd, b_d2b, b_d2bd, w_d1a, w_d1ad, b_d1a, b_d1ad, w_d1b, w_d1bd, b_d1b, b_d1bd, w_out, w_outd, b_out, b_outd, xpad0, xpad0d, t_e1, t_e1d, t_e1pad, t_e1padd, skip1, skip1d, p1, p1d, p1pad, p1padd, t_e2, t_e2d, t_e2pad, t_e2padd, skip2, skip2d, p2, p2d, p2pad, p2padd, t_b, t_bd, t_bpad, t_bpadd, bott, bottd, u2, u2d, cat2, cat2d, cat2pad, cat2padd, t_d2, t_d2d, t_d2pad, t_d2padd, dec2out, dec2outd, u1, u1d, cat1, cat1d, cat1pad, cat1padd, t_d1, t_d1d, t_d1pad, t_d1padd, dec1out, dec1outd, y, yd)
    nthread_per_block = 256
    twod = 0.0
    two = 2
    fourd = 0.0
    four = 4
    h2d = 0.0
    h2 = div(h, two)
    w2d = 0.0
    w2 = div(w, two)
    h4d = 0.0
    h4 = div(h, four)
    w4d = 0.0
    w4 = div(w, four)
    hp1d = 0.0
    hp1 = h + 2
    wp1d = 0.0
    wp1 = w + 2
    hp2d = 0.0
    hp2 = h2 + 2
    wp2d = 0.0
    wp2 = w2 + 2
    hp4d = 0.0
    hp4 = h4 + 2
    wp4d = 0.0
    wp4 = w4 + 2
    padd = 0.0
    pad = 1
    khd = 0.0
    kh = 3
    kwd = 0.0
    kw = 3
    khkwd = 0.0
    khkw = kh * kw
    kh_outd = 0.0
    kh_out = 1
    kw_outd = 0.0
    kw_out = 1
    khkw_outd = 0.0
    khkw_out = kh_out * kw_out
    scaled = 0.0
    scale = 2
    zero_vald = 0.0
    zero_val = 0.0
    c32d = 0.0
    c32 = c3 + c2
    c21d = 0.0
    c21 = c2 + c1
    hwd = 0.0
    hw = h * w
    hw2d = 0.0
    hw2 = h2 * w2
    hw4d = 0.0
    hw4 = h4 * w4
    n_xpad0d = 0.0
    n_xpad0 = c_in * hp1 * wp1
    n_e1_midd = 0.0
    n_e1_mid = c1 * hw
    n_e1_midpadd = 0.0
    n_e1_midpad = c1 * hp1 * wp1
    n_e1_outd = 0.0
    n_e1_out = c1 * hw
    n_p1padd = 0.0
    n_p1pad = c1 * hp2 * wp2
    n_e2_midd = 0.0
    n_e2_mid = c2 * hw2
    n_e2_midpadd = 0.0
    n_e2_midpad = c2 * hp2 * wp2
    n_e2_outd = 0.0
    n_e2_out = c2 * hw2
    n_p2padd = 0.0
    n_p2pad = c2 * hp4 * wp4
    n_b_midd = 0.0
    n_b_mid = c3 * hw4
    n_b_midpadd = 0.0
    n_b_midpad = c3 * hp4 * wp4
    n_b_outd = 0.0
    n_b_out = c3 * hw4
    n_cat2padd = 0.0
    n_cat2pad = c32 * hp2 * wp2
    n_d2_midd = 0.0
    n_d2_mid = c2 * hw2
    n_d2_midpadd = 0.0
    n_d2_midpad = c2 * hp2 * wp2
    n_d2_outd = 0.0
    n_d2_out = c2 * hw2
    n_cat1padd = 0.0
    n_cat1pad = c21 * hp1 * wp1
    n_d1_midd = 0.0
    n_d1_mid = c1 * hw
    n_d1_midpadd = 0.0
    n_d1_midpad = c1 * hp1 * wp1
    n_d1_outd = 0.0
    n_d1_out = c1 * hw
    @cuda threads = nthread_per_block blocks = cld(div(n_xpad0 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_1!(n_xpad0, xpad0, xpad0d, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c_in * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_2!(c_in, hp1, hw, pad, w, wp1, x, xd, xpad0, xpad0d)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_3!(b_e1a, b_e1ad, c1, c_in, hp1, hw, kh, khkw, kw, t_e1, t_e1d, w, w_e1a, w_e1ad, wp1, xpad0, xpad0d)
    @cuda threads = nthread_per_block blocks = cld(div(n_e1_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_4!(n_e1_mid, t_e1, t_e1d, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(n_e1_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_5!(n_e1_midpad, t_e1pad, t_e1padd, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_6!(c1, hp1, hw, pad, t_e1, t_e1d, t_e1pad, t_e1padd, w, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_7!(b_e1b, b_e1bd, c1, hp1, hw, kh, khkw, kw, skip1, skip1d, t_e1pad, t_e1padd, w, w_e1b, w_e1bd, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(n_e1_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_8!(n_e1_out, skip1, skip1d, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_9!(c1, hw, hw2, p1, p1d, skip1, skip1d, w, w2)
    @cuda threads = nthread_per_block blocks = cld(div(n_p1pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_10!(n_p1pad, p1pad, p1padd, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_11!(c1, hp2, hw2, p1, p1d, p1pad, p1padd, pad, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_12!(b_e2a, b_e2ad, c1, c2, hp2, hw2, kh, khkw, kw, p1pad, p1padd, t_e2, t_e2d, w2, w_e2a, w_e2ad, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_e2_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_13!(n_e2_mid, t_e2, t_e2d, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(n_e2_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_14!(n_e2_midpad, t_e2pad, t_e2padd, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_15!(c2, hp2, hw2, pad, t_e2, t_e2d, t_e2pad, t_e2padd, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_16!(b_e2b, b_e2bd, c2, hp2, hw2, kh, khkw, kw, skip2, skip2d, t_e2pad, t_e2padd, w2, w_e2b, w_e2bd, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_e2_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_17!(n_e2_out, skip2, skip2d, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_18!(c2, hw2, hw4, p2, p2d, skip2, skip2d, w2, w4)
    @cuda threads = nthread_per_block blocks = cld(div(n_p2pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_19!(n_p2pad, p2pad, p2padd, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_20!(c2, hp4, hw4, p2, p2d, p2pad, p2padd, pad, w4, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_21!(b_ba, b_bad, c2, c3, hp4, hw4, kh, khkw, kw, p2pad, p2padd, t_b, t_bd, w4, w_ba, w_bad, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(n_b_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_22!(n_b_mid, t_b, t_bd, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(n_b_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_23!(n_b_midpad, t_bpad, t_bpadd, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_24!(c3, hp4, hw4, pad, t_b, t_bd, t_bpad, t_bpadd, w4, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_25!(b_bb, b_bbd, bott, bottd, c3, hp4, hw4, kh, khkw, kw, t_bpad, t_bpadd, w4, w_bb, w_bbd, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(n_b_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_26!(bott, bottd, n_b_out, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_27!(bott, bottd, c3, hw2, hw4, scale, u2, u2d, w2, w4)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_28!(c3, cat2, cat2d, hw2, u2, u2d)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_29!(c2, c3, cat2, cat2d, hw2, skip2, skip2d)
    @cuda threads = nthread_per_block blocks = cld(div(n_cat2pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_30!(cat2pad, cat2padd, n_cat2pad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c32 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_31!(c32, cat2, cat2d, cat2pad, cat2padd, hp2, hw2, pad, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_32!(b_d2a, b_d2ad, c2, c32, cat2pad, cat2padd, hp2, hw2, kh, khkw, kw, t_d2, t_d2d, w2, w_d2a, w_d2ad, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_d2_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_33!(n_d2_mid, t_d2, t_d2d, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(n_d2_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_34!(n_d2_midpad, t_d2pad, t_d2padd, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_35!(c2, hp2, hw2, pad, t_d2, t_d2d, t_d2pad, t_d2padd, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_36!(b_d2b, b_d2bd, c2, dec2out, dec2outd, hp2, hw2, kh, khkw, kw, t_d2pad, t_d2padd, w2, w_d2b, w_d2bd, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_d2_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_37!(dec2out, dec2outd, n_d2_out, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_38!(c2, dec2out, dec2outd, hw, hw2, scale, u1, u1d, w, w2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_39!(c2, cat1, cat1d, hw, u1, u1d)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_40!(c1, c2, cat1, cat1d, hw, skip1, skip1d)
    @cuda threads = nthread_per_block blocks = cld(div(n_cat1pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_41!(cat1pad, cat1padd, n_cat1pad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c21 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_42!(c21, cat1, cat1d, cat1pad, cat1padd, hp1, hw, pad, w, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_43!(b_d1a, b_d1ad, c1, c21, cat1pad, cat1padd, hp1, hw, kh, khkw, kw, t_d1, t_d1d, w, w_d1a, w_d1ad, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(n_d1_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_44!(n_d1_mid, t_d1, t_d1d, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(n_d1_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_45!(n_d1_midpad, t_d1pad, t_d1padd, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_46!(c1, hp1, hw, pad, t_d1, t_d1d, t_d1pad, t_d1padd, w, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_47!(b_d1b, b_d1bd, c1, dec1out, dec1outd, hp1, hw, kh, khkw, kw, t_d1pad, t_d1padd, w, w_d1b, w_d1bd, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(n_d1_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_48!(dec1out, dec1outd, n_d1_out, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c_out * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_d_49!(b_out, b_outd, c1, c_out, dec1out, dec1outd, hw, kh_out, khkw_out, kw_out, w, w_out, w_outd, y, yd)
    return nothing
end

function unet_cuda(x, h, w, c_in, c1, c2, c3, c_out, w_e1a, b_e1a, w_e1b, b_e1b, w_e2a, b_e2a, w_e2b, b_e2b, w_ba, b_ba, w_bb, b_bb, w_d2a, b_d2a, w_d2b, b_d2b, w_d1a, b_d1a, w_d1b, b_d1b, w_out, b_out, xpad0, t_e1, t_e1pad, skip1, p1, p1pad, t_e2, t_e2pad, skip2, p2, p2pad, t_b, t_bpad, bott, u2, cat2, cat2pad, t_d2, t_d2pad, dec2out, u1, cat1, cat1pad, t_d1, t_d1pad, dec1out, y)
    nthread_per_block = 256
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
    @cuda threads = nthread_per_block blocks = cld(div(n_xpad0 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_1!(n_xpad0, xpad0, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c_in * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_2!(c_in, hp1, hw, pad, w, wp1, x, xpad0)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_3!(b_e1a, c1, c_in, hp1, hw, kh, khkw, kw, t_e1, w, w_e1a, wp1, xpad0)
    @cuda threads = nthread_per_block blocks = cld(div(n_e1_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_4!(n_e1_mid, t_e1, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(n_e1_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_5!(n_e1_midpad, t_e1pad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_6!(c1, hp1, hw, pad, t_e1, t_e1pad, w, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_7!(b_e1b, c1, hp1, hw, kh, khkw, kw, skip1, t_e1pad, w, w_e1b, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(n_e1_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_8!(n_e1_out, skip1, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_9!(c1, hw, hw2, p1, skip1, w, w2)
    @cuda threads = nthread_per_block blocks = cld(div(n_p1pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_10!(n_p1pad, p1pad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_11!(c1, hp2, hw2, p1, p1pad, pad, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_12!(b_e2a, c1, c2, hp2, hw2, kh, khkw, kw, p1pad, t_e2, w2, w_e2a, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_e2_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_13!(n_e2_mid, t_e2, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(n_e2_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_14!(n_e2_midpad, t_e2pad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_15!(c2, hp2, hw2, pad, t_e2, t_e2pad, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_16!(b_e2b, c2, hp2, hw2, kh, khkw, kw, skip2, t_e2pad, w2, w_e2b, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_e2_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_17!(n_e2_out, skip2, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_18!(c2, hw2, hw4, p2, skip2, w2, w4)
    @cuda threads = nthread_per_block blocks = cld(div(n_p2pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_19!(n_p2pad, p2pad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_20!(c2, hp4, hw4, p2, p2pad, pad, w4, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_21!(b_ba, c2, c3, hp4, hw4, kh, khkw, kw, p2pad, t_b, w4, w_ba, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(n_b_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_22!(n_b_mid, t_b, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(n_b_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_23!(n_b_midpad, t_bpad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_24!(c3, hp4, hw4, pad, t_b, t_bpad, w4, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_25!(b_bb, bott, c3, hp4, hw4, kh, khkw, kw, t_bpad, w4, w_bb, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(n_b_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_26!(bott, n_b_out, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_27!(bott, c3, hw2, hw4, scale, u2, w2, w4)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_28!(c3, cat2, hw2, u2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_29!(c2, c3, cat2, hw2, skip2)
    @cuda threads = nthread_per_block blocks = cld(div(n_cat2pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_30!(cat2pad, n_cat2pad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c32 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_31!(c32, cat2, cat2pad, hp2, hw2, pad, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_32!(b_d2a, c2, c32, cat2pad, hp2, hw2, kh, khkw, kw, t_d2, w2, w_d2a, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_d2_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_33!(n_d2_mid, t_d2, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(n_d2_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_34!(n_d2_midpad, t_d2pad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_35!(c2, hp2, hw2, pad, t_d2, t_d2pad, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_36!(b_d2b, c2, dec2out, hp2, hw2, kh, khkw, kw, t_d2pad, w2, w_d2b, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_d2_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_37!(dec2out, n_d2_out, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_38!(c2, dec2out, hw, hw2, scale, u1, w, w2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_39!(c2, cat1, hw, u1)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_40!(c1, c2, cat1, hw, skip1)
    @cuda threads = nthread_per_block blocks = cld(div(n_cat1pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_41!(cat1pad, n_cat1pad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c21 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_42!(c21, cat1, cat1pad, hp1, hw, pad, w, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_43!(b_d1a, c1, c21, cat1pad, hp1, hw, kh, khkw, kw, t_d1, w, w_d1a, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(n_d1_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_44!(n_d1_mid, t_d1, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(n_d1_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_45!(n_d1_midpad, t_d1pad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_46!(c1, hp1, hw, pad, t_d1, t_d1pad, w, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_47!(b_d1b, c1, dec1out, hp1, hw, kh, khkw, kw, t_d1pad, w, w_d1b, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(n_d1_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_48!(dec1out, n_d1_out, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c_out * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_49!(b_out, c1, c_out, dec1out, hw, kh_out, khkw_out, kw_out, w, w_out, y)
    return nothing
end
