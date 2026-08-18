import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_unet_hv_1!(n_xpad0, xpad0, xpad0d, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_xpad0 - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    xpad0d[i] = zero_vald
    xpad0[i] = zero_val
    return nothing
end

function cuda_kernel_unet_hv_2!(c_in, hp1, hw, pad, w, wp1, x, xd, xpad0, xpad0d)
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
    xpad0d[yi] = xd[idx]
    xpad0[yi] = x[idx]
    return nothing
end

function cuda_kernel_unet_hv_3!(b_e1a, b_e1ad, c1, c_in, hp1, hw, kh, khkw, kw, t_e1, t_e1d, w, w_e1a, w_e1ad, wp1, xpad0, xpad0d)
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
    sd = 0.0
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
        sd = sd + (w_e1a[wi] * xpad0d[xi] + xpad0[xi] * w_e1ad[wi])
        s = s + xpad0[xi] * w_e1a[wi]
    end
    t_e1d[idx] = sd + b_e1ad[co]
    t_e1[idx] = s + b_e1a[co]
    return nothing
end

function cuda_kernel_unet_hv_4!(n_e1_mid, t_e1, t_e1_stack, t_e1_stack_d, t_e1d, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e1_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e1_stack_d[(i - 1) + 1] = t_e1d[i]
    t_e1_stack[(i - 1) + 1] = t_e1[i]
    t_e1d[i] = (0.5 * (1.0 + sign(t_e1[i] - zero_val))) * t_e1d[i] + (0.5 * (1.0 + sign(zero_val - t_e1[i]))) * zero_vald
    t_e1[i] = max(t_e1[i], zero_val)
    return nothing
end

function cuda_kernel_unet_hv_5!(n_e1_midpad, t_e1pad, t_e1padd, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e1_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e1padd[i] = zero_vald
    t_e1pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_hv_6!(c1, hp1, hw, pad, t_e1, t_e1d, t_e1pad, t_e1padd, w, wp1)
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
    t_e1padd[yi] = t_e1d[idx]
    t_e1pad[yi] = t_e1[idx]
    return nothing
end

function cuda_kernel_unet_hv_7!(b_e1b, b_e1bd, c1, hp1, hw, kh, khkw, kw, skip1, skip1d, t_e1pad, t_e1padd, w, w_e1b, w_e1bd, wp1)
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
    sd = 0.0
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
        sd = sd + (w_e1b[wi] * t_e1padd[xi] + t_e1pad[xi] * w_e1bd[wi])
        s = s + t_e1pad[xi] * w_e1b[wi]
    end
    skip1d[idx] = sd + b_e1bd[co]
    skip1[idx] = s + b_e1b[co]
    return nothing
end

function cuda_kernel_unet_hv_8!(n_e1_out, skip1, skip1_stack, skip1_stack_d, skip1d, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e1_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    skip1_stack_d[(i - 1) + 1] = skip1d[i]
    skip1_stack[(i - 1) + 1] = skip1[i]
    skip1d[i] = (0.5 * (1.0 + sign(skip1[i] - zero_val))) * skip1d[i] + (0.5 * (1.0 + sign(zero_val - skip1[i]))) * zero_vald
    skip1[i] = max(skip1[i], zero_val)
    return nothing
end

function cuda_kernel_unet_hv_9!(a11_stack, a11_stack_d, a12_stack, a12_stack_d, a21_stack, a21_stack_d, a22_stack, a22_stack_d, c1, hw, hw2, m1_stack, m1_stack_d, m2_stack, m2_stack_d, p1, p1d, skip1, skip1d, w, w2)
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
    a11_stack_d[(idx - 1) + 1] = a11d
    a11_stack[(idx - 1) + 1] = a11
    a11d = skip1d[(ci - 1) * hw + (i0 - 1) * w + j0]
    a11 = skip1[(ci - 1) * hw + (i0 - 1) * w + j0]
    a12_stack_d[(idx - 1) + 1] = a12d
    a12_stack[(idx - 1) + 1] = a12
    a12d = skip1d[(ci - 1) * hw + (i0 - 1) * w + j0 + 1]
    a12 = skip1[(ci - 1) * hw + (i0 - 1) * w + j0 + 1]
    a21_stack_d[(idx - 1) + 1] = a21d
    a21_stack[(idx - 1) + 1] = a21
    a21d = skip1d[(ci - 1) * hw + i0 * w + j0]
    a21 = skip1[(ci - 1) * hw + i0 * w + j0]
    a22_stack_d[(idx - 1) + 1] = a22d
    a22_stack[(idx - 1) + 1] = a22
    a22d = skip1d[(ci - 1) * hw + i0 * w + j0 + 1]
    a22 = skip1[(ci - 1) * hw + i0 * w + j0 + 1]
    m1_stack_d[(idx - 1) + 1] = m1d
    m1_stack[(idx - 1) + 1] = m1
    m1d = (0.5 * (1.0 + sign(a11 - a12))) * a11d + (0.5 * (1.0 + sign(a12 - a11))) * a12d
    m1 = max(a11, a12)
    m2_stack_d[(idx - 1) + 1] = m2d
    m2_stack[(idx - 1) + 1] = m2
    m2d = (0.5 * (1.0 + sign(a21 - a22))) * a21d + (0.5 * (1.0 + sign(a22 - a21))) * a22d
    m2 = max(a21, a22)
    p1d[idx] = (0.5 * (1.0 + sign(m1 - m2))) * m1d + (0.5 * (1.0 + sign(m2 - m1))) * m2d
    p1[idx] = max(m1, m2)
    return nothing
end

function cuda_kernel_unet_hv_10!(n_p1pad, p1pad, p1padd, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_p1pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    p1padd[i] = zero_vald
    p1pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_hv_11!(c1, hp2, hw2, p1, p1d, p1pad, p1padd, pad, w2, wp2)
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
    p1padd[yi] = p1d[idx]
    p1pad[yi] = p1[idx]
    return nothing
end

function cuda_kernel_unet_hv_12!(b_e2a, b_e2ad, c1, c2, hp2, hw2, kh, khkw, kw, p1pad, p1padd, t_e2, t_e2d, w2, w_e2a, w_e2ad, wp2)
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
    sd = 0.0
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
        sd = sd + (w_e2a[wi] * p1padd[xi] + p1pad[xi] * w_e2ad[wi])
        s = s + p1pad[xi] * w_e2a[wi]
    end
    t_e2d[idx] = sd + b_e2ad[co]
    t_e2[idx] = s + b_e2a[co]
    return nothing
end

function cuda_kernel_unet_hv_13!(n_e2_mid, t_e2, t_e2_stack, t_e2_stack_d, t_e2d, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e2_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e2_stack_d[(i - 1) + 1] = t_e2d[i]
    t_e2_stack[(i - 1) + 1] = t_e2[i]
    t_e2d[i] = (0.5 * (1.0 + sign(t_e2[i] - zero_val))) * t_e2d[i] + (0.5 * (1.0 + sign(zero_val - t_e2[i]))) * zero_vald
    t_e2[i] = max(t_e2[i], zero_val)
    return nothing
end

function cuda_kernel_unet_hv_14!(n_e2_midpad, t_e2pad, t_e2padd, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e2_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e2padd[i] = zero_vald
    t_e2pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_hv_15!(c2, hp2, hw2, pad, t_e2, t_e2d, t_e2pad, t_e2padd, w2, wp2)
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
    t_e2padd[yi] = t_e2d[idx]
    t_e2pad[yi] = t_e2[idx]
    return nothing
end

function cuda_kernel_unet_hv_16!(b_e2b, b_e2bd, c2, hp2, hw2, kh, khkw, kw, skip2, skip2d, t_e2pad, t_e2padd, w2, w_e2b, w_e2bd, wp2)
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
    sd = 0.0
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
        sd = sd + (w_e2b[wi] * t_e2padd[xi] + t_e2pad[xi] * w_e2bd[wi])
        s = s + t_e2pad[xi] * w_e2b[wi]
    end
    skip2d[idx] = sd + b_e2bd[co]
    skip2[idx] = s + b_e2b[co]
    return nothing
end

function cuda_kernel_unet_hv_17!(n_e2_out, skip2, skip2_stack, skip2_stack_d, skip2d, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e2_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    skip2_stack_d[(i - 1) + 1] = skip2d[i]
    skip2_stack[(i - 1) + 1] = skip2[i]
    skip2d[i] = (0.5 * (1.0 + sign(skip2[i] - zero_val))) * skip2d[i] + (0.5 * (1.0 + sign(zero_val - skip2[i]))) * zero_vald
    skip2[i] = max(skip2[i], zero_val)
    return nothing
end

function cuda_kernel_unet_hv_18!(a11_stack, a11_stack_d, a12_stack, a12_stack_d, a21_stack, a21_stack_d, a22_stack, a22_stack_d, c1, c2, hw2, hw4, m1_stack, m1_stack_d, m2_stack, m2_stack_d, p2, p2d, skip2, skip2d, w2, w4)
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
    a11_stack_d[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)] = a11d
    a11_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)] = a11
    a11d = skip2d[(ci - 1) * hw2 + (i0 - 1) * w2 + j0]
    a11 = skip2[(ci - 1) * hw2 + (i0 - 1) * w2 + j0]
    a12_stack_d[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)] = a12d
    a12_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)] = a12
    a12d = skip2d[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1]
    a12 = skip2[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1]
    a21_stack_d[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)] = a21d
    a21_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)] = a21
    a21d = skip2d[(ci - 1) * hw2 + i0 * w2 + j0]
    a21 = skip2[(ci - 1) * hw2 + i0 * w2 + j0]
    a22_stack_d[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)] = a22d
    a22_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)] = a22
    a22d = skip2d[(ci - 1) * hw2 + i0 * w2 + j0 + 1]
    a22 = skip2[(ci - 1) * hw2 + i0 * w2 + j0 + 1]
    m1_stack_d[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)] = m1d
    m1_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)] = m1
    m1d = (0.5 * (1.0 + sign(a11 - a12))) * a11d + (0.5 * (1.0 + sign(a12 - a11))) * a12d
    m1 = max(a11, a12)
    m2_stack_d[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)] = m2d
    m2_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)] = m2
    m2d = (0.5 * (1.0 + sign(a21 - a22))) * a21d + (0.5 * (1.0 + sign(a22 - a21))) * a22d
    m2 = max(a21, a22)
    p2d[idx] = (0.5 * (1.0 + sign(m1 - m2))) * m1d + (0.5 * (1.0 + sign(m2 - m1))) * m2d
    p2[idx] = max(m1, m2)
    return nothing
end

function cuda_kernel_unet_hv_19!(n_p2pad, p2pad, p2padd, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_p2pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    p2padd[i] = zero_vald
    p2pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_hv_20!(c2, hp4, hw4, p2, p2d, p2pad, p2padd, pad, w4, wp4)
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
    p2padd[yi] = p2d[idx]
    p2pad[yi] = p2[idx]
    return nothing
end

function cuda_kernel_unet_hv_21!(b_ba, b_bad, c2, c3, hp4, hw4, kh, khkw, kw, p2pad, p2padd, t_b, t_bd, w4, w_ba, w_bad, wp4)
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
    sd = 0.0
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
        sd = sd + (w_ba[wi] * p2padd[xi] + p2pad[xi] * w_bad[wi])
        s = s + p2pad[xi] * w_ba[wi]
    end
    t_bd[idx] = sd + b_bad[co]
    t_b[idx] = s + b_ba[co]
    return nothing
end

function cuda_kernel_unet_hv_22!(n_b_mid, t_b, t_b_stack, t_b_stack_d, t_bd, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_b_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_b_stack_d[(i - 1) + 1] = t_bd[i]
    t_b_stack[(i - 1) + 1] = t_b[i]
    t_bd[i] = (0.5 * (1.0 + sign(t_b[i] - zero_val))) * t_bd[i] + (0.5 * (1.0 + sign(zero_val - t_b[i]))) * zero_vald
    t_b[i] = max(t_b[i], zero_val)
    return nothing
end

function cuda_kernel_unet_hv_23!(n_b_midpad, t_bpad, t_bpadd, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_b_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_bpadd[i] = zero_vald
    t_bpad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_hv_24!(c3, hp4, hw4, pad, t_b, t_bd, t_bpad, t_bpadd, w4, wp4)
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
    t_bpadd[yi] = t_bd[idx]
    t_bpad[yi] = t_b[idx]
    return nothing
end

function cuda_kernel_unet_hv_25!(b_bb, b_bbd, bott, bottd, c3, hp4, hw4, kh, khkw, kw, t_bpad, t_bpadd, w4, w_bb, w_bbd, wp4)
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
    sd = 0.0
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
        sd = sd + (w_bb[wi] * t_bpadd[xi] + t_bpad[xi] * w_bbd[wi])
        s = s + t_bpad[xi] * w_bb[wi]
    end
    bottd[idx] = sd + b_bbd[co]
    bott[idx] = s + b_bb[co]
    return nothing
end

function cuda_kernel_unet_hv_26!(bott, bott_stack, bott_stack_d, bottd, n_b_out, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_b_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    bott_stack_d[(i - 1) + 1] = bottd[i]
    bott_stack[(i - 1) + 1] = bott[i]
    bottd[i] = (0.5 * (1.0 + sign(bott[i] - zero_val))) * bottd[i] + (0.5 * (1.0 + sign(zero_val - bott[i]))) * zero_vald
    bott[i] = max(bott[i], zero_val)
    return nothing
end

function cuda_kernel_unet_hv_27!(bott, bottd, c3, hw2, hw4, scale, u2, u2d, w2, w4)
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
    u2d[idx] = bottd[xi]
    u2[idx] = bott[xi]
    return nothing
end

function cuda_kernel_unet_hv_28!(c3, cat2, cat2d, hw2, u2, u2d)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c3 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    cat2d[idx] = u2d[idx]
    cat2[idx] = u2[idx]
    return nothing
end

function cuda_kernel_unet_hv_29!(c2, c3, cat2, cat2d, hw2, skip2, skip2d)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    cat2d[c3 * hw2 + idx] = skip2d[idx]
    cat2[c3 * hw2 + idx] = skip2[idx]
    return nothing
end

function cuda_kernel_unet_hv_30!(cat2pad, cat2padd, n_cat2pad, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_cat2pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    cat2padd[i] = zero_vald
    cat2pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_hv_31!(c32, cat2, cat2d, cat2pad, cat2padd, hp2, hw2, pad, w2, wp2)
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
    cat2padd[yi] = cat2d[idx]
    cat2pad[yi] = cat2[idx]
    return nothing
end

function cuda_kernel_unet_hv_32!(b_d2a, b_d2ad, c2, c32, cat2pad, cat2padd, hp2, hw2, kh, khkw, kw, t_d2, t_d2d, w2, w_d2a, w_d2ad, wp2)
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
    sd = 0.0
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
        sd = sd + (w_d2a[wi] * cat2padd[xi] + cat2pad[xi] * w_d2ad[wi])
        s = s + cat2pad[xi] * w_d2a[wi]
    end
    t_d2d[idx] = sd + b_d2ad[co]
    t_d2[idx] = s + b_d2a[co]
    return nothing
end

function cuda_kernel_unet_hv_33!(n_d2_mid, t_d2, t_d2_stack, t_d2_stack_d, t_d2d, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d2_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d2_stack_d[(i - 1) + 1] = t_d2d[i]
    t_d2_stack[(i - 1) + 1] = t_d2[i]
    t_d2d[i] = (0.5 * (1.0 + sign(t_d2[i] - zero_val))) * t_d2d[i] + (0.5 * (1.0 + sign(zero_val - t_d2[i]))) * zero_vald
    t_d2[i] = max(t_d2[i], zero_val)
    return nothing
end

function cuda_kernel_unet_hv_34!(n_d2_midpad, t_d2pad, t_d2padd, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d2_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d2padd[i] = zero_vald
    t_d2pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_hv_35!(c2, hp2, hw2, pad, t_d2, t_d2d, t_d2pad, t_d2padd, w2, wp2)
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
    t_d2padd[yi] = t_d2d[idx]
    t_d2pad[yi] = t_d2[idx]
    return nothing
end

function cuda_kernel_unet_hv_36!(b_d2b, b_d2bd, c2, dec2out, dec2outd, hp2, hw2, kh, khkw, kw, t_d2pad, t_d2padd, w2, w_d2b, w_d2bd, wp2)
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
    sd = 0.0
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
        sd = sd + (w_d2b[wi] * t_d2padd[xi] + t_d2pad[xi] * w_d2bd[wi])
        s = s + t_d2pad[xi] * w_d2b[wi]
    end
    dec2outd[idx] = sd + b_d2bd[co]
    dec2out[idx] = s + b_d2b[co]
    return nothing
end

function cuda_kernel_unet_hv_37!(dec2out, dec2out_stack, dec2out_stack_d, dec2outd, n_d2_out, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d2_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    dec2out_stack_d[(i - 1) + 1] = dec2outd[i]
    dec2out_stack[(i - 1) + 1] = dec2out[i]
    dec2outd[i] = (0.5 * (1.0 + sign(dec2out[i] - zero_val))) * dec2outd[i] + (0.5 * (1.0 + sign(zero_val - dec2out[i]))) * zero_vald
    dec2out[i] = max(dec2out[i], zero_val)
    return nothing
end

function cuda_kernel_unet_hv_38!(c2, dec2out, dec2outd, hw, hw2, scale, u1, u1d, w, w2)
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
    u1d[idx] = dec2outd[xi]
    u1[idx] = dec2out[xi]
    return nothing
end

function cuda_kernel_unet_hv_39!(c2, cat1, cat1d, hw, u1, u1d)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    cat1d[idx] = u1d[idx]
    cat1[idx] = u1[idx]
    return nothing
end

function cuda_kernel_unet_hv_40!(c1, c2, cat1, cat1d, hw, skip1, skip1d)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    cat1d[c2 * hw + idx] = skip1d[idx]
    cat1[c2 * hw + idx] = skip1[idx]
    return nothing
end

function cuda_kernel_unet_hv_41!(cat1pad, cat1padd, n_cat1pad, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_cat1pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    cat1padd[i] = zero_vald
    cat1pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_hv_42!(c21, cat1, cat1d, cat1pad, cat1padd, hp1, hw, pad, w, wp1)
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
    cat1padd[yi] = cat1d[idx]
    cat1pad[yi] = cat1[idx]
    return nothing
end

function cuda_kernel_unet_hv_43!(b_d1a, b_d1ad, c1, c21, cat1pad, cat1padd, hp1, hw, kh, khkw, kw, t_d1, t_d1d, w, w_d1a, w_d1ad, wp1)
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
    sd = 0.0
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
        sd = sd + (w_d1a[wi] * cat1padd[xi] + cat1pad[xi] * w_d1ad[wi])
        s = s + cat1pad[xi] * w_d1a[wi]
    end
    t_d1d[idx] = sd + b_d1ad[co]
    t_d1[idx] = s + b_d1a[co]
    return nothing
end

function cuda_kernel_unet_hv_44!(n_d1_mid, t_d1, t_d1_stack, t_d1_stack_d, t_d1d, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d1_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d1_stack_d[(i - 1) + 1] = t_d1d[i]
    t_d1_stack[(i - 1) + 1] = t_d1[i]
    t_d1d[i] = (0.5 * (1.0 + sign(t_d1[i] - zero_val))) * t_d1d[i] + (0.5 * (1.0 + sign(zero_val - t_d1[i]))) * zero_vald
    t_d1[i] = max(t_d1[i], zero_val)
    return nothing
end

function cuda_kernel_unet_hv_45!(n_d1_midpad, t_d1pad, t_d1padd, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d1_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d1padd[i] = zero_vald
    t_d1pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_hv_46!(c1, hp1, hw, pad, t_d1, t_d1d, t_d1pad, t_d1padd, w, wp1)
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
    t_d1padd[yi] = t_d1d[idx]
    t_d1pad[yi] = t_d1[idx]
    return nothing
end

function cuda_kernel_unet_hv_47!(b_d1b, b_d1bd, c1, dec1out, dec1outd, hp1, hw, kh, khkw, kw, t_d1pad, t_d1padd, w, w_d1b, w_d1bd, wp1)
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
    sd = 0.0
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
        sd = sd + (w_d1b[wi] * t_d1padd[xi] + t_d1pad[xi] * w_d1bd[wi])
        s = s + t_d1pad[xi] * w_d1b[wi]
    end
    dec1outd[idx] = sd + b_d1bd[co]
    dec1out[idx] = s + b_d1b[co]
    return nothing
end

function cuda_kernel_unet_hv_48!(dec1out, dec1out_stack, dec1out_stack_d, dec1outd, n_d1_out, zero_val, zero_vald)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d1_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    dec1out_stack_d[(i - 1) + 1] = dec1outd[i]
    dec1out_stack[(i - 1) + 1] = dec1out[i]
    dec1outd[i] = (0.5 * (1.0 + sign(dec1out[i] - zero_val))) * dec1outd[i] + (0.5 * (1.0 + sign(zero_val - dec1out[i]))) * zero_vald
    dec1out[i] = max(dec1out[i], zero_val)
    return nothing
end

function cuda_kernel_unet_hv_49!(b_out, b_outd, c1, c_out, dec1out, dec1outd, hw, kh_out, khkw_out, kw_out, w, w_out, w_outd, y, yd)
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
    sd = 0.0
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
        sd = sd + (w_out[wi] * dec1outd[xi] + dec1out[xi] * w_outd[wi])
        s = s + dec1out[xi] * w_out[wi]
    end
    yd[idx] = sd + b_outd[co]
    y[idx] = s + b_out[co]
    return nothing
end

function cuda_kernel_unet_hv_50!(b_outb, b_outbd, c1, c_out, dec1out, dec1outb, dec1outbd, dec1outd, hw, kh_out, khkw_out, kw_out, w, w_out, w_outb, w_outbd, w_outd, yb, ybd)
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
    sbd = sbd + ybd[idx]
    sb = sb + yb[idx]
    b_outbd[co] = b_outbd[co] + ybd[idx]
    b_outb[co] = b_outb[co] + yb[idx]
    ybd[idx] = 0.0
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
        dec1outbd[xi] = dec1outbd[xi] + (sb * w_outd[wi] + w_out[wi] * sbd)
        dec1outb[xi] = dec1outb[xi] + w_out[wi] * sb
        w_outbd[wi] = w_outbd[wi] + (sb * dec1outd[xi] + dec1out[xi] * sbd)
        w_outb[wi] = w_outb[wi] + dec1out[xi] * sb
    end
    sbd = 0.0
    sb = 0.0
    return nothing
end

function cuda_kernel_unet_hv_51!(dec1out, dec1out_stack, dec1out_stack_d, dec1outb, dec1outbd, dec1outd, n_d1_out, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_d1_out, -1) + 1
        return nothing
    end
    i = n_d1_out + (__tid - 1) * -1
    dec1outd[i] = dec1out_stack_d[(i - 1) + 1]
    dec1out[i] = dec1out_stack[(i - 1) + 1]
    dec1outbd[i] = (0.5 * (1.0 + sign(dec1out[i] - zero_val))) * dec1outbd[i]
    dec1outb[i] = (0.5 * (1.0 + sign(dec1out[i] - zero_val))) * dec1outb[i]
    return nothing
end

function cuda_kernel_unet_hv_52!(b_d1bb, b_d1bbd, c1, dec1outb, dec1outbd, hp1, hw, kh, khkw, kw, t_d1pad, t_d1padb, t_d1padbd, t_d1padd, w, w_d1b, w_d1bb, w_d1bbd, w_d1bd, wp1)
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
    sbd = sbd + dec1outbd[idx]
    sb = sb + dec1outb[idx]
    b_d1bbd[co] = b_d1bbd[co] + dec1outbd[idx]
    b_d1bb[co] = b_d1bb[co] + dec1outb[idx]
    dec1outbd[idx] = 0.0
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
        t_d1padbd[xi] = t_d1padbd[xi] + (sb * w_d1bd[wi] + w_d1b[wi] * sbd)
        t_d1padb[xi] = t_d1padb[xi] + w_d1b[wi] * sb
        w_d1bbd[wi] = w_d1bbd[wi] + (sb * t_d1padd[xi] + t_d1pad[xi] * sbd)
        w_d1bb[wi] = w_d1bb[wi] + t_d1pad[xi] * sb
    end
    sbd = 0.0
    sb = 0.0
    return nothing
end

function cuda_kernel_unet_hv_53!(c1, hp1, hw, pad, t_d1b, t_d1bd, t_d1padb, t_d1padbd, w, wp1)
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
    t_d1bd[idx] = t_d1bd[idx] + t_d1padbd[yi]
    t_d1b[idx] = t_d1b[idx] + t_d1padb[yi]
    t_d1padbd[yi] = 0.0
    t_d1padb[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_54!(n_d1_midpad, t_d1padb, t_d1padbd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d1_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d1padbd[i] = 0.0
    t_d1padb[i] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_55!(n_d1_mid, t_d1, t_d1_stack, t_d1_stack_d, t_d1b, t_d1bd, t_d1d, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_d1_mid, -1) + 1
        return nothing
    end
    i = n_d1_mid + (__tid - 1) * -1
    t_d1d[i] = t_d1_stack_d[(i - 1) + 1]
    t_d1[i] = t_d1_stack[(i - 1) + 1]
    t_d1bd[i] = (0.5 * (1.0 + sign(t_d1[i] - zero_val))) * t_d1bd[i]
    t_d1b[i] = (0.5 * (1.0 + sign(t_d1[i] - zero_val))) * t_d1b[i]
    return nothing
end

function cuda_kernel_unet_hv_56!(b_d1ab, b_d1abd, c1, c21, cat1pad, cat1padb, cat1padbd, cat1padd, hp1, hw, kh, khkw, kw, t_d1b, t_d1bd, w, w_d1a, w_d1ab, w_d1abd, w_d1ad, wp1)
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
    sbd = sbd + t_d1bd[idx]
    sb = sb + t_d1b[idx]
    b_d1abd[co] = b_d1abd[co] + t_d1bd[idx]
    b_d1ab[co] = b_d1ab[co] + t_d1b[idx]
    t_d1bd[idx] = 0.0
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
        cat1padbd[xi] = cat1padbd[xi] + (sb * w_d1ad[wi] + w_d1a[wi] * sbd)
        cat1padb[xi] = cat1padb[xi] + w_d1a[wi] * sb
        w_d1abd[wi] = w_d1abd[wi] + (sb * cat1padd[xi] + cat1pad[xi] * sbd)
        w_d1ab[wi] = w_d1ab[wi] + cat1pad[xi] * sb
    end
    sbd = 0.0
    sb = 0.0
    return nothing
end

function cuda_kernel_unet_hv_57!(c21, cat1b, cat1bd, cat1padb, cat1padbd, hp1, hw, pad, w, wp1)
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
    cat1bd[idx] = cat1bd[idx] + cat1padbd[yi]
    cat1b[idx] = cat1b[idx] + cat1padb[yi]
    cat1padbd[yi] = 0.0
    cat1padb[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_58!(cat1padb, cat1padbd, n_cat1pad)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_cat1pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    cat1padbd[i] = 0.0
    cat1padb[i] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_59!(c1, c2, cat1b, cat1bd, hw, skip1b, skip1bd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    skip1bd[idx] = skip1bd[idx] + cat1bd[c2 * hw + idx]
    skip1b[idx] = skip1b[idx] + cat1b[c2 * hw + idx]
    cat1bd[c2 * hw + idx] = 0.0
    cat1b[c2 * hw + idx] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_60!(c2, cat1b, cat1bd, hw, u1b, u1bd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    u1bd[idx] = u1bd[idx] + cat1bd[idx]
    u1b[idx] = u1b[idx] + cat1b[idx]
    cat1bd[idx] = 0.0
    cat1b[idx] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_61!(c2, dec2outb, dec2outbd, hw, hw2, scale, u1b, u1bd, w, w2)
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
    dec2outbd[xi] = dec2outbd[xi] + u1bd[idx]
    dec2outb[xi] = dec2outb[xi] + u1b[idx]
    u1bd[idx] = 0.0
    u1b[idx] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_62!(dec2out, dec2out_stack, dec2out_stack_d, dec2outb, dec2outbd, dec2outd, n_d2_out, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_d2_out, -1) + 1
        return nothing
    end
    i = n_d2_out + (__tid - 1) * -1
    dec2outd[i] = dec2out_stack_d[(i - 1) + 1]
    dec2out[i] = dec2out_stack[(i - 1) + 1]
    dec2outbd[i] = (0.5 * (1.0 + sign(dec2out[i] - zero_val))) * dec2outbd[i]
    dec2outb[i] = (0.5 * (1.0 + sign(dec2out[i] - zero_val))) * dec2outb[i]
    return nothing
end

function cuda_kernel_unet_hv_63!(b_d2bb, b_d2bbd, c2, dec2outb, dec2outbd, hp2, hw2, kh, khkw, kw, t_d2pad, t_d2padb, t_d2padbd, t_d2padd, w2, w_d2b, w_d2bb, w_d2bbd, w_d2bd, wp2)
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
    sbd = sbd + dec2outbd[idx]
    sb = sb + dec2outb[idx]
    b_d2bbd[co] = b_d2bbd[co] + dec2outbd[idx]
    b_d2bb[co] = b_d2bb[co] + dec2outb[idx]
    dec2outbd[idx] = 0.0
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
        t_d2padbd[xi] = t_d2padbd[xi] + (sb * w_d2bd[wi] + w_d2b[wi] * sbd)
        t_d2padb[xi] = t_d2padb[xi] + w_d2b[wi] * sb
        w_d2bbd[wi] = w_d2bbd[wi] + (sb * t_d2padd[xi] + t_d2pad[xi] * sbd)
        w_d2bb[wi] = w_d2bb[wi] + t_d2pad[xi] * sb
    end
    sbd = 0.0
    sb = 0.0
    return nothing
end

function cuda_kernel_unet_hv_64!(c2, hp2, hw2, pad, t_d2b, t_d2bd, t_d2padb, t_d2padbd, w2, wp2)
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
    t_d2bd[idx] = t_d2bd[idx] + t_d2padbd[yi]
    t_d2b[idx] = t_d2b[idx] + t_d2padb[yi]
    t_d2padbd[yi] = 0.0
    t_d2padb[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_65!(n_d2_midpad, t_d2padb, t_d2padbd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d2_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d2padbd[i] = 0.0
    t_d2padb[i] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_66!(n_d2_mid, t_d2, t_d2_stack, t_d2_stack_d, t_d2b, t_d2bd, t_d2d, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_d2_mid, -1) + 1
        return nothing
    end
    i = n_d2_mid + (__tid - 1) * -1
    t_d2d[i] = t_d2_stack_d[(i - 1) + 1]
    t_d2[i] = t_d2_stack[(i - 1) + 1]
    t_d2bd[i] = (0.5 * (1.0 + sign(t_d2[i] - zero_val))) * t_d2bd[i]
    t_d2b[i] = (0.5 * (1.0 + sign(t_d2[i] - zero_val))) * t_d2b[i]
    return nothing
end

function cuda_kernel_unet_hv_67!(b_d2ab, b_d2abd, c2, c32, cat2pad, cat2padb, cat2padbd, cat2padd, hp2, hw2, kh, khkw, kw, t_d2b, t_d2bd, w2, w_d2a, w_d2ab, w_d2abd, w_d2ad, wp2)
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
    sbd = sbd + t_d2bd[idx]
    sb = sb + t_d2b[idx]
    b_d2abd[co] = b_d2abd[co] + t_d2bd[idx]
    b_d2ab[co] = b_d2ab[co] + t_d2b[idx]
    t_d2bd[idx] = 0.0
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
        cat2padbd[xi] = cat2padbd[xi] + (sb * w_d2ad[wi] + w_d2a[wi] * sbd)
        cat2padb[xi] = cat2padb[xi] + w_d2a[wi] * sb
        w_d2abd[wi] = w_d2abd[wi] + (sb * cat2padd[xi] + cat2pad[xi] * sbd)
        w_d2ab[wi] = w_d2ab[wi] + cat2pad[xi] * sb
    end
    sbd = 0.0
    sb = 0.0
    return nothing
end

function cuda_kernel_unet_hv_68!(c32, cat2b, cat2bd, cat2padb, cat2padbd, hp2, hw2, pad, w2, wp2)
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
    cat2bd[idx] = cat2bd[idx] + cat2padbd[yi]
    cat2b[idx] = cat2b[idx] + cat2padb[yi]
    cat2padbd[yi] = 0.0
    cat2padb[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_69!(cat2padb, cat2padbd, n_cat2pad)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_cat2pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    cat2padbd[i] = 0.0
    cat2padb[i] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_70!(c2, c3, cat2b, cat2bd, hw2, skip2b, skip2bd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    skip2bd[idx] = skip2bd[idx] + cat2bd[c3 * hw2 + idx]
    skip2b[idx] = skip2b[idx] + cat2b[c3 * hw2 + idx]
    cat2bd[c3 * hw2 + idx] = 0.0
    cat2b[c3 * hw2 + idx] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_71!(c3, cat2b, cat2bd, hw2, u2b, u2bd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c3 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    u2bd[idx] = u2bd[idx] + cat2bd[idx]
    u2b[idx] = u2b[idx] + cat2b[idx]
    cat2bd[idx] = 0.0
    cat2b[idx] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_72!(bottb, bottbd, c3, hw2, hw4, scale, u2b, u2bd, w2, w4)
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
    bottbd[xi] = bottbd[xi] + u2bd[idx]
    bottb[xi] = bottb[xi] + u2b[idx]
    u2bd[idx] = 0.0
    u2b[idx] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_73!(bott, bott_stack, bott_stack_d, bottb, bottbd, bottd, n_b_out, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_b_out, -1) + 1
        return nothing
    end
    i = n_b_out + (__tid - 1) * -1
    bottd[i] = bott_stack_d[(i - 1) + 1]
    bott[i] = bott_stack[(i - 1) + 1]
    bottbd[i] = (0.5 * (1.0 + sign(bott[i] - zero_val))) * bottbd[i]
    bottb[i] = (0.5 * (1.0 + sign(bott[i] - zero_val))) * bottb[i]
    return nothing
end

function cuda_kernel_unet_hv_74!(b_bbb, b_bbbd, bottb, bottbd, c3, hp4, hw4, kh, khkw, kw, t_bpad, t_bpadb, t_bpadbd, t_bpadd, w4, w_bb, w_bbb, w_bbbd, w_bbd, wp4)
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
    sbd = sbd + bottbd[idx]
    sb = sb + bottb[idx]
    b_bbbd[co] = b_bbbd[co] + bottbd[idx]
    b_bbb[co] = b_bbb[co] + bottb[idx]
    bottbd[idx] = 0.0
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
        t_bpadbd[xi] = t_bpadbd[xi] + (sb * w_bbd[wi] + w_bb[wi] * sbd)
        t_bpadb[xi] = t_bpadb[xi] + w_bb[wi] * sb
        w_bbbd[wi] = w_bbbd[wi] + (sb * t_bpadd[xi] + t_bpad[xi] * sbd)
        w_bbb[wi] = w_bbb[wi] + t_bpad[xi] * sb
    end
    sbd = 0.0
    sb = 0.0
    return nothing
end

function cuda_kernel_unet_hv_75!(c3, hp4, hw4, pad, t_bb, t_bbd, t_bpadb, t_bpadbd, w4, wp4)
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
    t_bbd[idx] = t_bbd[idx] + t_bpadbd[yi]
    t_bb[idx] = t_bb[idx] + t_bpadb[yi]
    t_bpadbd[yi] = 0.0
    t_bpadb[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_76!(n_b_midpad, t_bpadb, t_bpadbd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_b_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_bpadbd[i] = 0.0
    t_bpadb[i] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_77!(n_b_mid, t_b, t_b_stack, t_b_stack_d, t_bb, t_bbd, t_bd, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_b_mid, -1) + 1
        return nothing
    end
    i = n_b_mid + (__tid - 1) * -1
    t_bd[i] = t_b_stack_d[(i - 1) + 1]
    t_b[i] = t_b_stack[(i - 1) + 1]
    t_bbd[i] = (0.5 * (1.0 + sign(t_b[i] - zero_val))) * t_bbd[i]
    t_bb[i] = (0.5 * (1.0 + sign(t_b[i] - zero_val))) * t_bb[i]
    return nothing
end

function cuda_kernel_unet_hv_78!(b_bab, b_babd, c2, c3, hp4, hw4, kh, khkw, kw, p2pad, p2padb, p2padbd, p2padd, t_bb, t_bbd, w4, w_ba, w_bab, w_babd, w_bad, wp4)
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
    sbd = sbd + t_bbd[idx]
    sb = sb + t_bb[idx]
    b_babd[co] = b_babd[co] + t_bbd[idx]
    b_bab[co] = b_bab[co] + t_bb[idx]
    t_bbd[idx] = 0.0
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
        p2padbd[xi] = p2padbd[xi] + (sb * w_bad[wi] + w_ba[wi] * sbd)
        p2padb[xi] = p2padb[xi] + w_ba[wi] * sb
        w_babd[wi] = w_babd[wi] + (sb * p2padd[xi] + p2pad[xi] * sbd)
        w_bab[wi] = w_bab[wi] + p2pad[xi] * sb
    end
    sbd = 0.0
    sb = 0.0
    return nothing
end

function cuda_kernel_unet_hv_79!(c2, hp4, hw4, p2b, p2bd, p2padb, p2padbd, pad, w4, wp4)
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
    p2bd[idx] = p2bd[idx] + p2padbd[yi]
    p2b[idx] = p2b[idx] + p2padb[yi]
    p2padbd[yi] = 0.0
    p2padb[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_80!(n_p2pad, p2padb, p2padbd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_p2pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    p2padbd[i] = 0.0
    p2padb[i] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_81!(a11_stack, a11_stack_d, a12_stack, a12_stack_d, a21_stack, a21_stack_d, a22_stack, a22_stack_d, c1, c2, hw2, hw4, m1_stack, m1_stack_d, m2_stack, m2_stack_d, p2b, p2bd, skip2b, skip2bd, w2, w4)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - c2 * hw4, -1) + 1
        return nothing
    end
    idx = c2 * hw4 + (__tid - 1) * -1
    idxm1 = idx - 1
    ci = div(idxm1, hw4) + 1
    rem = mod(idxm1, hw4)
    oi = div(rem, w4) + 1
    oj = mod(rem, w4) + 1
    i0 = 2oi - 1
    j0 = 2oj - 1
    m1bd = m1bd + (0.5 * (1.0 + sign(m1 - m2))) * p2bd[idx]
    m1b = m1b + (0.5 * (1.0 + sign(m1 - m2))) * p2b[idx]
    m2bd = m2bd + (0.5 * (1.0 + sign(m2 - m1))) * p2bd[idx]
    m2b = m2b + (0.5 * (1.0 + sign(m2 - m1))) * p2b[idx]
    p2bd[idx] = 0.0
    p2b[idx] = 0.0
    m2d = m2_stack_d[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)]
    m2 = m2_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)]
    a21bd = a21bd + (0.5 * (1.0 + sign(a21 - a22))) * m2bd
    a21b = a21b + (0.5 * (1.0 + sign(a21 - a22))) * m2b
    a22bd = a22bd + (0.5 * (1.0 + sign(a22 - a21))) * m2bd
    a22b = a22b + (0.5 * (1.0 + sign(a22 - a21))) * m2b
    m2bd = 0.0
    m2b = 0.0
    m1d = m1_stack_d[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)]
    m1 = m1_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)]
    a11bd = a11bd + (0.5 * (1.0 + sign(a11 - a12))) * m1bd
    a11b = a11b + (0.5 * (1.0 + sign(a11 - a12))) * m1b
    a12bd = a12bd + (0.5 * (1.0 + sign(a12 - a11))) * m1bd
    a12b = a12b + (0.5 * (1.0 + sign(a12 - a11))) * m1b
    m1bd = 0.0
    m1b = 0.0
    a22d = a22_stack_d[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)]
    a22 = a22_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)]
    skip2bd[(ci - 1) * hw2 + i0 * w2 + j0 + 1] = skip2bd[(ci - 1) * hw2 + i0 * w2 + j0 + 1] + a22bd
    skip2b[(ci - 1) * hw2 + i0 * w2 + j0 + 1] = skip2b[(ci - 1) * hw2 + i0 * w2 + j0 + 1] + a22b
    a22bd = 0.0
    a22b = 0.0
    a21d = a21_stack_d[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)]
    a21 = a21_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)]
    skip2bd[(ci - 1) * hw2 + i0 * w2 + j0] = skip2bd[(ci - 1) * hw2 + i0 * w2 + j0] + a21bd
    skip2b[(ci - 1) * hw2 + i0 * w2 + j0] = skip2b[(ci - 1) * hw2 + i0 * w2 + j0] + a21b
    a21bd = 0.0
    a21b = 0.0
    a12d = a12_stack_d[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)]
    a12 = a12_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)]
    skip2bd[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1] = skip2bd[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1] + a12bd
    skip2b[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1] = skip2b[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1] + a12b
    a12bd = 0.0
    a12b = 0.0
    a11d = a11_stack_d[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)]
    a11 = a11_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)]
    skip2bd[(ci - 1) * hw2 + (i0 - 1) * w2 + j0] = skip2bd[(ci - 1) * hw2 + (i0 - 1) * w2 + j0] + a11bd
    skip2b[(ci - 1) * hw2 + (i0 - 1) * w2 + j0] = skip2b[(ci - 1) * hw2 + (i0 - 1) * w2 + j0] + a11b
    a11bd = 0.0
    a11b = 0.0
    return nothing
end

function cuda_kernel_unet_hv_82!(n_e2_out, skip2, skip2_stack, skip2_stack_d, skip2b, skip2bd, skip2d, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_e2_out, -1) + 1
        return nothing
    end
    i = n_e2_out + (__tid - 1) * -1
    skip2d[i] = skip2_stack_d[(i - 1) + 1]
    skip2[i] = skip2_stack[(i - 1) + 1]
    skip2bd[i] = (0.5 * (1.0 + sign(skip2[i] - zero_val))) * skip2bd[i]
    skip2b[i] = (0.5 * (1.0 + sign(skip2[i] - zero_val))) * skip2b[i]
    return nothing
end

function cuda_kernel_unet_hv_83!(b_e2bb, b_e2bbd, c2, hp2, hw2, kh, khkw, kw, skip2b, skip2bd, t_e2pad, t_e2padb, t_e2padbd, t_e2padd, w2, w_e2b, w_e2bb, w_e2bbd, w_e2bd, wp2)
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
    sbd = sbd + skip2bd[idx]
    sb = sb + skip2b[idx]
    b_e2bbd[co] = b_e2bbd[co] + skip2bd[idx]
    b_e2bb[co] = b_e2bb[co] + skip2b[idx]
    skip2bd[idx] = 0.0
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
        t_e2padbd[xi] = t_e2padbd[xi] + (sb * w_e2bd[wi] + w_e2b[wi] * sbd)
        t_e2padb[xi] = t_e2padb[xi] + w_e2b[wi] * sb
        w_e2bbd[wi] = w_e2bbd[wi] + (sb * t_e2padd[xi] + t_e2pad[xi] * sbd)
        w_e2bb[wi] = w_e2bb[wi] + t_e2pad[xi] * sb
    end
    sbd = 0.0
    sb = 0.0
    return nothing
end

function cuda_kernel_unet_hv_84!(c2, hp2, hw2, pad, t_e2b, t_e2bd, t_e2padb, t_e2padbd, w2, wp2)
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
    t_e2bd[idx] = t_e2bd[idx] + t_e2padbd[yi]
    t_e2b[idx] = t_e2b[idx] + t_e2padb[yi]
    t_e2padbd[yi] = 0.0
    t_e2padb[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_85!(n_e2_midpad, t_e2padb, t_e2padbd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e2_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e2padbd[i] = 0.0
    t_e2padb[i] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_86!(n_e2_mid, t_e2, t_e2_stack, t_e2_stack_d, t_e2b, t_e2bd, t_e2d, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_e2_mid, -1) + 1
        return nothing
    end
    i = n_e2_mid + (__tid - 1) * -1
    t_e2d[i] = t_e2_stack_d[(i - 1) + 1]
    t_e2[i] = t_e2_stack[(i - 1) + 1]
    t_e2bd[i] = (0.5 * (1.0 + sign(t_e2[i] - zero_val))) * t_e2bd[i]
    t_e2b[i] = (0.5 * (1.0 + sign(t_e2[i] - zero_val))) * t_e2b[i]
    return nothing
end

function cuda_kernel_unet_hv_87!(b_e2ab, b_e2abd, c1, c2, hp2, hw2, kh, khkw, kw, p1pad, p1padb, p1padbd, p1padd, t_e2b, t_e2bd, w2, w_e2a, w_e2ab, w_e2abd, w_e2ad, wp2)
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
    sbd = sbd + t_e2bd[idx]
    sb = sb + t_e2b[idx]
    b_e2abd[co] = b_e2abd[co] + t_e2bd[idx]
    b_e2ab[co] = b_e2ab[co] + t_e2b[idx]
    t_e2bd[idx] = 0.0
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
        p1padbd[xi] = p1padbd[xi] + (sb * w_e2ad[wi] + w_e2a[wi] * sbd)
        p1padb[xi] = p1padb[xi] + w_e2a[wi] * sb
        w_e2abd[wi] = w_e2abd[wi] + (sb * p1padd[xi] + p1pad[xi] * sbd)
        w_e2ab[wi] = w_e2ab[wi] + p1pad[xi] * sb
    end
    sbd = 0.0
    sb = 0.0
    return nothing
end

function cuda_kernel_unet_hv_88!(c1, hp2, hw2, p1b, p1bd, p1padb, p1padbd, pad, w2, wp2)
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
    p1bd[idx] = p1bd[idx] + p1padbd[yi]
    p1b[idx] = p1b[idx] + p1padb[yi]
    p1padbd[yi] = 0.0
    p1padb[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_89!(n_p1pad, p1padb, p1padbd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_p1pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    p1padbd[i] = 0.0
    p1padb[i] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_90!(a11_stack, a11_stack_d, a12_stack, a12_stack_d, a21_stack, a21_stack_d, a22_stack, a22_stack_d, c1, hw, hw2, m1_stack, m1_stack_d, m2_stack, m2_stack_d, p1b, p1bd, skip1b, skip1bd, w, w2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - c1 * hw2, -1) + 1
        return nothing
    end
    idx = c1 * hw2 + (__tid - 1) * -1
    idxm1 = idx - 1
    ci = div(idxm1, hw2) + 1
    rem = mod(idxm1, hw2)
    oi = div(rem, w2) + 1
    oj = mod(rem, w2) + 1
    i0 = 2oi - 1
    j0 = 2oj - 1
    m1bd = m1bd + (0.5 * (1.0 + sign(m1 - m2))) * p1bd[idx]
    m1b = m1b + (0.5 * (1.0 + sign(m1 - m2))) * p1b[idx]
    m2bd = m2bd + (0.5 * (1.0 + sign(m2 - m1))) * p1bd[idx]
    m2b = m2b + (0.5 * (1.0 + sign(m2 - m1))) * p1b[idx]
    p1bd[idx] = 0.0
    p1b[idx] = 0.0
    m2d = m2_stack_d[(idx - 1) + 1]
    m2 = m2_stack[(idx - 1) + 1]
    a21bd = a21bd + (0.5 * (1.0 + sign(a21 - a22))) * m2bd
    a21b = a21b + (0.5 * (1.0 + sign(a21 - a22))) * m2b
    a22bd = a22bd + (0.5 * (1.0 + sign(a22 - a21))) * m2bd
    a22b = a22b + (0.5 * (1.0 + sign(a22 - a21))) * m2b
    m2bd = 0.0
    m2b = 0.0
    m1d = m1_stack_d[(idx - 1) + 1]
    m1 = m1_stack[(idx - 1) + 1]
    a11bd = a11bd + (0.5 * (1.0 + sign(a11 - a12))) * m1bd
    a11b = a11b + (0.5 * (1.0 + sign(a11 - a12))) * m1b
    a12bd = a12bd + (0.5 * (1.0 + sign(a12 - a11))) * m1bd
    a12b = a12b + (0.5 * (1.0 + sign(a12 - a11))) * m1b
    m1bd = 0.0
    m1b = 0.0
    a22d = a22_stack_d[(idx - 1) + 1]
    a22 = a22_stack[(idx - 1) + 1]
    skip1bd[(ci - 1) * hw + i0 * w + j0 + 1] = skip1bd[(ci - 1) * hw + i0 * w + j0 + 1] + a22bd
    skip1b[(ci - 1) * hw + i0 * w + j0 + 1] = skip1b[(ci - 1) * hw + i0 * w + j0 + 1] + a22b
    a22bd = 0.0
    a22b = 0.0
    a21d = a21_stack_d[(idx - 1) + 1]
    a21 = a21_stack[(idx - 1) + 1]
    skip1bd[(ci - 1) * hw + i0 * w + j0] = skip1bd[(ci - 1) * hw + i0 * w + j0] + a21bd
    skip1b[(ci - 1) * hw + i0 * w + j0] = skip1b[(ci - 1) * hw + i0 * w + j0] + a21b
    a21bd = 0.0
    a21b = 0.0
    a12d = a12_stack_d[(idx - 1) + 1]
    a12 = a12_stack[(idx - 1) + 1]
    skip1bd[(ci - 1) * hw + (i0 - 1) * w + j0 + 1] = skip1bd[(ci - 1) * hw + (i0 - 1) * w + j0 + 1] + a12bd
    skip1b[(ci - 1) * hw + (i0 - 1) * w + j0 + 1] = skip1b[(ci - 1) * hw + (i0 - 1) * w + j0 + 1] + a12b
    a12bd = 0.0
    a12b = 0.0
    a11d = a11_stack_d[(idx - 1) + 1]
    a11 = a11_stack[(idx - 1) + 1]
    skip1bd[(ci - 1) * hw + (i0 - 1) * w + j0] = skip1bd[(ci - 1) * hw + (i0 - 1) * w + j0] + a11bd
    skip1b[(ci - 1) * hw + (i0 - 1) * w + j0] = skip1b[(ci - 1) * hw + (i0 - 1) * w + j0] + a11b
    a11bd = 0.0
    a11b = 0.0
    return nothing
end

function cuda_kernel_unet_hv_91!(n_e1_out, skip1, skip1_stack, skip1_stack_d, skip1b, skip1bd, skip1d, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_e1_out, -1) + 1
        return nothing
    end
    i = n_e1_out + (__tid - 1) * -1
    skip1d[i] = skip1_stack_d[(i - 1) + 1]
    skip1[i] = skip1_stack[(i - 1) + 1]
    skip1bd[i] = (0.5 * (1.0 + sign(skip1[i] - zero_val))) * skip1bd[i]
    skip1b[i] = (0.5 * (1.0 + sign(skip1[i] - zero_val))) * skip1b[i]
    return nothing
end

function cuda_kernel_unet_hv_92!(b_e1bb, b_e1bbd, c1, hp1, hw, kh, khkw, kw, skip1b, skip1bd, t_e1pad, t_e1padb, t_e1padbd, t_e1padd, w, w_e1b, w_e1bb, w_e1bbd, w_e1bd, wp1)
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
    sbd = sbd + skip1bd[idx]
    sb = sb + skip1b[idx]
    b_e1bbd[co] = b_e1bbd[co] + skip1bd[idx]
    b_e1bb[co] = b_e1bb[co] + skip1b[idx]
    skip1bd[idx] = 0.0
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
        t_e1padbd[xi] = t_e1padbd[xi] + (sb * w_e1bd[wi] + w_e1b[wi] * sbd)
        t_e1padb[xi] = t_e1padb[xi] + w_e1b[wi] * sb
        w_e1bbd[wi] = w_e1bbd[wi] + (sb * t_e1padd[xi] + t_e1pad[xi] * sbd)
        w_e1bb[wi] = w_e1bb[wi] + t_e1pad[xi] * sb
    end
    sbd = 0.0
    sb = 0.0
    return nothing
end

function cuda_kernel_unet_hv_93!(c1, hp1, hw, pad, t_e1b, t_e1bd, t_e1padb, t_e1padbd, w, wp1)
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
    t_e1bd[idx] = t_e1bd[idx] + t_e1padbd[yi]
    t_e1b[idx] = t_e1b[idx] + t_e1padb[yi]
    t_e1padbd[yi] = 0.0
    t_e1padb[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_94!(n_e1_midpad, t_e1padb, t_e1padbd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e1_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e1padbd[i] = 0.0
    t_e1padb[i] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_95!(n_e1_mid, t_e1, t_e1_stack, t_e1_stack_d, t_e1b, t_e1bd, t_e1d, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_e1_mid, -1) + 1
        return nothing
    end
    i = n_e1_mid + (__tid - 1) * -1
    t_e1d[i] = t_e1_stack_d[(i - 1) + 1]
    t_e1[i] = t_e1_stack[(i - 1) + 1]
    t_e1bd[i] = (0.5 * (1.0 + sign(t_e1[i] - zero_val))) * t_e1bd[i]
    t_e1b[i] = (0.5 * (1.0 + sign(t_e1[i] - zero_val))) * t_e1b[i]
    return nothing
end

function cuda_kernel_unet_hv_96!(b_e1ab, b_e1abd, c1, c_in, hp1, hw, kh, khkw, kw, t_e1b, t_e1bd, w, w_e1a, w_e1ab, w_e1abd, w_e1ad, wp1, xpad0, xpad0b, xpad0bd, xpad0d)
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
    sbd = sbd + t_e1bd[idx]
    sb = sb + t_e1b[idx]
    b_e1abd[co] = b_e1abd[co] + t_e1bd[idx]
    b_e1ab[co] = b_e1ab[co] + t_e1b[idx]
    t_e1bd[idx] = 0.0
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
        xpad0bd[xi] = xpad0bd[xi] + (sb * w_e1ad[wi] + w_e1a[wi] * sbd)
        xpad0b[xi] = xpad0b[xi] + w_e1a[wi] * sb
        w_e1abd[wi] = w_e1abd[wi] + (sb * xpad0d[xi] + xpad0[xi] * sbd)
        w_e1ab[wi] = w_e1ab[wi] + xpad0[xi] * sb
    end
    sbd = 0.0
    sb = 0.0
    return nothing
end

function cuda_kernel_unet_hv_97!(c_in, hp1, hw, pad, w, wp1, xb, xbd, xpad0b, xpad0bd)
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
    xbd[idx] = xbd[idx] + xpad0bd[yi]
    xb[idx] = xb[idx] + xpad0b[yi]
    xpad0bd[yi] = 0.0
    xpad0b[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_hv_98!(n_xpad0, xpad0b, xpad0bd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_xpad0 - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    xpad0bd[i] = 0.0
    xpad0b[i] = 0.0
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

function initstacks_unet_b_cuda(c1, c2, hw2, hw4, n_b_mid, n_b_out, n_d1_mid, n_d1_out, n_d2_mid, n_d2_out, n_e1_mid, n_e1_out, n_e2_mid, n_e2_out)
    t_e1_stack = CuArray{Float64}(undef, div(n_e1_mid - 1, 1) + 1)
    skip1_stack = CuArray{Float64}(undef, div(n_e1_out - 1, 1) + 1)
    a11_stack = CuArray{Float64}(undef, ((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1)
    a12_stack = CuArray{Float64}(undef, ((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1)
    a21_stack = CuArray{Float64}(undef, ((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1)
    a22_stack = CuArray{Float64}(undef, ((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1)
    m1_stack = CuArray{Float64}(undef, ((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1)
    m2_stack = CuArray{Float64}(undef, ((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1)
    t_e2_stack = CuArray{Float64}(undef, div(n_e2_mid - 1, 1) + 1)
    skip2_stack = CuArray{Float64}(undef, div(n_e2_out - 1, 1) + 1)
    t_b_stack = CuArray{Float64}(undef, div(n_b_mid - 1, 1) + 1)
    bott_stack = CuArray{Float64}(undef, div(n_b_out - 1, 1) + 1)
    t_d2_stack = CuArray{Float64}(undef, div(n_d2_mid - 1, 1) + 1)
    dec2out_stack = CuArray{Float64}(undef, div(n_d2_out - 1, 1) + 1)
    t_d1_stack = CuArray{Float64}(undef, div(n_d1_mid - 1, 1) + 1)
    dec1out_stack = CuArray{Float64}(undef, div(n_d1_out - 1, 1) + 1)
    return (t_e1_stack, skip1_stack, a11_stack, a12_stack, a21_stack, a22_stack, m1_stack, m2_stack, t_e2_stack, skip2_stack, t_b_stack, bott_stack, t_d2_stack, dec2out_stack, t_d1_stack, dec1out_stack)
end

function unet_hv_cuda(x, xb, h, w, c_in, c1, c2, c3, c_out, w_e1a, w_e1ab, b_e1a, b_e1ab, w_e1b, w_e1bb, b_e1b, b_e1bb, w_e2a, w_e2ab, b_e2a, b_e2ab, w_e2b, w_e2bb, b_e2b, b_e2bb, w_ba, w_bab, b_ba, b_bab, w_bb, w_bbb, b_bb, b_bbb, w_d2a, w_d2ab, b_d2a, b_d2ab, w_d2b, w_d2bb, b_d2b, b_d2bb, w_d1a, w_d1ab, b_d1a, b_d1ab, w_d1b, w_d1bb, b_d1b, b_d1bb, w_out, w_outb, b_out, b_outb, xpad0, xpad0b, t_e1, t_e1b, t_e1pad, t_e1padb, skip1, skip1b, p1, p1b, p1pad, p1padb, t_e2, t_e2b, t_e2pad, t_e2padb, skip2, skip2b, p2, p2b, p2pad, p2padb, t_b, t_bb, t_bpad, t_bpadb, bott, bottb, u2, u2b, cat2, cat2b, cat2pad, cat2padb, t_d2, t_d2b, t_d2pad, t_d2padb, dec2out, dec2outb, u1, u1b, cat1, cat1b, cat1pad, cat1padb, t_d1, t_d1b, t_d1pad, t_d1padb, dec1out, dec1outb, y, yb, xd, xbd, w_e1ad, w_e1abd, b_e1ad, b_e1abd, w_e1bd, w_e1bbd, b_e1bd, b_e1bbd, w_e2ad, w_e2abd, b_e2ad, b_e2abd, w_e2bd, w_e2bbd, b_e2bd, b_e2bbd, w_bad, w_babd, b_bad, b_babd, w_bbd, w_bbbd, b_bbd, b_bbbd, w_d2ad, w_d2abd, b_d2ad, b_d2abd, w_d2bd, w_d2bbd, b_d2bd, b_d2bbd, w_d1ad, w_d1abd, b_d1ad, b_d1abd, w_d1bd, w_d1bbd, b_d1bd, b_d1bbd, w_outd, w_outbd, b_outd, b_outbd, xpad0d, xpad0bd, t_e1d, t_e1bd, t_e1padd, t_e1padbd, skip1d, skip1bd, p1d, p1bd, p1padd, p1padbd, t_e2d, t_e2bd, t_e2padd, t_e2padbd, skip2d, skip2bd, p2d, p2bd, p2padd, p2padbd, t_bd, t_bbd, t_bpadd, t_bpadbd, bottd, bottbd, u2d, u2bd, cat2d, cat2bd, cat2padd, cat2padbd, t_d2d, t_d2bd, t_d2padd, t_d2padbd, dec2outd, dec2outbd, u1d, u1bd, cat1d, cat1bd, cat1padd, cat1padbd, t_d1d, t_d1bd, t_d1padd, t_d1padbd, dec1outd, dec1outbd, yd, ybd, t_e1_stack, skip1_stack, a11_stack, a12_stack, a21_stack, a22_stack, m1_stack, m2_stack, t_e2_stack, skip2_stack, t_b_stack, bott_stack, t_d2_stack, dec2out_stack, t_d1_stack, dec1out_stack)
    nthread_per_block = 256
    t_e1_stack_d = CuArray{Float64}(undef, div(n_e1_mid - 1, 1) + 1)
    skip1_stack_d = CuArray{Float64}(undef, div(n_e1_out - 1, 1) + 1)
    a11_stack_d = CuArray{Float64}(undef, ((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1)
    a12_stack_d = CuArray{Float64}(undef, ((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1)
    a21_stack_d = CuArray{Float64}(undef, ((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1)
    a22_stack_d = CuArray{Float64}(undef, ((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1)
    m1_stack_d = CuArray{Float64}(undef, ((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1)
    m2_stack_d = CuArray{Float64}(undef, ((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1)
    t_e2_stack_d = CuArray{Float64}(undef, div(n_e2_mid - 1, 1) + 1)
    skip2_stack_d = CuArray{Float64}(undef, div(n_e2_out - 1, 1) + 1)
    t_b_stack_d = CuArray{Float64}(undef, div(n_b_mid - 1, 1) + 1)
    bott_stack_d = CuArray{Float64}(undef, div(n_b_out - 1, 1) + 1)
    t_d2_stack_d = CuArray{Float64}(undef, div(n_d2_mid - 1, 1) + 1)
    dec2out_stack_d = CuArray{Float64}(undef, div(n_d2_out - 1, 1) + 1)
    t_d1_stack_d = CuArray{Float64}(undef, div(n_d1_mid - 1, 1) + 1)
    dec1out_stack_d = CuArray{Float64}(undef, div(n_d1_out - 1, 1) + 1)
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
    a11d = 0.0
    a11bd = 0.0
    a12d = 0.0
    a12bd = 0.0
    a21d = 0.0
    a21bd = 0.0
    a22d = 0.0
    a22bd = 0.0
    m1d = 0.0
    m1bd = 0.0
    m2d = 0.0
    m2bd = 0.0
    sd = 0.0
    sbd = 0.0
    zero_vald = 0.0
    zero_valbd = 0.0
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
    zero_vald = 0.0
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
    @cuda threads = nthread_per_block blocks = cld(div(n_xpad0 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_1!(n_xpad0, xpad0, xpad0d, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(c_in * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_2!(c_in, hp1, hw, pad, w, wp1, x, xd, xpad0, xpad0d)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_3!(b_e1a, b_e1ad, c1, c_in, hp1, hw, kh, khkw, kw, t_e1, t_e1d, w, w_e1a, w_e1ad, wp1, xpad0, xpad0d)
    @cuda threads = nthread_per_block blocks = cld(div(n_e1_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_4!(n_e1_mid, t_e1, t_e1_stack, t_e1_stack_d, t_e1d, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(n_e1_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_5!(n_e1_midpad, t_e1pad, t_e1padd, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_6!(c1, hp1, hw, pad, t_e1, t_e1d, t_e1pad, t_e1padd, w, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_7!(b_e1b, b_e1bd, c1, hp1, hw, kh, khkw, kw, skip1, skip1d, t_e1pad, t_e1padd, w, w_e1b, w_e1bd, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(n_e1_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_8!(n_e1_out, skip1, skip1_stack, skip1_stack_d, skip1d, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_9!(a11_stack, a11_stack_d, a12_stack, a12_stack_d, a21_stack, a21_stack_d, a22_stack, a22_stack_d, c1, hw, hw2, m1_stack, m1_stack_d, m2_stack, m2_stack_d, p1, p1d, skip1, skip1d, w, w2)
    @cuda threads = nthread_per_block blocks = cld(div(n_p1pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_10!(n_p1pad, p1pad, p1padd, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_11!(c1, hp2, hw2, p1, p1d, p1pad, p1padd, pad, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_12!(b_e2a, b_e2ad, c1, c2, hp2, hw2, kh, khkw, kw, p1pad, p1padd, t_e2, t_e2d, w2, w_e2a, w_e2ad, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_e2_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_13!(n_e2_mid, t_e2, t_e2_stack, t_e2_stack_d, t_e2d, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(n_e2_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_14!(n_e2_midpad, t_e2pad, t_e2padd, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_15!(c2, hp2, hw2, pad, t_e2, t_e2d, t_e2pad, t_e2padd, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_16!(b_e2b, b_e2bd, c2, hp2, hw2, kh, khkw, kw, skip2, skip2d, t_e2pad, t_e2padd, w2, w_e2b, w_e2bd, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_e2_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_17!(n_e2_out, skip2, skip2_stack, skip2_stack_d, skip2d, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_18!(a11_stack, a11_stack_d, a12_stack, a12_stack_d, a21_stack, a21_stack_d, a22_stack, a22_stack_d, c1, c2, hw2, hw4, m1_stack, m1_stack_d, m2_stack, m2_stack_d, p2, p2d, skip2, skip2d, w2, w4)
    @cuda threads = nthread_per_block blocks = cld(div(n_p2pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_19!(n_p2pad, p2pad, p2padd, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_20!(c2, hp4, hw4, p2, p2d, p2pad, p2padd, pad, w4, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_21!(b_ba, b_bad, c2, c3, hp4, hw4, kh, khkw, kw, p2pad, p2padd, t_b, t_bd, w4, w_ba, w_bad, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(n_b_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_22!(n_b_mid, t_b, t_b_stack, t_b_stack_d, t_bd, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(n_b_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_23!(n_b_midpad, t_bpad, t_bpadd, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_24!(c3, hp4, hw4, pad, t_b, t_bd, t_bpad, t_bpadd, w4, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_25!(b_bb, b_bbd, bott, bottd, c3, hp4, hw4, kh, khkw, kw, t_bpad, t_bpadd, w4, w_bb, w_bbd, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(n_b_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_26!(bott, bott_stack, bott_stack_d, bottd, n_b_out, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_27!(bott, bottd, c3, hw2, hw4, scale, u2, u2d, w2, w4)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_28!(c3, cat2, cat2d, hw2, u2, u2d)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_29!(c2, c3, cat2, cat2d, hw2, skip2, skip2d)
    @cuda threads = nthread_per_block blocks = cld(div(n_cat2pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_30!(cat2pad, cat2padd, n_cat2pad, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(c32 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_31!(c32, cat2, cat2d, cat2pad, cat2padd, hp2, hw2, pad, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_32!(b_d2a, b_d2ad, c2, c32, cat2pad, cat2padd, hp2, hw2, kh, khkw, kw, t_d2, t_d2d, w2, w_d2a, w_d2ad, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_d2_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_33!(n_d2_mid, t_d2, t_d2_stack, t_d2_stack_d, t_d2d, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(n_d2_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_34!(n_d2_midpad, t_d2pad, t_d2padd, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_35!(c2, hp2, hw2, pad, t_d2, t_d2d, t_d2pad, t_d2padd, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_36!(b_d2b, b_d2bd, c2, dec2out, dec2outd, hp2, hw2, kh, khkw, kw, t_d2pad, t_d2padd, w2, w_d2b, w_d2bd, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_d2_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_37!(dec2out, dec2out_stack, dec2out_stack_d, dec2outd, n_d2_out, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_38!(c2, dec2out, dec2outd, hw, hw2, scale, u1, u1d, w, w2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_39!(c2, cat1, cat1d, hw, u1, u1d)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_40!(c1, c2, cat1, cat1d, hw, skip1, skip1d)
    @cuda threads = nthread_per_block blocks = cld(div(n_cat1pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_41!(cat1pad, cat1padd, n_cat1pad, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(c21 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_42!(c21, cat1, cat1d, cat1pad, cat1padd, hp1, hw, pad, w, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_43!(b_d1a, b_d1ad, c1, c21, cat1pad, cat1padd, hp1, hw, kh, khkw, kw, t_d1, t_d1d, w, w_d1a, w_d1ad, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(n_d1_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_44!(n_d1_mid, t_d1, t_d1_stack, t_d1_stack_d, t_d1d, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(n_d1_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_45!(n_d1_midpad, t_d1pad, t_d1padd, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_46!(c1, hp1, hw, pad, t_d1, t_d1d, t_d1pad, t_d1padd, w, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_47!(b_d1b, b_d1bd, c1, dec1out, dec1outd, hp1, hw, kh, khkw, kw, t_d1pad, t_d1padd, w, w_d1b, w_d1bd, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(n_d1_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_48!(dec1out, dec1out_stack, dec1out_stack_d, dec1outd, n_d1_out, zero_val, zero_vald)
    @cuda threads = nthread_per_block blocks = cld(div(c_out * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_49!(b_out, b_outd, c1, c_out, dec1out, dec1outd, hw, kh_out, khkw_out, kw_out, w, w_out, w_outd, y, yd)
    a11_stack_d[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1] = a11d
    a11_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1] = a11
    a12_stack_d[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1] = a12d
    a12_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1] = a12
    a21_stack_d[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1] = a21d
    a21_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1] = a21
    a22_stack_d[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1] = a22d
    a22_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1] = a22
    m1_stack_d[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1] = m1d
    m1_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1] = m1
    m2_stack_d[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1] = m2d
    m2_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1] = m2
    a11d = a11_stack_d[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1]
    a11 = a11_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1]
    a12d = a12_stack_d[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1]
    a12 = a12_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1]
    a21d = a21_stack_d[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1]
    a21 = a21_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1]
    a22d = a22_stack_d[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1]
    a22 = a22_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1]
    m1d = m1_stack_d[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1]
    m1 = m1_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1]
    m2d = m2_stack_d[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1]
    m2 = m2_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1]
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
    @cuda threads = nthread_per_block blocks = cld(div(c_out * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_50!(b_outb, b_outbd, c1, c_out, dec1out, dec1outb, dec1outbd, dec1outd, hw, kh_out, khkw_out, kw_out, w, w_out, w_outb, w_outbd, w_outd, yb, ybd)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_d1_out, -1) + 1, nthread_per_block) cuda_kernel_unet_hv_51!(dec1out, dec1out_stack, dec1out_stack_d, dec1outb, dec1outbd, dec1outd, n_d1_out, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_52!(b_d1bb, b_d1bbd, c1, dec1outb, dec1outbd, hp1, hw, kh, khkw, kw, t_d1pad, t_d1padb, t_d1padbd, t_d1padd, w, w_d1b, w_d1bb, w_d1bbd, w_d1bd, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_53!(c1, hp1, hw, pad, t_d1b, t_d1bd, t_d1padb, t_d1padbd, w, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(n_d1_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_54!(n_d1_midpad, t_d1padb, t_d1padbd)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_d1_mid, -1) + 1, nthread_per_block) cuda_kernel_unet_hv_55!(n_d1_mid, t_d1, t_d1_stack, t_d1_stack_d, t_d1b, t_d1bd, t_d1d, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_56!(b_d1ab, b_d1abd, c1, c21, cat1pad, cat1padb, cat1padbd, cat1padd, hp1, hw, kh, khkw, kw, t_d1b, t_d1bd, w, w_d1a, w_d1ab, w_d1abd, w_d1ad, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(c21 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_57!(c21, cat1b, cat1bd, cat1padb, cat1padbd, hp1, hw, pad, w, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(n_cat1pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_58!(cat1padb, cat1padbd, n_cat1pad)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_59!(c1, c2, cat1b, cat1bd, hw, skip1b, skip1bd)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_60!(c2, cat1b, cat1bd, hw, u1b, u1bd)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_61!(c2, dec2outb, dec2outbd, hw, hw2, scale, u1b, u1bd, w, w2)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_d2_out, -1) + 1, nthread_per_block) cuda_kernel_unet_hv_62!(dec2out, dec2out_stack, dec2out_stack_d, dec2outb, dec2outbd, dec2outd, n_d2_out, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_63!(b_d2bb, b_d2bbd, c2, dec2outb, dec2outbd, hp2, hw2, kh, khkw, kw, t_d2pad, t_d2padb, t_d2padbd, t_d2padd, w2, w_d2b, w_d2bb, w_d2bbd, w_d2bd, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_64!(c2, hp2, hw2, pad, t_d2b, t_d2bd, t_d2padb, t_d2padbd, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_d2_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_65!(n_d2_midpad, t_d2padb, t_d2padbd)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_d2_mid, -1) + 1, nthread_per_block) cuda_kernel_unet_hv_66!(n_d2_mid, t_d2, t_d2_stack, t_d2_stack_d, t_d2b, t_d2bd, t_d2d, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_67!(b_d2ab, b_d2abd, c2, c32, cat2pad, cat2padb, cat2padbd, cat2padd, hp2, hw2, kh, khkw, kw, t_d2b, t_d2bd, w2, w_d2a, w_d2ab, w_d2abd, w_d2ad, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c32 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_68!(c32, cat2b, cat2bd, cat2padb, cat2padbd, hp2, hw2, pad, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_cat2pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_69!(cat2padb, cat2padbd, n_cat2pad)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_70!(c2, c3, cat2b, cat2bd, hw2, skip2b, skip2bd)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_71!(c3, cat2b, cat2bd, hw2, u2b, u2bd)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_72!(bottb, bottbd, c3, hw2, hw4, scale, u2b, u2bd, w2, w4)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_b_out, -1) + 1, nthread_per_block) cuda_kernel_unet_hv_73!(bott, bott_stack, bott_stack_d, bottb, bottbd, bottd, n_b_out, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_74!(b_bbb, b_bbbd, bottb, bottbd, c3, hp4, hw4, kh, khkw, kw, t_bpad, t_bpadb, t_bpadbd, t_bpadd, w4, w_bb, w_bbb, w_bbbd, w_bbd, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_75!(c3, hp4, hw4, pad, t_bb, t_bbd, t_bpadb, t_bpadbd, w4, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(n_b_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_76!(n_b_midpad, t_bpadb, t_bpadbd)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_b_mid, -1) + 1, nthread_per_block) cuda_kernel_unet_hv_77!(n_b_mid, t_b, t_b_stack, t_b_stack_d, t_bb, t_bbd, t_bd, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_78!(b_bab, b_babd, c2, c3, hp4, hw4, kh, khkw, kw, p2pad, p2padb, p2padbd, p2padd, t_bb, t_bbd, w4, w_ba, w_bab, w_babd, w_bad, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_79!(c2, hp4, hw4, p2b, p2bd, p2padb, p2padbd, pad, w4, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(n_p2pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_80!(n_p2pad, p2padb, p2padbd)
    @cuda threads = nthread_per_block blocks = cld(div(1 - c2 * hw4, -1) + 1, nthread_per_block) cuda_kernel_unet_hv_81!(a11_stack, a11_stack_d, a12_stack, a12_stack_d, a21_stack, a21_stack_d, a22_stack, a22_stack_d, c1, c2, hw2, hw4, m1_stack, m1_stack_d, m2_stack, m2_stack_d, p2b, p2bd, skip2b, skip2bd, w2, w4)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_e2_out, -1) + 1, nthread_per_block) cuda_kernel_unet_hv_82!(n_e2_out, skip2, skip2_stack, skip2_stack_d, skip2b, skip2bd, skip2d, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_83!(b_e2bb, b_e2bbd, c2, hp2, hw2, kh, khkw, kw, skip2b, skip2bd, t_e2pad, t_e2padb, t_e2padbd, t_e2padd, w2, w_e2b, w_e2bb, w_e2bbd, w_e2bd, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_84!(c2, hp2, hw2, pad, t_e2b, t_e2bd, t_e2padb, t_e2padbd, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_e2_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_85!(n_e2_midpad, t_e2padb, t_e2padbd)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_e2_mid, -1) + 1, nthread_per_block) cuda_kernel_unet_hv_86!(n_e2_mid, t_e2, t_e2_stack, t_e2_stack_d, t_e2b, t_e2bd, t_e2d, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_87!(b_e2ab, b_e2abd, c1, c2, hp2, hw2, kh, khkw, kw, p1pad, p1padb, p1padbd, p1padd, t_e2b, t_e2bd, w2, w_e2a, w_e2ab, w_e2abd, w_e2ad, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_88!(c1, hp2, hw2, p1b, p1bd, p1padb, p1padbd, pad, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_p1pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_89!(n_p1pad, p1padb, p1padbd)
    @cuda threads = nthread_per_block blocks = cld(div(1 - c1 * hw2, -1) + 1, nthread_per_block) cuda_kernel_unet_hv_90!(a11_stack, a11_stack_d, a12_stack, a12_stack_d, a21_stack, a21_stack_d, a22_stack, a22_stack_d, c1, hw, hw2, m1_stack, m1_stack_d, m2_stack, m2_stack_d, p1b, p1bd, skip1b, skip1bd, w, w2)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_e1_out, -1) + 1, nthread_per_block) cuda_kernel_unet_hv_91!(n_e1_out, skip1, skip1_stack, skip1_stack_d, skip1b, skip1bd, skip1d, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_92!(b_e1bb, b_e1bbd, c1, hp1, hw, kh, khkw, kw, skip1b, skip1bd, t_e1pad, t_e1padb, t_e1padbd, t_e1padd, w, w_e1b, w_e1bb, w_e1bbd, w_e1bd, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_93!(c1, hp1, hw, pad, t_e1b, t_e1bd, t_e1padb, t_e1padbd, w, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(n_e1_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_94!(n_e1_midpad, t_e1padb, t_e1padbd)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_e1_mid, -1) + 1, nthread_per_block) cuda_kernel_unet_hv_95!(n_e1_mid, t_e1, t_e1_stack, t_e1_stack_d, t_e1b, t_e1bd, t_e1d, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_96!(b_e1ab, b_e1abd, c1, c_in, hp1, hw, kh, khkw, kw, t_e1b, t_e1bd, w, w_e1a, w_e1ab, w_e1abd, w_e1ad, wp1, xpad0, xpad0b, xpad0bd, xpad0d)
    @cuda threads = nthread_per_block blocks = cld(div(c_in * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_97!(c_in, hp1, hw, pad, w, wp1, xb, xbd, xpad0b, xpad0bd)
    @cuda threads = nthread_per_block blocks = cld(div(n_xpad0 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_hv_98!(n_xpad0, xpad0b, xpad0bd)
    zero_valbd = 0.0
    zero_valb = 0.0
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
