import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_unet_b_1!(n_xpad0, xpad0, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_xpad0 - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    xpad0[i] = zero_val
    return nothing
end

function cuda_kernel_unet_b_2!(c_in, hp1, hw, pad, w, wp1, x, xpad0)
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

function cuda_kernel_unet_b_3!(b_e1a, c1, c_in, hp1, hw, kh, khkw, kw, t_e1, w, w_e1a, wp1, xpad0)
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

function cuda_kernel_unet_b_4!(n_e1_mid, t_e1, t_e1_stack, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e1_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e1_stack[(i - 1) + 1] = t_e1[i]
    t_e1[i] = max(t_e1[i], zero_val)
    return nothing
end

function cuda_kernel_unet_b_5!(n_e1_midpad, t_e1pad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e1_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e1pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_b_6!(c1, hp1, hw, pad, t_e1, t_e1pad, w, wp1)
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

function cuda_kernel_unet_b_7!(b_e1b, c1, hp1, hw, kh, khkw, kw, skip1, t_e1pad, w, w_e1b, wp1)
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

function cuda_kernel_unet_b_8!(n_e1_out, skip1, skip1_stack, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e1_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    skip1_stack[(i - 1) + 1] = skip1[i]
    skip1[i] = max(skip1[i], zero_val)
    return nothing
end

function cuda_kernel_unet_b_9!(a11_stack, a12_stack, a21_stack, a22_stack, c1, hw, hw2, m1_stack, m2_stack, p1, skip1, w, w2)
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
    a11_stack[(idx - 1) + 1] = a11
    a11 = skip1[(ci - 1) * hw + (i0 - 1) * w + j0]
    a12_stack[(idx - 1) + 1] = a12
    a12 = skip1[(ci - 1) * hw + (i0 - 1) * w + j0 + 1]
    a21_stack[(idx - 1) + 1] = a21
    a21 = skip1[(ci - 1) * hw + i0 * w + j0]
    a22_stack[(idx - 1) + 1] = a22
    a22 = skip1[(ci - 1) * hw + i0 * w + j0 + 1]
    m1_stack[(idx - 1) + 1] = m1
    m1 = max(a11, a12)
    m2_stack[(idx - 1) + 1] = m2
    m2 = max(a21, a22)
    p1[idx] = max(m1, m2)
    return nothing
end

function cuda_kernel_unet_b_10!(n_p1pad, p1pad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_p1pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    p1pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_b_11!(c1, hp2, hw2, p1, p1pad, pad, w2, wp2)
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

function cuda_kernel_unet_b_12!(b_e2a, c1, c2, hp2, hw2, kh, khkw, kw, p1pad, t_e2, w2, w_e2a, wp2)
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

function cuda_kernel_unet_b_13!(n_e2_mid, t_e2, t_e2_stack, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e2_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e2_stack[(i - 1) + 1] = t_e2[i]
    t_e2[i] = max(t_e2[i], zero_val)
    return nothing
end

function cuda_kernel_unet_b_14!(n_e2_midpad, t_e2pad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e2_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e2pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_b_15!(c2, hp2, hw2, pad, t_e2, t_e2pad, w2, wp2)
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

function cuda_kernel_unet_b_16!(b_e2b, c2, hp2, hw2, kh, khkw, kw, skip2, t_e2pad, w2, w_e2b, wp2)
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

function cuda_kernel_unet_b_17!(n_e2_out, skip2, skip2_stack, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e2_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    skip2_stack[(i - 1) + 1] = skip2[i]
    skip2[i] = max(skip2[i], zero_val)
    return nothing
end

function cuda_kernel_unet_b_18!(a11_stack, a12_stack, a21_stack, a22_stack, c1, c2, hw2, hw4, m1_stack, m2_stack, p2, skip2, w2, w4)
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
    a11_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)] = a11
    a11 = skip2[(ci - 1) * hw2 + (i0 - 1) * w2 + j0]
    a12_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)] = a12
    a12 = skip2[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1]
    a21_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)] = a21
    a21 = skip2[(ci - 1) * hw2 + i0 * w2 + j0]
    a22_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)] = a22
    a22 = skip2[(ci - 1) * hw2 + i0 * w2 + j0 + 1]
    m1_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)] = m1
    m1 = max(a11, a12)
    m2_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)] = m2
    m2 = max(a21, a22)
    p2[idx] = max(m1, m2)
    return nothing
end

function cuda_kernel_unet_b_19!(n_p2pad, p2pad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_p2pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    p2pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_b_20!(c2, hp4, hw4, p2, p2pad, pad, w4, wp4)
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

function cuda_kernel_unet_b_21!(b_ba, c2, c3, hp4, hw4, kh, khkw, kw, p2pad, t_b, w4, w_ba, wp4)
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

function cuda_kernel_unet_b_22!(n_b_mid, t_b, t_b_stack, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_b_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_b_stack[(i - 1) + 1] = t_b[i]
    t_b[i] = max(t_b[i], zero_val)
    return nothing
end

function cuda_kernel_unet_b_23!(n_b_midpad, t_bpad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_b_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_bpad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_b_24!(c3, hp4, hw4, pad, t_b, t_bpad, w4, wp4)
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

function cuda_kernel_unet_b_25!(b_bb, bott, c3, hp4, hw4, kh, khkw, kw, t_bpad, w4, w_bb, wp4)
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

function cuda_kernel_unet_b_26!(bott, bott_stack, n_b_out, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_b_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    bott_stack[(i - 1) + 1] = bott[i]
    bott[i] = max(bott[i], zero_val)
    return nothing
end

function cuda_kernel_unet_b_27!(bott, c3, hw2, hw4, scale, u2, w2, w4)
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

function cuda_kernel_unet_b_28!(c3, cat2, hw2, u2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c3 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    cat2[idx] = u2[idx]
    return nothing
end

function cuda_kernel_unet_b_29!(c2, c3, cat2, hw2, skip2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    cat2[c3 * hw2 + idx] = skip2[idx]
    return nothing
end

function cuda_kernel_unet_b_30!(cat2pad, n_cat2pad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_cat2pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    cat2pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_b_31!(c32, cat2, cat2pad, hp2, hw2, pad, w2, wp2)
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

function cuda_kernel_unet_b_32!(b_d2a, c2, c32, cat2pad, hp2, hw2, kh, khkw, kw, t_d2, w2, w_d2a, wp2)
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

function cuda_kernel_unet_b_33!(n_d2_mid, t_d2, t_d2_stack, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d2_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d2_stack[(i - 1) + 1] = t_d2[i]
    t_d2[i] = max(t_d2[i], zero_val)
    return nothing
end

function cuda_kernel_unet_b_34!(n_d2_midpad, t_d2pad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d2_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d2pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_b_35!(c2, hp2, hw2, pad, t_d2, t_d2pad, w2, wp2)
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

function cuda_kernel_unet_b_36!(b_d2b, c2, dec2out, hp2, hw2, kh, khkw, kw, t_d2pad, w2, w_d2b, wp2)
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

function cuda_kernel_unet_b_37!(dec2out, dec2out_stack, n_d2_out, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d2_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    dec2out_stack[(i - 1) + 1] = dec2out[i]
    dec2out[i] = max(dec2out[i], zero_val)
    return nothing
end

function cuda_kernel_unet_b_38!(c2, dec2out, hw, hw2, scale, u1, w, w2)
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

function cuda_kernel_unet_b_39!(c2, cat1, hw, u1)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    cat1[idx] = u1[idx]
    return nothing
end

function cuda_kernel_unet_b_40!(c1, c2, cat1, hw, skip1)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    cat1[c2 * hw + idx] = skip1[idx]
    return nothing
end

function cuda_kernel_unet_b_41!(cat1pad, n_cat1pad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_cat1pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    cat1pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_b_42!(c21, cat1, cat1pad, hp1, hw, pad, w, wp1)
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

function cuda_kernel_unet_b_43!(b_d1a, c1, c21, cat1pad, hp1, hw, kh, khkw, kw, t_d1, w, w_d1a, wp1)
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

function cuda_kernel_unet_b_44!(n_d1_mid, t_d1, t_d1_stack, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d1_mid - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d1_stack[(i - 1) + 1] = t_d1[i]
    t_d1[i] = max(t_d1[i], zero_val)
    return nothing
end

function cuda_kernel_unet_b_45!(n_d1_midpad, t_d1pad, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d1_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d1pad[i] = zero_val
    return nothing
end

function cuda_kernel_unet_b_46!(c1, hp1, hw, pad, t_d1, t_d1pad, w, wp1)
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

function cuda_kernel_unet_b_47!(b_d1b, c1, dec1out, hp1, hw, kh, khkw, kw, t_d1pad, w, w_d1b, wp1)
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

function cuda_kernel_unet_b_48!(dec1out, dec1out_stack, n_d1_out, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d1_out - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    dec1out_stack[(i - 1) + 1] = dec1out[i]
    dec1out[i] = max(dec1out[i], zero_val)
    return nothing
end

function cuda_kernel_unet_b_49!(b_out, c1, c_out, dec1out, hw, kh_out, khkw_out, kw_out, w, w_out, y)
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

function cuda_kernel_unet_b_50!(b_outb, c1, c_out, dec1out, dec1outb, hw, kh_out, khkw_out, kw_out, w, w_out, w_outb, yb)
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
    return nothing
end

function cuda_kernel_unet_b_51!(dec1out, dec1out_stack, dec1outb, n_d1_out, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_d1_out, -1) + 1
        return nothing
    end
    i = n_d1_out + (__tid - 1) * -1
    dec1out[i] = dec1out_stack[(i - 1) + 1]
    dec1outb[i] = (0.5 * (1.0 + sign(dec1out[i] - zero_val))) * dec1outb[i]
    return nothing
end

function cuda_kernel_unet_b_52!(b_d1bb, c1, dec1outb, hp1, hw, kh, khkw, kw, t_d1pad, t_d1padb, w, w_d1b, w_d1bb, wp1)
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
    return nothing
end

function cuda_kernel_unet_b_53!(c1, hp1, hw, pad, t_d1b, t_d1padb, w, wp1)
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
    t_d1b[idx] = t_d1b[idx] + t_d1padb[yi]
    t_d1padb[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_b_54!(n_d1_midpad, t_d1padb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d1_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d1padb[i] = 0.0
    return nothing
end

function cuda_kernel_unet_b_55!(n_d1_mid, t_d1, t_d1_stack, t_d1b, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_d1_mid, -1) + 1
        return nothing
    end
    i = n_d1_mid + (__tid - 1) * -1
    t_d1[i] = t_d1_stack[(i - 1) + 1]
    t_d1b[i] = (0.5 * (1.0 + sign(t_d1[i] - zero_val))) * t_d1b[i]
    return nothing
end

function cuda_kernel_unet_b_56!(b_d1ab, c1, c21, cat1pad, cat1padb, hp1, hw, kh, khkw, kw, t_d1b, w, w_d1a, w_d1ab, wp1)
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
    return nothing
end

function cuda_kernel_unet_b_57!(c21, cat1b, cat1padb, hp1, hw, pad, w, wp1)
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
    cat1b[idx] = cat1b[idx] + cat1padb[yi]
    cat1padb[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_b_58!(cat1padb, n_cat1pad)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_cat1pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    cat1padb[i] = 0.0
    return nothing
end

function cuda_kernel_unet_b_59!(c1, c2, cat1b, hw, skip1b)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c1 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    skip1b[idx] = skip1b[idx] + cat1b[c2 * hw + idx]
    cat1b[c2 * hw + idx] = 0.0
    return nothing
end

function cuda_kernel_unet_b_60!(c2, cat1b, hw, u1b)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    u1b[idx] = u1b[idx] + cat1b[idx]
    cat1b[idx] = 0.0
    return nothing
end

function cuda_kernel_unet_b_61!(c2, dec2outb, hw, hw2, scale, u1b, w, w2)
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
    dec2outb[xi] = dec2outb[xi] + u1b[idx]
    u1b[idx] = 0.0
    return nothing
end

function cuda_kernel_unet_b_62!(dec2out, dec2out_stack, dec2outb, n_d2_out, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_d2_out, -1) + 1
        return nothing
    end
    i = n_d2_out + (__tid - 1) * -1
    dec2out[i] = dec2out_stack[(i - 1) + 1]
    dec2outb[i] = (0.5 * (1.0 + sign(dec2out[i] - zero_val))) * dec2outb[i]
    return nothing
end

function cuda_kernel_unet_b_63!(b_d2bb, c2, dec2outb, hp2, hw2, kh, khkw, kw, t_d2pad, t_d2padb, w2, w_d2b, w_d2bb, wp2)
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
    return nothing
end

function cuda_kernel_unet_b_64!(c2, hp2, hw2, pad, t_d2b, t_d2padb, w2, wp2)
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
    t_d2b[idx] = t_d2b[idx] + t_d2padb[yi]
    t_d2padb[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_b_65!(n_d2_midpad, t_d2padb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_d2_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_d2padb[i] = 0.0
    return nothing
end

function cuda_kernel_unet_b_66!(n_d2_mid, t_d2, t_d2_stack, t_d2b, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_d2_mid, -1) + 1
        return nothing
    end
    i = n_d2_mid + (__tid - 1) * -1
    t_d2[i] = t_d2_stack[(i - 1) + 1]
    t_d2b[i] = (0.5 * (1.0 + sign(t_d2[i] - zero_val))) * t_d2b[i]
    return nothing
end

function cuda_kernel_unet_b_67!(b_d2ab, c2, c32, cat2pad, cat2padb, hp2, hw2, kh, khkw, kw, t_d2b, w2, w_d2a, w_d2ab, wp2)
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
    return nothing
end

function cuda_kernel_unet_b_68!(c32, cat2b, cat2padb, hp2, hw2, pad, w2, wp2)
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
    cat2b[idx] = cat2b[idx] + cat2padb[yi]
    cat2padb[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_b_69!(cat2padb, n_cat2pad)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_cat2pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    cat2padb[i] = 0.0
    return nothing
end

function cuda_kernel_unet_b_70!(c2, c3, cat2b, hw2, skip2b)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c2 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    skip2b[idx] = skip2b[idx] + cat2b[c3 * hw2 + idx]
    cat2b[c3 * hw2 + idx] = 0.0
    return nothing
end

function cuda_kernel_unet_b_71!(c3, cat2b, hw2, u2b)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(c3 * hw2 - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    u2b[idx] = u2b[idx] + cat2b[idx]
    cat2b[idx] = 0.0
    return nothing
end

function cuda_kernel_unet_b_72!(bottb, c3, hw2, hw4, scale, u2b, w2, w4)
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
    bottb[xi] = bottb[xi] + u2b[idx]
    u2b[idx] = 0.0
    return nothing
end

function cuda_kernel_unet_b_73!(bott, bott_stack, bottb, n_b_out, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_b_out, -1) + 1
        return nothing
    end
    i = n_b_out + (__tid - 1) * -1
    bott[i] = bott_stack[(i - 1) + 1]
    bottb[i] = (0.5 * (1.0 + sign(bott[i] - zero_val))) * bottb[i]
    return nothing
end

function cuda_kernel_unet_b_74!(b_bbb, bottb, c3, hp4, hw4, kh, khkw, kw, t_bpad, t_bpadb, w4, w_bb, w_bbb, wp4)
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
    return nothing
end

function cuda_kernel_unet_b_75!(c3, hp4, hw4, pad, t_bb, t_bpadb, w4, wp4)
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
    t_bb[idx] = t_bb[idx] + t_bpadb[yi]
    t_bpadb[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_b_76!(n_b_midpad, t_bpadb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_b_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_bpadb[i] = 0.0
    return nothing
end

function cuda_kernel_unet_b_77!(n_b_mid, t_b, t_b_stack, t_bb, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_b_mid, -1) + 1
        return nothing
    end
    i = n_b_mid + (__tid - 1) * -1
    t_b[i] = t_b_stack[(i - 1) + 1]
    t_bb[i] = (0.5 * (1.0 + sign(t_b[i] - zero_val))) * t_bb[i]
    return nothing
end

function cuda_kernel_unet_b_78!(b_bab, c2, c3, hp4, hw4, kh, khkw, kw, p2pad, p2padb, t_bb, w4, w_ba, w_bab, wp4)
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
    return nothing
end

function cuda_kernel_unet_b_79!(c2, hp4, hw4, p2b, p2padb, pad, w4, wp4)
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
    p2b[idx] = p2b[idx] + p2padb[yi]
    p2padb[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_b_80!(n_p2pad, p2padb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_p2pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    p2padb[i] = 0.0
    return nothing
end

function cuda_kernel_unet_b_81!(a11_stack, a12_stack, a21_stack, a22_stack, c1, c2, hw2, hw4, m1_stack, m2_stack, p2b, skip2b, w2, w4)
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
    m1b = m1b + (0.5 * (1.0 + sign(m1 - m2))) * p2b[idx]
    m2b = m2b + (0.5 * (1.0 + sign(m2 - m1))) * p2b[idx]
    p2b[idx] = 0.0
    m2 = m2_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)]
    a21b = a21b + (0.5 * (1.0 + sign(a21 - a22))) * m2b
    a22b = a22b + (0.5 * (1.0 + sign(a22 - a21))) * m2b
    m2b = 0.0
    m1 = m1_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)]
    a11b = a11b + (0.5 * (1.0 + sign(a11 - a12))) * m1b
    a12b = a12b + (0.5 * (1.0 + sign(a12 - a11))) * m1b
    m1b = 0.0
    a22 = a22_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)]
    skip2b[(ci - 1) * hw2 + i0 * w2 + j0 + 1] = skip2b[(ci - 1) * hw2 + i0 * w2 + j0 + 1] + a22b
    a22b = 0.0
    a21 = a21_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)]
    skip2b[(ci - 1) * hw2 + i0 * w2 + j0] = skip2b[(ci - 1) * hw2 + i0 * w2 + j0] + a21b
    a21b = 0.0
    a12 = a12_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)]
    skip2b[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1] = skip2b[(ci - 1) * hw2 + (i0 - 1) * w2 + j0 + 1] + a12b
    a12b = 0.0
    a11 = a11_stack[(div(c1 * hw2 - 1, 1) + 1) + ((idx - 1) + 1)]
    skip2b[(ci - 1) * hw2 + (i0 - 1) * w2 + j0] = skip2b[(ci - 1) * hw2 + (i0 - 1) * w2 + j0] + a11b
    a11b = 0.0
    return nothing
end

function cuda_kernel_unet_b_82!(n_e2_out, skip2, skip2_stack, skip2b, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_e2_out, -1) + 1
        return nothing
    end
    i = n_e2_out + (__tid - 1) * -1
    skip2[i] = skip2_stack[(i - 1) + 1]
    skip2b[i] = (0.5 * (1.0 + sign(skip2[i] - zero_val))) * skip2b[i]
    return nothing
end

function cuda_kernel_unet_b_83!(b_e2bb, c2, hp2, hw2, kh, khkw, kw, skip2b, t_e2pad, t_e2padb, w2, w_e2b, w_e2bb, wp2)
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
    return nothing
end

function cuda_kernel_unet_b_84!(c2, hp2, hw2, pad, t_e2b, t_e2padb, w2, wp2)
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
    t_e2b[idx] = t_e2b[idx] + t_e2padb[yi]
    t_e2padb[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_b_85!(n_e2_midpad, t_e2padb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e2_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e2padb[i] = 0.0
    return nothing
end

function cuda_kernel_unet_b_86!(n_e2_mid, t_e2, t_e2_stack, t_e2b, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_e2_mid, -1) + 1
        return nothing
    end
    i = n_e2_mid + (__tid - 1) * -1
    t_e2[i] = t_e2_stack[(i - 1) + 1]
    t_e2b[i] = (0.5 * (1.0 + sign(t_e2[i] - zero_val))) * t_e2b[i]
    return nothing
end

function cuda_kernel_unet_b_87!(b_e2ab, c1, c2, hp2, hw2, kh, khkw, kw, p1pad, p1padb, t_e2b, w2, w_e2a, w_e2ab, wp2)
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
    return nothing
end

function cuda_kernel_unet_b_88!(c1, hp2, hw2, p1b, p1padb, pad, w2, wp2)
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
    p1b[idx] = p1b[idx] + p1padb[yi]
    p1padb[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_b_89!(n_p1pad, p1padb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_p1pad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    p1padb[i] = 0.0
    return nothing
end

function cuda_kernel_unet_b_90!(a11_stack, a12_stack, a21_stack, a22_stack, c1, hw, hw2, m1_stack, m2_stack, p1b, skip1b, w, w2)
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
    m1b = m1b + (0.5 * (1.0 + sign(m1 - m2))) * p1b[idx]
    m2b = m2b + (0.5 * (1.0 + sign(m2 - m1))) * p1b[idx]
    p1b[idx] = 0.0
    m2 = m2_stack[(idx - 1) + 1]
    a21b = a21b + (0.5 * (1.0 + sign(a21 - a22))) * m2b
    a22b = a22b + (0.5 * (1.0 + sign(a22 - a21))) * m2b
    m2b = 0.0
    m1 = m1_stack[(idx - 1) + 1]
    a11b = a11b + (0.5 * (1.0 + sign(a11 - a12))) * m1b
    a12b = a12b + (0.5 * (1.0 + sign(a12 - a11))) * m1b
    m1b = 0.0
    a22 = a22_stack[(idx - 1) + 1]
    skip1b[(ci - 1) * hw + i0 * w + j0 + 1] = skip1b[(ci - 1) * hw + i0 * w + j0 + 1] + a22b
    a22b = 0.0
    a21 = a21_stack[(idx - 1) + 1]
    skip1b[(ci - 1) * hw + i0 * w + j0] = skip1b[(ci - 1) * hw + i0 * w + j0] + a21b
    a21b = 0.0
    a12 = a12_stack[(idx - 1) + 1]
    skip1b[(ci - 1) * hw + (i0 - 1) * w + j0 + 1] = skip1b[(ci - 1) * hw + (i0 - 1) * w + j0 + 1] + a12b
    a12b = 0.0
    a11 = a11_stack[(idx - 1) + 1]
    skip1b[(ci - 1) * hw + (i0 - 1) * w + j0] = skip1b[(ci - 1) * hw + (i0 - 1) * w + j0] + a11b
    a11b = 0.0
    return nothing
end

function cuda_kernel_unet_b_91!(n_e1_out, skip1, skip1_stack, skip1b, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_e1_out, -1) + 1
        return nothing
    end
    i = n_e1_out + (__tid - 1) * -1
    skip1[i] = skip1_stack[(i - 1) + 1]
    skip1b[i] = (0.5 * (1.0 + sign(skip1[i] - zero_val))) * skip1b[i]
    return nothing
end

function cuda_kernel_unet_b_92!(b_e1bb, c1, hp1, hw, kh, khkw, kw, skip1b, t_e1pad, t_e1padb, w, w_e1b, w_e1bb, wp1)
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
    return nothing
end

function cuda_kernel_unet_b_93!(c1, hp1, hw, pad, t_e1b, t_e1padb, w, wp1)
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
    t_e1b[idx] = t_e1b[idx] + t_e1padb[yi]
    t_e1padb[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_b_94!(n_e1_midpad, t_e1padb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_e1_midpad - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
    t_e1padb[i] = 0.0
    return nothing
end

function cuda_kernel_unet_b_95!(n_e1_mid, t_e1, t_e1_stack, t_e1b, zero_val)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_e1_mid, -1) + 1
        return nothing
    end
    i = n_e1_mid + (__tid - 1) * -1
    t_e1[i] = t_e1_stack[(i - 1) + 1]
    t_e1b[i] = (0.5 * (1.0 + sign(t_e1[i] - zero_val))) * t_e1b[i]
    return nothing
end

function cuda_kernel_unet_b_96!(b_e1ab, c1, c_in, hp1, hw, kh, khkw, kw, t_e1b, w, w_e1a, w_e1ab, wp1, xpad0, xpad0b)
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
    return nothing
end

function cuda_kernel_unet_b_97!(c_in, hp1, hw, pad, w, wp1, xb, xpad0b)
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
    xb[idx] = xb[idx] + xpad0b[yi]
    xpad0b[yi] = 0.0
    return nothing
end

function cuda_kernel_unet_b_98!(n_xpad0, xpad0b)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_xpad0 - 1, 1) + 1
        return nothing
    end
    i = 1 + (__tid - 1)
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

function unet_b_cuda(x, xb, h, w, c_in, c1, c2, c3, c_out, w_e1a, w_e1ab, b_e1a, b_e1ab, w_e1b, w_e1bb, b_e1b, b_e1bb, w_e2a, w_e2ab, b_e2a, b_e2ab, w_e2b, w_e2bb, b_e2b, b_e2bb, w_ba, w_bab, b_ba, b_bab, w_bb, w_bbb, b_bb, b_bbb, w_d2a, w_d2ab, b_d2a, b_d2ab, w_d2b, w_d2bb, b_d2b, b_d2bb, w_d1a, w_d1ab, b_d1a, b_d1ab, w_d1b, w_d1bb, b_d1b, b_d1bb, w_out, w_outb, b_out, b_outb, xpad0, xpad0b, t_e1, t_e1b, t_e1pad, t_e1padb, skip1, skip1b, p1, p1b, p1pad, p1padb, t_e2, t_e2b, t_e2pad, t_e2padb, skip2, skip2b, p2, p2b, p2pad, p2padb, t_b, t_bb, t_bpad, t_bpadb, bott, bottb, u2, u2b, cat2, cat2b, cat2pad, cat2padb, t_d2, t_d2b, t_d2pad, t_d2padb, dec2out, dec2outb, u1, u1b, cat1, cat1b, cat1pad, cat1padb, t_d1, t_d1b, t_d1pad, t_d1padb, dec1out, dec1outb, y, yb, t_e1_stack, skip1_stack, a11_stack, a12_stack, a21_stack, a22_stack, m1_stack, m2_stack, t_e2_stack, skip2_stack, t_b_stack, bott_stack, t_d2_stack, dec2out_stack, t_d1_stack, dec1out_stack)
    nthread_per_block = 256
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
    @cuda threads = nthread_per_block blocks = cld(div(n_xpad0 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_1!(n_xpad0, xpad0, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c_in * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_2!(c_in, hp1, hw, pad, w, wp1, x, xpad0)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_3!(b_e1a, c1, c_in, hp1, hw, kh, khkw, kw, t_e1, w, w_e1a, wp1, xpad0)
    @cuda threads = nthread_per_block blocks = cld(div(n_e1_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_4!(n_e1_mid, t_e1, t_e1_stack, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(n_e1_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_5!(n_e1_midpad, t_e1pad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_6!(c1, hp1, hw, pad, t_e1, t_e1pad, w, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_7!(b_e1b, c1, hp1, hw, kh, khkw, kw, skip1, t_e1pad, w, w_e1b, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(n_e1_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_8!(n_e1_out, skip1, skip1_stack, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_9!(a11_stack, a12_stack, a21_stack, a22_stack, c1, hw, hw2, m1_stack, m2_stack, p1, skip1, w, w2)
    @cuda threads = nthread_per_block blocks = cld(div(n_p1pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_10!(n_p1pad, p1pad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_11!(c1, hp2, hw2, p1, p1pad, pad, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_12!(b_e2a, c1, c2, hp2, hw2, kh, khkw, kw, p1pad, t_e2, w2, w_e2a, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_e2_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_13!(n_e2_mid, t_e2, t_e2_stack, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(n_e2_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_14!(n_e2_midpad, t_e2pad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_15!(c2, hp2, hw2, pad, t_e2, t_e2pad, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_16!(b_e2b, c2, hp2, hw2, kh, khkw, kw, skip2, t_e2pad, w2, w_e2b, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_e2_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_17!(n_e2_out, skip2, skip2_stack, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_18!(a11_stack, a12_stack, a21_stack, a22_stack, c1, c2, hw2, hw4, m1_stack, m2_stack, p2, skip2, w2, w4)
    @cuda threads = nthread_per_block blocks = cld(div(n_p2pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_19!(n_p2pad, p2pad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_20!(c2, hp4, hw4, p2, p2pad, pad, w4, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_21!(b_ba, c2, c3, hp4, hw4, kh, khkw, kw, p2pad, t_b, w4, w_ba, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(n_b_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_22!(n_b_mid, t_b, t_b_stack, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(n_b_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_23!(n_b_midpad, t_bpad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_24!(c3, hp4, hw4, pad, t_b, t_bpad, w4, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_25!(b_bb, bott, c3, hp4, hw4, kh, khkw, kw, t_bpad, w4, w_bb, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(n_b_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_26!(bott, bott_stack, n_b_out, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_27!(bott, c3, hw2, hw4, scale, u2, w2, w4)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_28!(c3, cat2, hw2, u2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_29!(c2, c3, cat2, hw2, skip2)
    @cuda threads = nthread_per_block blocks = cld(div(n_cat2pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_30!(cat2pad, n_cat2pad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c32 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_31!(c32, cat2, cat2pad, hp2, hw2, pad, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_32!(b_d2a, c2, c32, cat2pad, hp2, hw2, kh, khkw, kw, t_d2, w2, w_d2a, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_d2_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_33!(n_d2_mid, t_d2, t_d2_stack, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(n_d2_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_34!(n_d2_midpad, t_d2pad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_35!(c2, hp2, hw2, pad, t_d2, t_d2pad, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_36!(b_d2b, c2, dec2out, hp2, hw2, kh, khkw, kw, t_d2pad, w2, w_d2b, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_d2_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_37!(dec2out, dec2out_stack, n_d2_out, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_38!(c2, dec2out, hw, hw2, scale, u1, w, w2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_39!(c2, cat1, hw, u1)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_40!(c1, c2, cat1, hw, skip1)
    @cuda threads = nthread_per_block blocks = cld(div(n_cat1pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_41!(cat1pad, n_cat1pad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c21 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_42!(c21, cat1, cat1pad, hp1, hw, pad, w, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_43!(b_d1a, c1, c21, cat1pad, hp1, hw, kh, khkw, kw, t_d1, w, w_d1a, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(n_d1_mid - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_44!(n_d1_mid, t_d1, t_d1_stack, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(n_d1_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_45!(n_d1_midpad, t_d1pad, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_46!(c1, hp1, hw, pad, t_d1, t_d1pad, w, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_47!(b_d1b, c1, dec1out, hp1, hw, kh, khkw, kw, t_d1pad, w, w_d1b, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(n_d1_out - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_48!(dec1out, dec1out_stack, n_d1_out, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c_out * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_49!(b_out, c1, c_out, dec1out, hw, kh_out, khkw_out, kw_out, w, w_out, y)
    CUDA.@allowscalar begin
            a11_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1] = a11
            a12_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1] = a12
            a21_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1] = a21
            a22_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1] = a22
            m1_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1] = m1
            m2_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1] = m2
            a11 = a11_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1]
            a12 = a12_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1]
            a21 = a21_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1]
            a22 = a22_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1]
            m1 = m1_stack[((div(c1 * hw2 - 1, 1) + 1) + (div(c2 * hw4 - 1, 1) + 1)) + 1]
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
        end
    @cuda threads = nthread_per_block blocks = cld(div(c_out * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_50!(b_outb, c1, c_out, dec1out, dec1outb, hw, kh_out, khkw_out, kw_out, w, w_out, w_outb, yb)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_d1_out, -1) + 1, nthread_per_block) cuda_kernel_unet_b_51!(dec1out, dec1out_stack, dec1outb, n_d1_out, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_52!(b_d1bb, c1, dec1outb, hp1, hw, kh, khkw, kw, t_d1pad, t_d1padb, w, w_d1b, w_d1bb, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_53!(c1, hp1, hw, pad, t_d1b, t_d1padb, w, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(n_d1_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_54!(n_d1_midpad, t_d1padb)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_d1_mid, -1) + 1, nthread_per_block) cuda_kernel_unet_b_55!(n_d1_mid, t_d1, t_d1_stack, t_d1b, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_56!(b_d1ab, c1, c21, cat1pad, cat1padb, hp1, hw, kh, khkw, kw, t_d1b, w, w_d1a, w_d1ab, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(c21 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_57!(c21, cat1b, cat1padb, hp1, hw, pad, w, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(n_cat1pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_58!(cat1padb, n_cat1pad)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_59!(c1, c2, cat1b, hw, skip1b)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_60!(c2, cat1b, hw, u1b)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_61!(c2, dec2outb, hw, hw2, scale, u1b, w, w2)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_d2_out, -1) + 1, nthread_per_block) cuda_kernel_unet_b_62!(dec2out, dec2out_stack, dec2outb, n_d2_out, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_63!(b_d2bb, c2, dec2outb, hp2, hw2, kh, khkw, kw, t_d2pad, t_d2padb, w2, w_d2b, w_d2bb, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_64!(c2, hp2, hw2, pad, t_d2b, t_d2padb, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_d2_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_65!(n_d2_midpad, t_d2padb)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_d2_mid, -1) + 1, nthread_per_block) cuda_kernel_unet_b_66!(n_d2_mid, t_d2, t_d2_stack, t_d2b, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_67!(b_d2ab, c2, c32, cat2pad, cat2padb, hp2, hw2, kh, khkw, kw, t_d2b, w2, w_d2a, w_d2ab, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c32 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_68!(c32, cat2b, cat2padb, hp2, hw2, pad, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_cat2pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_69!(cat2padb, n_cat2pad)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_70!(c2, c3, cat2b, hw2, skip2b)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_71!(c3, cat2b, hw2, u2b)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_72!(bottb, c3, hw2, hw4, scale, u2b, w2, w4)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_b_out, -1) + 1, nthread_per_block) cuda_kernel_unet_b_73!(bott, bott_stack, bottb, n_b_out, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_74!(b_bbb, bottb, c3, hp4, hw4, kh, khkw, kw, t_bpad, t_bpadb, w4, w_bb, w_bbb, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_75!(c3, hp4, hw4, pad, t_bb, t_bpadb, w4, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(n_b_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_76!(n_b_midpad, t_bpadb)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_b_mid, -1) + 1, nthread_per_block) cuda_kernel_unet_b_77!(n_b_mid, t_b, t_b_stack, t_bb, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c3 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_78!(b_bab, c2, c3, hp4, hw4, kh, khkw, kw, p2pad, p2padb, t_bb, w4, w_ba, w_bab, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw4 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_79!(c2, hp4, hw4, p2b, p2padb, pad, w4, wp4)
    @cuda threads = nthread_per_block blocks = cld(div(n_p2pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_80!(n_p2pad, p2padb)
    @cuda threads = nthread_per_block blocks = cld(div(1 - c2 * hw4, -1) + 1, nthread_per_block) cuda_kernel_unet_b_81!(a11_stack, a12_stack, a21_stack, a22_stack, c1, c2, hw2, hw4, m1_stack, m2_stack, p2b, skip2b, w2, w4)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_e2_out, -1) + 1, nthread_per_block) cuda_kernel_unet_b_82!(n_e2_out, skip2, skip2_stack, skip2b, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_83!(b_e2bb, c2, hp2, hw2, kh, khkw, kw, skip2b, t_e2pad, t_e2padb, w2, w_e2b, w_e2bb, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_84!(c2, hp2, hw2, pad, t_e2b, t_e2padb, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_e2_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_85!(n_e2_midpad, t_e2padb)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_e2_mid, -1) + 1, nthread_per_block) cuda_kernel_unet_b_86!(n_e2_mid, t_e2, t_e2_stack, t_e2b, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c2 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_87!(b_e2ab, c1, c2, hp2, hw2, kh, khkw, kw, p1pad, p1padb, t_e2b, w2, w_e2a, w_e2ab, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw2 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_88!(c1, hp2, hw2, p1b, p1padb, pad, w2, wp2)
    @cuda threads = nthread_per_block blocks = cld(div(n_p1pad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_89!(n_p1pad, p1padb)
    @cuda threads = nthread_per_block blocks = cld(div(1 - c1 * hw2, -1) + 1, nthread_per_block) cuda_kernel_unet_b_90!(a11_stack, a12_stack, a21_stack, a22_stack, c1, hw, hw2, m1_stack, m2_stack, p1b, skip1b, w, w2)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_e1_out, -1) + 1, nthread_per_block) cuda_kernel_unet_b_91!(n_e1_out, skip1, skip1_stack, skip1b, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_92!(b_e1bb, c1, hp1, hw, kh, khkw, kw, skip1b, t_e1pad, t_e1padb, w, w_e1b, w_e1bb, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_93!(c1, hp1, hw, pad, t_e1b, t_e1padb, w, wp1)
    @cuda threads = nthread_per_block blocks = cld(div(n_e1_midpad - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_94!(n_e1_midpad, t_e1padb)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_e1_mid, -1) + 1, nthread_per_block) cuda_kernel_unet_b_95!(n_e1_mid, t_e1, t_e1_stack, t_e1b, zero_val)
    @cuda threads = nthread_per_block blocks = cld(div(c1 * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_96!(b_e1ab, c1, c_in, hp1, hw, kh, khkw, kw, t_e1b, w, w_e1a, w_e1ab, wp1, xpad0, xpad0b)
    @cuda threads = nthread_per_block blocks = cld(div(c_in * hw - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_97!(c_in, hp1, hw, pad, w, wp1, xb, xpad0b)
    @cuda threads = nthread_per_block blocks = cld(div(n_xpad0 - 1, 1) + 1, nthread_per_block) cuda_kernel_unet_b_98!(n_xpad0, xpad0b)
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
