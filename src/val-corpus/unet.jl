function unet(h, w, c_in, c1, b_e1a, t_e1)

    kh = 3
    kw = 3
    khkw = kh * kw
    hw = h * w

    for idx = 1:c1 * hw
        idxm1 = idx - 1
        co = div(idxm1, hw) + 1
        s = 0.0
        for i_seq_k = 1:c_in * khkw
        end
        t_e1[idx] = s + b_e1a[co]
    end

    return nothing
end