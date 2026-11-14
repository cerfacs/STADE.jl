# Regression for v0.5.18: loops that mix array writes with accumulation into a LOCAL scalar are offloaded.
#   Defect: `rsn = 0.0; for i ...; r[i] = ...; rsn += r[i]*r[i]; end` (the squared residual of CG, `alphab`/`betab` in the reverse sweep) stayed on the host,
#   one device access per element, because the accumulator is a plain scalar that a kernel cannot update.
#   Fix: a local scalar that is only ever accumulated by the loops that touch it is boxed (1-element device array, atomic add in the kernel, `v[1]` on the host).
#   `cgen_boxable_locals` decides which locals may be boxed; the split-loop gate refuses a loop that READS its own accumulator (:accread), which also
#   closed a defect that existed before: a scalar ARGUMENT used as a running sum (`acc += x[i]; y[i] = acc * x[i]`) was offloaded and gave the wrong values.
# Part 1 counts the loops left on the host per kernel (primal and adjoint) and their reasons; the shapes that must stay on the host stay there.
# Part 2 validates every kernel against finite differences, the tangent-adjoint dot-product identity and the HVP, with a zero-trip draw (int_lo = 0).
# Part 3 reads the CUDA and JACC output: every boxed local is used only as the target of the atomic helper inside a kernel, and is a device array at entry.
# Part 4 checks that the HVP offloads exactly as well as the adjoint. No device is used. The device check is a V100 run of the same corpus.
include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))

# kernel => (host loops in the primal, reasons, host loops in the adjoint keep_push_pop=false fuse=true, reasons); "n/a": the adjoint needs keep_push_pop = true
# the remaining host loops are the outer counted loops of the solver shapes and the loops the gate must keep (see the comment on each)
const SA_EXPECT = Dict(
    "sa_a0"  => (0, "", 0, ""),                          # basic mixed loop: offloaded (host before)
    "sa_a1"  => (1, "array", 3, "array+scalar"),         # CG with a local rsn under a guard: only the outer iteration loop stays
    "sa_a10" => (1, "scalar", 3, "scalar"),              # acc = acc * 0.5 + y[i] is no accumulation: must stay on the host
    "sa_a11" => (0, "", 0, ""),                          # subtraction, scalar argument
    "sa_a12" => (1, "scope", nothing, ""),               # Int counter: one shape, never boxed as a Float64 array
    "sa_a13" => (0, "", 0, ""),                          # private per-iteration accumulator: offloaded before and unchanged
    "sa_a14" => (1, "accread", 3, "accread+scalar"),     # scalar ARGUMENT as a running sum: host (it was a silent wrong offload)
    "sa_a2"  => (1, "accread", 3, "accread+scalar"),     # running sum read in its own loop: order-dependent, host
    "sa_a3"  => (0, "", 0, ""),                          # a second loop reads the accumulated scalar as a plain value: its kernel reads the box (v0.5.19; host before)
    "sa_a4"  => (0, "", 0, ""),                          # two accumulators, if/else
    "sa_a5"  => (0, "", 0, ""),                          # the name `acc` is private in one kernel and an accumulator in another
    "sa_a6"  => (0, "", 0, ""),                          # private accumulator inside, boxed accumulator outside
    "sa_a7"  => (1, "accread", 3, "accread"),            # zero-trip, an outer loop reads the accumulator; the inner loop is offloaded
    "sa_a8"  => (0, "", 0, ""),                          # two loops accumulate the same local, host read between
    "sa_a9"  => (1, "array", 3, "array"),                # a sequential recurrence reads the accumulator: host; the accumulating loop and the other reader are offloaded (v0.5.19)
    "sa_pw"  => (1, "array", 3, "array+scalar"),         # power iteration, two local accumulators
    "sa_sd"  => (1, "array", 3, "array+scalar"),         # steepest descent, two local accumulators
)
# loops lowered as a pure reduction (mapreduce) rather than a kernel with a box: a loop whose only effect is the accumulation keeps that lowering
const SA_RED = Dict("sa_a1" => (1, 2), "sa_sd" => (1, 2))   # (primal, adjoint); every other kernel 0
# locals boxed in the primal CUDA output (a scalar argument that is boxed is not listed: only locals have a literal start value); every other kernel none
const SA_BOXED = Dict("sa_a0" => ["acc"], "sa_a1" => ["rsn"], "sa_a11" => ["acc"], "sa_a3" => ["acc"], "sa_a9" => ["acc"], "sa_a4" => ["a", "b"], "sa_a5" => ["acc"], "sa_a6" => ["tot"],
                      "sa_a7" => ["acc"], "sa_a8" => ["acc"], "sa_pw" => ["nrm", "num"], "sa_sd" => ["rr"])
# kernels whose derivatives need keep_push_pop = true (an Int that is read after an iteration-dependent update)
const SA_KPP_TRUE_ONLY = ("sa_a12",)

function sa_reduction_loops(path)
    return count(r -> r.depth >= 0 && r.verdict == :reduction, STADE.stade_gpu_report_file(path))
end

function sa_host_loops(path)
    recs = STADE.stade_gpu_report_file(path)
    h = [r for r in recs if r.depth >= 0 && r.verdict == :host]
    return length(h), join(sort(unique([string(r.reason) for r in h])), "+")
end

# kernel function bodies of a generated file, by name
function sa_functions(text)
    out = Dict{String,String}()
    ms = collect(eachmatch(r"(?m)^function (\w+!?)\(", text))
    for (k, m) in enumerate(ms)
        stop = k < length(ms) ? ms[k + 1].offset - 1 : lastindex(text)
        out[m.captures[1]] = text[m.offset:stop]
    end
    return out
end

# boxed locals in a generated file: names assigned `CuArray([0.0...])` / `JACC.array([0.0...])` at a function entry
sa_boxed_names(text) = Set{String}(m.captures[1] for m in eachmatch(r"(?m)^\s*(\w+) = (?:CuArray|JACC\.array)\(\[0\.0(?:f0)?\]\)", text))

# every kernel (cuda_kernel_ / jacc_kernel_) that receives a boxed name uses it only as the target of the atomic helper or reads it as `b[1]` (v0.5.19:
# a loop that reads a boxed accumulator as a plain value; validate_plain_read.jl has the exact sets of readers)
function sa_check_boxes(text, say, label)
    boxed = sa_boxed_names(text)
    fns = sa_functions(text)
    received = 0   # kernels that got a box: the check below must have looked at one whenever a box exists, or it checks nothing
    for (fname, body) in fns
        occursin(r"^(?:cuda|jacc)_kernel_", fname) || continue
        lines = split(body, "\n")
        sig = lines[1]
        rest = join(lines[2:end], "\n")
        for b in boxed
            occursin(Regex("\\b" * b * "\\b"), sig) || continue
            received += 1
            stripped = replace(rest, Regex("__stade_wa_add!\\(\\s*" * b * ",\\s*1,") => "")
            stripped = replace(stripped, Regex("\\b" * b * "\\[1\\](?!\\s*[-+*/]?=(?!=))") => "")
            say(!occursin(Regex("\\b" * b * "\\b"), stripped), "$label: kernel $fname uses the boxed local $b other than through the atomic helper or as $b[1]")
        end
    end
    say(isempty(boxed) || received > 0, "$label: boxed locals $(sort(collect(boxed))) exist but no kernel receives one (the static check saw nothing)")
    return boxed
end

function validate_scalar_acc(; verbose = true, only = String[])
    bad = 0; checks = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("NOT OK ", msg)))
    dir = joinpath(@__DIR__, "scalar-acc-corpus")
    tmp = mktempdir()
    names = sort(String[f[1:end-3] for f in readdir(dir) if endswith(f, ".jl")])
    isempty(only) || (names = filter(n -> n in only, names))
    for nm in names
        path = joinpath(dir, nm * ".jl")
        haskey(SA_EXPECT, nm) || (say(false, "$nm: no expectation in SA_EXPECT"); continue)
        (hp, rp, ha, ra) = SA_EXPECT[nm]
        # Part 1: loops left on the host
        n, r = sa_host_loops(path)
        say(n == hp && r == rp, "$nm primal: $n host loops ($r), expected $hp ($rp)")
        b = joinpath(tmp, nm * "_b.jl")
        (rp_expect, ra_expect) = get(SA_RED, nm, (0, 0))
        nr = sa_reduction_loops(path)
        say(nr == rp_expect, "$nm primal: $nr loops lowered as a reduction, expected $rp_expect")
        if ha === nothing
            say(try STADE.stade_adjoint_file(path, b; keep_push_pop = false, fuse_ii_loops = true); false catch; true end,
                "$nm: the adjoint with keep_push_pop = false was expected to be refused")
        else
            STADE.stade_adjoint_file(path, b; keep_push_pop = false, fuse_ii_loops = true)
            n, r = sa_host_loops(b)
            say(n == ha && r == ra, "$nm adjoint: $n host loops ($r), expected $ha ($ra)")
            nr = sa_reduction_loops(b)
            say(nr == ra_expect, "$nm adjoint: $nr loops lowered as a reduction, expected $ra_expect")
            # Part 4: the HVP offloads exactly as well as the adjoint
            h = joinpath(tmp, nm * "_h.jl")
            STADE.stade_hvp_file(path, h; keep_push_pop = false, fuse_ii_loops = true)
            nh, rh = sa_host_loops(h)
            say(nh == n && rh == r, "$nm hvp: $nh host loops ($rh) differ from the adjoint ($n, $r)")
        end
        # Part 2: values
        for lo in (3, 0)
            yp = joinpath(tmp, "$nm.lo$lo.yaml")
            STADE.stade_generate_baseline_file(path; yaml_path = yp, int_lo = lo, int_hi = 5)
            for (kp, fz) in (nm in SA_KPP_TRUE_ONLY ? ((true, false), (true, true)) : ((false, true), (false, false), (true, false)))
                r1 = STADE.stade_validate_adjoint_file(path; yaml_path = yp, keep_push_pop = kp, fuse_ii_loops = fz, int_lo = lo, int_hi = 5)
                say(r1.ok, "$nm adjoint keep_push_pop=$kp fuse=$fz int_lo=$lo")
                r2 = STADE.stade_validate_dotprod_file(path; yaml_path = yp, keep_push_pop = kp, fuse_ii_loops = fz)
                say(r2.ok, "$nm dot-product identity keep_push_pop=$kp fuse=$fz int_lo=$lo")
            end
            if lo == 3 && !(nm in SA_KPP_TRUE_ONLY)
                for fz in (true, false)
                    r3 = STADE.stade_validate_hvp_file(path; yaml_path = yp, keep_push_pop = false, fuse_ii_loops = fz, int_lo = lo, int_hi = 5)
                    say(r3.ok, "$nm hvp keep_push_pop=$fz")
                end
            end
        end
        # Part 3: generated code
        for (lab, src) in (("primal", path), ("adjoint", b))
            lab == "adjoint" && ha === nothing && continue
            c = joinpath(tmp, nm * "_$lab" * "_cuda.jl"); j = joinpath(tmp, nm * "_$lab" * "_jacc.jl")
            STADE.stade_cuda_file(src, c); STADE.stade_jacc_file(src, j)
            bc = sa_check_boxes(read(c, String), say, "$nm $lab CUDA")
            bj = sa_check_boxes(read(j, String), say, "$nm $lab JACC")
            say(bc == bj, "$nm $lab: the CUDA and JACC outputs box different locals ($(sort(collect(bc))) vs $(sort(collect(bj))))")
            # no Int and no boxed scalar argument: every box is a Float64 literal zero at entry
            say(!occursin(r"CuArray\(\[0\]\)|JACC\.array\(\[0\]\)", read(c, String) * read(j, String)), "$nm $lab: an Int box")
        end
    end
    # exactly these locals are boxed in the primal output: no more (a private or Int scalar, a loop that stays on the host) and no fewer
    for nm in names
        c = joinpath(tmp, nm * "_primal_cuda.jl")
        isfile(c) || continue
        got = sort(collect(sa_boxed_names(read(c, String))))
        say(got == get(SA_BOXED, nm, String[]), "$nm: boxed locals $got, expected $(get(SA_BOXED, nm, String[]))")
    end
    verbose && println("validate_scalar_acc: $(checks - bad)/$checks ok")
    return bad == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_scalar_acc(; only = ARGS) ? 0 : 1)
end
