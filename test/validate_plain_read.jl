# Regression for v0.5.19: a loop that READS a boxed scalar as a plain value is offloaded, and its kernel reads the box as `v[1]`.
#   Defect (v0.5.18 and before): the scalar-argument form (`pv_a1`) generated a kernel that received the box and multiplied by it as a value; the
#   local form was kept on the host by a prune rule (a loop that mentions a local without accumulating into it refused the boxing).
#   Fix: cgen_box_reads_loop rewrites the reads of a name boxed in the final output to `v[1]` on the copy of the loop the kernel is built from; a host
#   reduction (mapreduce, JACC.@parallel_reduce) reads a host copy `__pv_v = v[1]` made inside its own branch. cgen_emit/jgen_emit generate a second
#   pass when some loop reads a boxed name, so the first pass of every other kernel is unchanged.
# Part 1 counts the loops left on the host per kernel (primal, adjoint, HVP) and the loops lowered as a reduction.
# Part 2 checks the generated CUDA and JACC text: the boxed locals and arguments (exact), the kernels that read a box as `v[1]` (exact), and that a
#   kernel uses a box only as `v[1]` or as the target of the atomic helper, never as a bare value and never written other than atomically.
# Part 3 validates each kernel against finite differences, the tangent-adjoint dot-product identity and the HVP, with a zero-trip draw (int_lo = 0).
# No device is used. The device check is a V100 run of the same corpus.
include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))

# kernel => (primal host loops, reasons, primal reductions, adjoint host loops, reasons, adjoint reductions); host loops of the HVP equal the adjoint's
# the adjoint is the keep_push_pop = false, fuse_ii_loops = true one; `nothing`: it needs keep_push_pop = true (a `while` has no closed-form stack size)
const PV_EXPECT = Dict(
    "pv_a0"  => (0, "", 0, 0, "", 0),                              # accumulate, then read
    "pv_a1"  => (0, "", 0, 0, "", 0),                              # the same with a scalar ARGUMENT (the older defect)
    "pv_a10" => (0, "", 0, nothing, "", 0),                        # read in a while condition
    "pv_a11" => (1, "array", 2, 3, "array", 4),                    # CG with host scalars: only the outer loop stays
    "pv_a12" => (1, "array", 1, 3, "array+scalar", 2),             # power iteration, norm a local reduction
    "pv_a13" => (0, "", 0, 1, "accread", 0),                       # phases; the stacked running sum of the adjoint is order dependent
    "pv_a14" => (1, "array", 0, 3, "array", 0),                    # a host recurrence reads the box
    "pv_a15" => (1, "array", 0, 3, "array", 0),                    # one reader offloaded, one recurrence on the host
    "pv_a16" => (1, "accread", 0, 3, "accread+scalar", 0),         # a running sum read in its own loop: host
    "pv_a17" => (1, "array", 0, 3, "array+scalar", 0),             # power iteration, norm boxed, read in the outer loop
    "pv_a18" => (1, "array", 1, 3, "array+scalar", 2),             # CG, rsn boxed, read in the next loop
    "pv_a19" => (0, "", 0, 0, "", 0),                              # an Int ARGUMENT counted in one loop, the bound of the next
    "pv_a2"  => (0, "", 0, 0, "", 0),                              # read in an if
    "pv_a3"  => (0, "", 0, 0, "", 0),                              # read in an inner loop
    "pv_a4"  => (0, "", 0, 0, "", 0),                              # split into two kernels, one reads
    "pv_a5"  => (0, "", 0, 0, "", 0),                              # the reader also accumulates a second local
    "pv_a6"  => (0, "", 1, 0, "", 2),                              # the reader is a local reduction (host copy)
    "pv_a7"  => (0, "", 1, 0, "", 2),                              # the reader reduces into an array element
    "pv_a8"  => (0, "", 0, 2, "liveout+scalar", 1),                # `acc` private in one loop, boxed elsewhere
    "pv_a9"  => (0, "", 0, 0, "", 0),                              # zero-trip accumulating loop
)
# boxed locals of the primal CUDA output (exact); every other kernel none
const PV_BOXED = Dict("pv_a0" => ["acc"], "pv_a10" => ["acc"], "pv_a13" => ["acc"], "pv_a14" => ["acc"], "pv_a15" => ["acc"], "pv_a17" => ["nrm"],
                      "pv_a18" => ["rsn"], "pv_a2" => ["acc"], "pv_a3" => ["acc"], "pv_a4" => ["acc"], "pv_a5" => ["acc", "nrm"], "pv_a6" => ["acc"],
                      "pv_a7" => ["acc"], "pv_a8" => ["acc"], "pv_a9" => ["acc"])
# boxed scalar ARGUMENTS of the primal CUDA output (exact)
const PV_BOXED_ARGS = Dict("pv_a1" => ["acc"], "pv_a19" => ["cnt"])
# boxes some kernel of the primal CUDA output reads as `b[1]` (exact; a reader that is a host reduction or a host loop does not count)
const PV_READS = Dict("pv_a0" => ["acc"], "pv_a1" => ["acc"], "pv_a10" => ["acc"], "pv_a13" => ["acc"], "pv_a15" => ["acc"], "pv_a17" => ["nrm"],
                      "pv_a18" => ["rsn"], "pv_a19" => ["cnt"], "pv_a2" => ["acc"], "pv_a3" => ["acc"], "pv_a4" => ["acc"], "pv_a5" => ["acc"],
                      "pv_a7" => ["acc"], "pv_a8" => ["acc"], "pv_a9" => ["acc"])
# kernels whose only box read is the CUDA guard `__tid > trip count` (a JACC kernel has no guard: its range is the launch's)
const PV_CUDA_ONLY = ("pv_a19",)
# kernels whose host reduction reads a host copy `__pv_v` in both outputs
const PV_COPY = Dict("pv_a6" => ["acc"])
# kernels whose derivatives need keep_push_pop = true
const PV_KPP_TRUE_ONLY = ("pv_a10",)

function pv_host_loops(path)
    recs = STADE.stade_gpu_report_file(path)
    h = [r for r in recs if r.depth >= 0 && r.verdict == :host]
    return length(h), join(sort(unique([string(r.reason) for r in h])), "+")
end
pv_reduction_loops(path) = count(r -> r.depth >= 0 && r.verdict == :reduction, STADE.stade_gpu_report_file(path))

# kernel and host function bodies of a generated file, by name
function pv_functions(text)
    out = Dict{String,String}()
    ms = collect(eachmatch(r"(?m)^function (\w+!?)\(", text))
    for (k, m) in enumerate(ms)
        stop = k < length(ms) ? ms[k + 1].offset - 1 : lastindex(text)
        out[m.captures[1]] = text[m.offset:stop]
    end
    return out
end

# boxed locals (`v = CuArray([0.0])`) and boxed arguments (`v = CuArray([v])`) at a function entry
pv_boxed_locals(text) = Set{String}(m.captures[1] for m in eachmatch(r"(?m)^\s*(\w+) = (?:CuArray|JACC\.array)\(\[0\.0(?:f0)?\]\)", text))
pv_boxed_args(text) = Set{String}(m.captures[1] for m in eachmatch(r"(?m)^\s*(\w+) = (?:CuArray|JACC\.array)\(\[(\w+)\]\)", text) if m.captures[1] == m.captures[2])

# every kernel that receives a box uses it only as `b[1]` (read) or as the target of the atomic helper; returns the boxes some kernel reads.
# A generated file can hold several host functions (an adjoint file carries its primal too): a box is a box of ITS OWNER, and only that owner's
# kernels (`cuda_kernel_<owner>_<k>!`) are checked against it.
function pv_check_boxes(text, say, label)
    fns = pv_functions(text)
    boxed_all = Set{String}()
    received = 0
    reads = Set{String}()
    for (hname, hbody) in fns
        m = match(r"^(.+)_(?:cuda|jacc)$", hname)
        m === nothing && continue
        owner = m.captures[1]
        boxed = union(pv_boxed_locals(hbody), pv_boxed_args(hbody))
        union!(boxed_all, boxed)
        # a grid expression (`blocks = ...`, `range = ...`) is computed on the host: it reads `b[1]`, never the box
        for b in boxed
            for gm in eachmatch(r"(?:blocks|range) = (.*?) (?:cuda|jacc)_kernel_", hbody)
                say(!occursin(Regex("\\b" * b * "\\b(?!\\[)"), gm.captures[1]), "$label: the host grid expression of $hname reads the box $b as a value")
            end
        end
        for (fname, body) in fns
            km = match(r"^(?:cuda|jacc)_kernel_(.+?)_(\d+)(?:_\w+)?!$", fname)
            km === nothing && continue
            km.captures[1] == owner || continue
            lines = split(body, "\n")
            sig = lines[1]
            rest = join(lines[2:end], "\n")
            for b in boxed
                # a kernel that reads `b[1]` must have received the box: a private scalar of the same name would be an undefined variable
                occursin(Regex("\\b" * b * "\\b"), sig) || (say(!occursin(Regex("\\b" * b * "\\[1\\]"), rest), "$label: kernel $fname reads $b[1] but does not receive the box"); continue)
                received += 1
                stripped = replace(rest, Regex("__stade_wa_add!\\(\\s*" * b * ",\\s*1,") => "")
                stripped = replace(stripped, Regex("CUDA\\.@atomic\\s+" * b * "\\[1\\]\\s*\\+=") => "")
                say(!occursin(Regex("\\b" * b * "\\[1\\]\\s*[-+*/]?=(?!=)"), stripped), "$label: kernel $fname writes the box $b other than atomically")
                if occursin(Regex("\\b" * b * "\\[1\\]"), stripped)
                    push!(reads, b)
                    stripped = replace(stripped, Regex("\\b" * b * "\\[1\\]") => "")
                end
                say(!occursin(Regex("\\b" * b * "\\b"), stripped), "$label: kernel $fname uses the box $b as a bare value")
            end
        end
    end
    say(isempty(boxed_all) || received > 0, "$label: boxes $(sort(collect(boxed_all))) exist but no kernel receives one (the static check saw nothing)")
    # every host copy `__pv_x` is made from the box: `__pv_x = x[1]`
    for m in eachmatch(r"__pv_(\w+)", text)
        nm = m.captures[1]
        say(occursin(Regex("__pv_" * nm * " = " * nm * "\\[1\\]"), text), "$label: host copy __pv_$nm is never assigned from $nm[1]")
    end
    return boxed_all, reads
end

function validate_plain_read(; verbose = true, only = String[], values = true)
    bad = 0; checks = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("NOT OK ", msg)))
    dir = joinpath(@__DIR__, "plain-read-corpus")
    tmp = mktempdir()
    names = sort(String[f[1:end-3] for f in readdir(dir) if endswith(f, ".jl")])
    isempty(only) || (names = filter(n -> n in only, names))
    for nm in names
        path = joinpath(dir, nm * ".jl")
        haskey(PV_EXPECT, nm) || (say(false, "$nm: no expectation in PV_EXPECT"); continue)
        (hp, rp, redp, ha, ra, reda) = PV_EXPECT[nm]
        # Part 1: loops left on the host
        n, r = pv_host_loops(path)
        say(n == hp && r == rp, "$nm primal: $n host loops ($r), expected $hp ($rp)")
        nr = pv_reduction_loops(path)
        say(nr == redp, "$nm primal: $nr loops lowered as a reduction, expected $redp")
        b = joinpath(tmp, nm * "_b.jl")
        if ha === nothing
            say(try STADE.stade_adjoint_file(path, b; keep_push_pop = false, fuse_ii_loops = true); false catch; true end,
                "$nm: the adjoint with keep_push_pop = false was expected to be refused")
        else
            STADE.stade_adjoint_file(path, b; keep_push_pop = false, fuse_ii_loops = true)
            n, r = pv_host_loops(b)
            say(n == ha && r == ra, "$nm adjoint: $n host loops ($r), expected $ha ($ra)")
            nr = pv_reduction_loops(b)
            say(nr == reda, "$nm adjoint: $nr loops lowered as a reduction, expected $reda")
            # the HVP offloads exactly as well as the adjoint
            h = joinpath(tmp, nm * "_h.jl")
            STADE.stade_hvp_file(path, h; keep_push_pop = false, fuse_ii_loops = true)
            nh, rh = pv_host_loops(h)
            say(nh == n && rh == r, "$nm hvp: $nh host loops ($rh) differ from the adjoint ($n, $r)")
        end
        # Part 2: generated code
        for (lab, src) in (("primal", path), ("adjoint", b), ("hvp", joinpath(tmp, nm * "_h.jl")))
            lab == "primal" || ha === nothing && continue
            c = joinpath(tmp, nm * "_$lab" * "_cuda.jl"); j = joinpath(tmp, nm * "_$lab" * "_jacc.jl")
            STADE.stade_cuda_file(src, c); STADE.stade_jacc_file(src, j)
            tc = read(c, String); tj = read(j, String)
            bc, rc = pv_check_boxes(tc, say, "$nm $lab CUDA")
            bj, rj = pv_check_boxes(tj, say, "$nm $lab JACC")
            say(bc == bj, "$nm $lab: the CUDA and JACC outputs box different names ($(sort(collect(bc))) vs $(sort(collect(bj))))")
            say(rc == rj || nm in PV_CUDA_ONLY, "$nm $lab: the CUDA and JACC kernels read different boxes ($(sort(collect(rc))) vs $(sort(collect(rj))))")
            say(!occursin(r"CuArray\(\[0\]\)|JACC\.array\(\[0\]\)", tc * tj), "$nm $lab: an Int box")
            if lab == "primal"
                say(sort(collect(pv_boxed_locals(tc))) == get(PV_BOXED, nm, String[]), "$nm: boxed locals $(sort(collect(pv_boxed_locals(tc)))), expected $(get(PV_BOXED, nm, String[]))")
                say(sort(collect(pv_boxed_args(tc))) == get(PV_BOXED_ARGS, nm, String[]), "$nm: boxed arguments $(sort(collect(pv_boxed_args(tc)))), expected $(get(PV_BOXED_ARGS, nm, String[]))")
                say(sort(collect(rc)) == get(PV_READS, nm, String[]), "$nm CUDA: boxes read by a kernel $(sort(collect(rc))), expected $(get(PV_READS, nm, String[]))")
                say(sort(collect(rj)) == get(PV_READS, nm, String[]) || haskey(PV_COPY, nm) || nm in PV_CUDA_ONLY, "$nm JACC: boxes read by a kernel $(sort(collect(rj))), expected $(get(PV_READS, nm, String[]))")
                cp = sort(unique(String[m.captures[1] for m in eachmatch(r"__pv_(\w+)", tc)]))
                say(haskey(PV_COPY, nm) ? cp == PV_COPY[nm] : true, "$nm CUDA: host copies $cp, expected $(get(PV_COPY, nm, String[]))")
            end
        end
        values || continue
        # Part 3: values
        for lo in (3, 0)
            yp = joinpath(tmp, "$nm.lo$lo.yaml")
            STADE.stade_generate_baseline_file(path; yaml_path = yp, int_lo = lo, int_hi = 5)
            for (kp, fz) in (nm in PV_KPP_TRUE_ONLY ? ((true, false), (true, true)) : ((false, true), (false, false), (true, false)))
                r1 = STADE.stade_validate_adjoint_file(path; yaml_path = yp, keep_push_pop = kp, fuse_ii_loops = fz, int_lo = lo, int_hi = 5)
                say(r1.ok, "$nm adjoint keep_push_pop=$kp fuse=$fz int_lo=$lo")
                r2 = STADE.stade_validate_dotprod_file(path; yaml_path = yp, keep_push_pop = kp, fuse_ii_loops = fz)
                say(r2.ok, "$nm dot-product identity keep_push_pop=$kp fuse=$fz int_lo=$lo")
            end
            if lo == 3 && !(nm in PV_KPP_TRUE_ONLY)
                for fz in (true, false)
                    r3 = STADE.stade_validate_hvp_file(path; yaml_path = yp, keep_push_pop = false, fuse_ii_loops = fz, int_lo = lo, int_hi = 5)
                    say(r3.ok, "$nm hvp fuse=$fz")
                end
            end
        end
    end
    verbose && println("validate_plain_read: $(checks - bad)/$checks ok")
    return bad == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    only = filter(a -> a != "--no-values", ARGS)
    exit(validate_plain_read(; only, values = !("--no-values" in ARGS)) ? 0 : 1)
end
