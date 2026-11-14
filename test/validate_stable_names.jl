# The generated source must not depend on the process. The `while` counters (`__wcount_*`, `__witer_*`) were named from
# an object id, which differs between two runs of the same input (v0.5.19 and before).
# Checks:
#  (1) two fresh processes generate the same tangent, adjoint and HVP source for every kernel with a `while`, byte for byte;
#  (2) the tags have the shape the design needs (sn_a0: two loops in one body, two tags. sn_a1: identical loops in the two arms of an if, one tag.
#      sn_a2: an inner loop that repeats an earlier loop's text and position shares its tag, the outer loop has its own);
#  (3) the adjoint and the dot-product identity hold on those kernels (they run, so a shared name is not a wrong name).
# Sabotage: STADE_SRC=<a STADE.jl with the old naming> makes (1) fail.
# usage: julia validate_stable_names.jl [--emit]      (--emit prints one hash per generated function, for the child processes)
const SN_SRC = get(ENV, "STADE_SRC", joinpath(@__DIR__, "..", "src", "STADE.jl"))
include(SN_SRC)
using SHA

const SN_DIRS = ("battle-corpus", "counted-corpus", "int-state-corpus", "boxed-reset-corpus", "stable-names-corpus", "val-corpus")

function sn_files()
    out = String[]
    for d in SN_DIRS
        dir = joinpath(@__DIR__, d)
        isdir(dir) || continue
        for f in sort(readdir(dir))
            (endswith(f, ".jl") && !occursin(r"_(b|d|hv|bd)\.jl$", f)) || continue
            p = joinpath(dir, f)
            occursin(r"^\s*while\s"m, read(p, String)) && push!(out, p)
        end
    end
    return out
end

sn_hash(x) = bytes2hex(sha256(string(x)))[1:16]

function sn_emit()
    for p in sn_files()
        e = try STADE.io_read_corpus_entry(p) catch; continue end
        rel = relpath(p, @__DIR__)
        for kpp in (true, false), fz in (false, true), (m, fn) in (("adj", STADE.stade_adjoint), ("hvp", STADE.stade_hvp), ("tan", STADE.stade_tangent))
            (m == "tan" && fz) && continue
            h = try sn_hash(m == "tan" ? fn(e; keep_push_pop = kpp) : fn(e; keep_push_pop = kpp, fuse_ii_loops = fz)) catch ex "ERR:" * first(sprint(showerror, ex), 40) end
            println(rel, " ", m, " kpp=", kpp, " fuse=", fz, " ", h)
        end
    end
end

function sn_child()
    cmd = `$(Base.julia_cmd()) --startup-file=no $(@__FILE__) --emit`
    return read(addenv(cmd, "STADE_SRC" => SN_SRC), String)
end

function sn_tags(path, kpp)
    e = STADE.io_read_corpus_entry(path)
    txt = string(STADE.stade_adjoint(e; keep_push_pop = kpp))
    return sort(unique([m.match for m in eachmatch(r"__wcount_[0-9a-f]+", txt)]))
end

function validate_stable_names(; verbose = true)
    checks = 0; bad = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1); verbose && println(ok ? "ok   " : "FAIL ", msg))
    files = sn_files()
    say(length(files) >= 30, "kernels with a while: $(length(files)), expected at least 30 (37 when written)")
    a = sn_child(); b = sn_child()
    la = split(strip(a), '\n'); lb = split(strip(b), '\n')
    say(length(la) > 100 && length(la) == length(lb), "child outputs: $(length(la)) and $(length(lb)) lines")
    diffs = [(x, y) for (x, y) in zip(la, lb) if x != y]
    say(isempty(diffs), "two processes generate identical source ($(length(diffs)) of $(length(la)) functions differ)")
    for (x, y) in first(diffs, 3)
        verbose && println("     ", x, "\n  vs ", y)
    end
    # keep_push_pop = false refuses a while by design, so only the stack-mode lines must generate
say(!any(occursin("ERR:", l) for l in la if occursin("stable-names-corpus", l) && occursin("kpp=true", l)), "no generation error on the stable-names kernels in stack mode")
    dir = joinpath(@__DIR__, "stable-names-corpus")
    say(length(sn_tags(joinpath(dir, "sn_a0.jl"), true)) == 2, "sn_a0: two loops in one body have two counters")
    say(length(sn_tags(joinpath(dir, "sn_a1.jl"), true)) == 1, "sn_a1: identical loops in the two arms share one counter")
    t2 = sn_tags(joinpath(dir, "sn_a2.jl"), true)
    say(length(t2) == 2, "sn_a2: the inner loop shares the earlier loop's counter, the outer loop has its own ($(length(t2)) counters)")
    tmp = mktempdir()
    for nm in ("sn_a0", "sn_a1", "sn_a2")
        path = joinpath(dir, nm * ".jl")
        for lo in (3, 0)
            yp = joinpath(tmp, "$nm.lo$lo.yaml")
            STADE.stade_generate_baseline_file(path; yaml_path = yp, int_lo = lo, int_hi = 5)
            r1 = STADE.stade_validate_adjoint_file(path; yaml_path = yp, keep_push_pop = true, int_lo = lo, int_hi = 5)
            say(r1.ok, "$nm adjoint int_lo=$lo")
            r2 = STADE.stade_validate_dotprod_file(path; yaml_path = yp, keep_push_pop = true)
            say(r2.ok, "$nm dot-product identity int_lo=$lo")
        end
    end
    verbose && println("validate_stable_names: $(checks - bad)/$checks ok")
    return bad == 0
end

if "--emit" in ARGS
    sn_emit()
elseif abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_stable_names() ? 0 : 1)
end
