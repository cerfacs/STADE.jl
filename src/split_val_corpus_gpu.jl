#!/usr/bin/env julia
#
# split_val_corpus_gpu.jl
#
# Builds `val-corpus-gpu-split/<basename>/` for every kernel file found in
# `val-corpus-gpu/`. Each subfolder gets:
#   1. A copy of `validate_corpus_gpu.jl`
#      whose trailing call
#          validate_corpus_gpu()
#      has been rewritten to pass the subfolder's basename as `dir`:
#          validate_corpus_gpu("unet")
#   2. A copy of the fixed, pre-existing `STADE.jl` file (the template
#      `include("STADE.jl")`s this), pasted alongside the template file.
#   3. A copy of the original kernel file (e.g. `unet.jl`), nested one level
#      deeper inside a subsubfolder of the same basename, e.g.:
#          val-corpus-gpu-split/unet/unet/unet.jl
#      so that the rewritten call's `readdir(dir)` (with dir="unet") finds
#      exactly that one kernel inside `unet/unet/`.
#
# Usage:
#   julia split_val_corpus_gpu.jl [src_dir] [template_file] [out_dir] [stade_file]
#
# Defaults:
#   src_dir       = "val-corpus-gpu"
#   template_file = "validate_corpus_gpu.jl"
#   out_dir       = "val-corpus-gpu-split"
#   stade_file    = "STADE.jl"

function split_val_corpus_gpu(
    src_dir::String = "val-corpus-gpu",
    template_file::String = "validate_corpus_gpu.jl",
    out_dir::String = "val-corpus-gpu-split",
    stade_file::String = "STADE.jl",
)
    isdir(src_dir) || error("source directory not found: $src_dir")
    isfile(template_file) || error("template file not found: $template_file")
    isfile(stade_file) || error("STADE file not found: $stade_file")

    template_name = basename(template_file)
    template_src  = read(template_file, String)
    stade_name    = basename(stade_file)

    fname = basename(template_file)
    # strip the trailing .jl to get the bare function name being called
    fn_name, _ = splitext(fname)
    old_call = fn_name * "()"

    mkpath(out_dir)

    kernel_files = filter(f -> endswith(f, ".jl"), readdir(src_dir))
    isempty(kernel_files) && @warn "no .jl files found in $src_dir"

    for f in kernel_files
        base = splitext(f)[1]
        subdir = joinpath(out_dir, base)
        mkpath(subdir)

        # locate the LAST occurrence of the bare call and replace only that one
        rng = findlast(old_call, template_src)
        rng === nothing && error("could not find trailing call `$old_call` in $template_file")

        new_call = "$fn_name(\"$base\")"
        new_src = string(
            template_src[1:prevind(template_src, first(rng))],
            new_call,
            template_src[nextind(template_src, last(rng)):end],
        )

        write(joinpath(subdir, template_name), new_src)

        # paste the fixed STADE.jl alongside the template file (unchanged)
        cp(stade_file, joinpath(subdir, stade_name); force = true)

        # nest the original kernel file one level deeper, in a subsubfolder
        # of the same basename, e.g. val-corpus-gpu-split/unet/unet/unet.jl
        subsubdir = joinpath(subdir, base)
        mkpath(subsubdir)
        cp(joinpath(src_dir, f), joinpath(subsubdir, f); force = true)

        println("created ", subdir, "  (", template_name, " + ", stade_name, " at top, ", f, " under ", base, "/)")
    end

    return out_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = ARGS
    split_val_corpus_gpu(args...)
end