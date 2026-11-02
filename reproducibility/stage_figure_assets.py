"""Deposit the checkpoints and datasets behind each figure, and write the per-figure manifests.

Reads reproducibility/figure_assets.yaml, uploads anything it declares that the store does not
already hold, and writes one MANIFEST.yaml and one fetch.sh per figure under <store>/by_figure/
together with an index README.

The store is passed in and is never written down here or in anything this script generates. Assets
are named by their path relative to the deposit root, which is the same path they have in the
repository, so one name identifies an asset in the staging store and in the published Figshare
record alike. `fetch.sh` takes the deposit root as its argument for the same reason.

Nothing is stored twice. Every asset keeps one object, and the per-figure folders hold only the
manifests and the fetch scripts that name it. A figure that shares a model with another figure names
the same object.

    python reproducibility/stage_figure_assets.py --store gs://BUCKET --dry-run
    python reproducibility/stage_figure_assets.py --store gs://BUCKET
    python reproducibility/stage_figure_assets.py --store gs://BUCKET --figures fig4 ed2
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import date

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(REPO_ROOT, "reproducibility", "figure_assets.yaml")

# The store prefixes an existence check has to cover. Listing these once beats one stat per object.
LISTED_PREFIXES = ["cortex/", "xenium/", "atera_breast/", "human_tonsil/", "semisyn/"]


def run(cmd, **kwargs):
    """Run a command, returning its completed process; stderr is left on the terminal."""
    return subprocess.run(cmd, text=True, capture_output=True, **kwargs)


def md5_of(path):
    """Hex md5 of a file, read in chunks."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_index():
    """Load the asset index and resolve every {ref: ...} against the catalog."""
    with open(INDEX) as handle:
        index = yaml.safe_load(handle)
    catalog = index.get("catalog") or {}

    for figure in index["figures"]:
        resolved = []
        for asset in figure.get("assets") or []:
            if "ref" in asset:
                name = asset["ref"]
                if name not in catalog:
                    sys.exit(f"{figure['id']}: unknown catalog entry {name!r}")
                merged = dict(catalog[name])
                merged.update({k: v for k, v in asset.items() if k != "ref"})
                merged["ref"] = name
                resolved.append(merged)
            else:
                resolved.append(dict(asset))
        figure["assets"] = resolved
    return index


def existing_objects(store):
    """Names of the objects the store already holds, relative to its root."""
    found = set()
    for prefix in LISTED_PREFIXES:
        result = run(["gsutil", "ls", "-r", f"{store}/{prefix}**"])
        if result.returncode != 0:
            # An absent prefix is not an error; it just has nothing in it yet.
            continue
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith(store) and not line.endswith("/") and not line.endswith(":"):
                found.add(line[len(store) + 1 :])
    return found


def stored_hash(url):
    """Size and checksum of one stored object, for the manifest to be verifiable against.

    GCS keeps no md5 for an object uploaded in parallel components, so the larger datasets report a
    crc32c instead. `gsutil hash -c <local file>` prints the value to compare it with.
    """
    result = run(["gsutil", "stat", url])
    if result.returncode != 0:
        return {}
    out = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "Content-Length":
            out["bytes"] = int(value)
        elif key == "Hash (md5)":
            out["stored_md5"] = base64.b64decode(value).hex()
        elif key == "Hash (crc32c)":
            out["stored_crc32c"] = value
        elif key == "Component-Count":
            out["components"] = int(value)
    if "stored_md5" not in out and "stored_crc32c" in out:
        out["hash_note"] = "composite upload; the store keeps no md5, compare the crc32c"
    return out


def members_of(asset):
    """Local files a directory asset covers, as (absolute local path, path relative to its root)."""
    root = os.path.join(REPO_ROOT, asset["local"])
    include = asset.get("include")
    skip = set(asset.get("exclude_dirs") or [])
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        top = rel_dir.split(os.sep)[0]
        if top in skip:
            dirnames[:] = []
            continue
        for name in sorted(filenames):
            if include and name not in include:
                continue
            full = os.path.join(dirpath, name)
            out.append((full, os.path.relpath(full, root)))
    return sorted(out, key=lambda pair: pair[1])


def exclude_regex(asset):
    """A gsutil rsync -x pattern that keeps only the files this asset includes."""
    patterns = []
    include = asset.get("include")
    if include:
        alternatives = "|".join(name.replace(".", r"\.") for name in include)
        patterns.append(rf".*(?<!/(?:{alternatives}))$" if False else rf"^(?!.*/(?:{alternatives})$).*$")
    for name in asset.get("exclude_dirs") or []:
        patterns.append(rf"^{name}/.*$")
    return "|".join(f"(?:{p})" for p in patterns) if patterns else None


def rsync_members(local_abs, members, asset, url):
    """Send a directory asset's members, filtering in place when the source tree allows it.

    rsync over the source tree is the cheap path, but it refuses to walk a directory holding a broken
    symlink - wandb/ leaves several behind - and aborts the whole transfer. When that happens the
    members are mirrored into a scratch tree first, by hard link where the filesystem allows one, so
    only the declared files are ever listed.
    """
    pattern = exclude_regex(asset)
    cmd = ["gsutil", "-m", "rsync", "-r"] + (["-x", pattern] if pattern else [])
    result = run(cmd + [local_abs, url.rstrip("/")])
    if result.returncode == 0:
        return result

    with tempfile.TemporaryDirectory() as mirror:
        for full, rel in members:
            staged = os.path.join(mirror, rel)
            os.makedirs(os.path.dirname(staged), exist_ok=True)
            try:
                os.link(full, staged)
            except OSError:
                shutil.copy2(full, staged)
        return run(["gsutil", "-m", "rsync", "-r", mirror, url.rstrip("/")])


def stage_asset(asset, store, present, dry_run):
    """Make sure the store holds one asset. Returns its status and what was found or sent."""
    name = asset["object"]
    url = f"{store}/{name}"
    local = asset.get("local")
    local_abs = os.path.join(REPO_ROOT, local) if local else None

    if asset.get("missing"):
        return {"state": "not-available", "detail": "never kept; see note"}

    if name.endswith("/"):
        if not local_abs or not os.path.isdir(local_abs):
            held = sorted(o for o in present if o.startswith(name))
            if held:
                return {"state": "deposited", "objects": len(held)}
            return {"state": "ABSENT", "detail": "no local copy and nothing in the store"}

        members = members_of(asset)
        wanted = {name + rel for _, rel in members}
        gaps = sorted(wanted - present)
        if not gaps:
            return {"state": "deposited", "objects": len(members)}
        if dry_run:
            return {"state": "would-upload", "objects": len(members), "uploads": len(gaps)}

        result = rsync_members(local_abs, members, asset, url)
        if result.returncode != 0:
            return {"state": "FAILED", "detail": result.stderr.strip().splitlines()[-1:] or ["rsync failed"]}
        present.update(wanted)
        return {"state": "uploaded", "objects": len(members), "uploads": len(gaps)}

    if name in present:
        return {"state": "deposited", **stored_hash(url)}
    if not local_abs or not os.path.isfile(local_abs):
        return {"state": "ABSENT", "detail": "no local copy and not in the bucket"}
    if dry_run:
        return {"state": "would-upload"}

    # gsutil validates the md5 of every upload, so a clean exit means the stored copy matches.
    result = run(["gsutil", "cp", local_abs, url])
    if result.returncode != 0:
        return {"state": "FAILED", "detail": result.stderr.strip().splitlines()[-1:] or ["cp failed"]}
    present.add(name)
    return {"state": "uploaded", **stored_hash(url)}


def manifest_for(figure, assets):
    """The MANIFEST.yaml body for one figure.

    Assets are named, not linked: `object` is the path relative to the root of the deposit, which is
    also the path the asset has in the repository. `fetch.sh` takes that root as its argument.
    """
    entries = []
    for asset in assets:
        entry = {
            "panels": str(asset.get("panels", figure.get("panels", ""))),
            "kind": asset["kind"],
            "object": asset["object"],
            "repo_path": asset["repo_path"],
        }
        for key in ("md5", "also_repo_path", "note", "pending_fig6_revision", "missing"):
            if asset.get(key) is not None:
                entry[key] = asset[key]
        entry["status"] = asset["_status"]
        entries.append(entry)

    body = {
        "figure": figure["id"],
        "title": figure["title"],
        "panels": figure.get("panels"),
        "generated": str(date.today()),
        "generated_by": "reproducibility/stage_figure_assets.py from reproducibility/figure_assets.yaml",
        "scripts": figure.get("scripts") or [],
        "assets": entries,
    }
    body["object_paths_are"] = "relative to the root of the deposit; fetch.sh takes that root as its argument"
    if figure.get("note"):
        body["note"] = figure["note"]
    return body


def fetch_script_for(figure, assets):
    """A fetch.sh that puts every asset of one figure where the analysis scripts look for it.

    The deposit root is the script's argument, not a baked-in location, so the same script works
    against the Figshare record and against any staging copy.
    """
    lines = [
        "#!/usr/bin/env bash",
        "#",
        f"# {figure['title']}",
        "#",
        "# Fetches this figure's datasets and model checkpoints into the paths the scripts in the AMICI",
        "# repository expect. Run it from the repository root, naming the deposit to read from:",
        "#",
        "#   bash fetch.sh <deposit-root>        # or set AMICI_DEPOSIT and run it with no argument",
        "#",
        "# <deposit-root> is the root the object names below are relative to. Already-present files are",
        "# left alone. Generated by reproducibility/stage_figure_assets.py.",
        "",
        "set -euo pipefail",
        "",
        'DEPOSIT="${1:-${AMICI_DEPOSIT:-}}"',
        'if [ -z "$DEPOSIT" ]; then',
        '  echo "usage: bash fetch.sh <deposit-root>   (e.g. a gs:// prefix, or set AMICI_DEPOSIT)" >&2',
        "  exit 1",
        "fi",
        'DEPOSIT="${DEPOSIT%/}"',
        "",
        "if [ ! -d .git ] || [ ! -d src/amici ]; then",
        '  echo "Run this from the root of the AMICI repository." >&2',
        "  exit 1",
        "fi",
        "",
    ]

    for asset in assets:
        if asset.get("missing") or asset["_status"]["state"] in ("not-available", "ABSENT"):
            detail = asset.get("note", "").strip().replace("\n", " ")
            lines += [f"# SKIPPED {asset['repo_path']}: not deposited. {detail}", ""]
            continue
        name, repo_path = asset["object"], asset["repo_path"]
        lines.append(f"echo '==> {repo_path}'")
        if name.endswith("/"):
            lines += [
                f"mkdir -p {repo_path}",
                f'gsutil -m rsync -r "$DEPOSIT/{name.rstrip("/")}" {repo_path}',
            ]
        else:
            lines += [
                f"mkdir -p $(dirname {repo_path})",
                f'gsutil -m cp -n "$DEPOSIT/{name}" {repo_path}',
            ]
            if asset.get("also_repo_path"):
                other = asset["also_repo_path"]
                lines += [
                    f"mkdir -p $(dirname {other})",
                    f'gsutil -m cp -n "$DEPOSIT/{name}" {other}',
                ]
        lines.append("")

    lines += ['echo "Done."', ""]
    return "\n".join(lines)


def readme_for(index, figures):
    """The index README that sits at the top of by_figure/."""
    prefix = index["by_figure_prefix"]
    rows = []
    for figure in figures:
        assets = figure["assets"]
        if figure.get("deferred"):
            state = "deferred"
        elif not assets:
            state = "no checkpoint or dataset needed"
        else:
            gaps = sum(1 for a in assets if a["_status"]["state"] in ("ABSENT", "FAILED"))
            unavailable = sum(1 for a in assets if a["_status"]["state"] == "not-available")
            state = (
                "complete"
                if not gaps and not unavailable
                else f"{len(assets) - gaps - unavailable}/{len(assets)} deposited"
            )
        rows.append(f"| {figure['id']} | {figure['title']} | {figure.get('panels', '')} | {state} |")

    return "\n".join(
        [
            "# AMICI figures: model checkpoints and datasets, by figure",
            "",
            f"Generated {date.today()} by `reproducibility/stage_figure_assets.py` from",
            "`reproducibility/figure_assets.yaml` in the AMICI repository.",
            "",
            "One folder per figure of the manuscript, main figures and Extended Data alike. Each holds:",
            "",
            "- `MANIFEST.yaml` - every dataset and checkpoint that figure needs, which panels need it,",
            "  the name of its object in this deposit, the path the analysis scripts expect it at, its md5",
            "  where one is recorded, and the scripts that draw the figure.",
            "- `fetch.sh` - run it from the root of the AMICI repository, naming this deposit's root, and",
            "  it downloads that figure's assets into those paths, skipping anything already there.",
            "",
            "Assets are named, not linked. Every `object` is a path relative to the root of this deposit,",
            "and it is the same path the asset has in the repository (`<dataset>/data/...`,",
            "`<dataset>/saved_models/...`), so the same name resolves in any copy of the deposit. Each asset",
            "is stored once: figures that share a model name the same object, so the manifests are an index,",
            "not a second copy. Source Data workbooks are separate, under `source_data/`.",
            "",
            "| Figure | Title | Panels | Checkpoints and datasets |",
            "| --- | --- | --- | --- |",
            *rows,
            "",
            "## Using it",
            "",
            "`DEPOSIT` below is the root these object names are relative to -- the prefix you",
            "downloaded this README from, or the Figshare record once it is minted.",
            "",
            "```bash",
            "git clone <the AMICI repository> && cd amici",
            f"gsutil cp $DEPOSIT/{prefix}/fig4/fetch.sh .",
            "bash fetch.sh $DEPOSIT",
            "python xenium/xenium_analysis.py",
            "```",
            "",
            "## What is not here",
            "",
            "- **Derived caches.** The all-cell-type ablation scores, hub caches and grid-search caches are",
            "  1.4-3 GB per dataset and are recomputed from the checkpoint and the dataset by the script",
            "  that needs them. Each manifest says which script writes which cache.",
            "- **Models the analysis trains itself.** The semisynthetic sensitivity studies (ED 3f-j, 6c-h,",
            "  7a, 7e-i, 8, 9a) generate their datasets and train their models inside the script, so there",
            "  is no checkpoint to deposit; the manifest names the script instead.",
            "- **Panels that need a rerun.** Thirteen Extended Data panels never had their numbers written",
            "  to disk. `source_data/README.md` lists them and names the script to rerun for each.",
            "",
            "Provenance for the main-figure checkpoints - W&B sweep and run ids, verified md5 sums, Figshare",
            "file ids, training configs - is in `source_data/checkpoints.yaml` in the repository.",
            "",
        ]
    )


def main():
    """Stage every figure's assets and write the per-figure deposit."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--store",
        required=True,
        metavar="ROOT",
        help="where to stage the deposit, e.g. a gs:// prefix. Passed in rather than recorded, so that "
        "neither this repository nor anything this script writes names a staging location.",
    )
    parser.add_argument("--dry-run", action="store_true", help="report coverage without uploading anything")
    parser.add_argument("--figures", nargs="*", help="only these figure ids")
    parser.add_argument("--skip-manifests", action="store_true", help="stage assets but do not write by_figure/")
    args = parser.parse_args()

    index = load_index()
    store = args.store.rstrip("/")
    prefix = f"{store}/{index['by_figure_prefix'].strip('/')}"

    figures = index["figures"]
    if args.figures:
        wanted = set(args.figures)
        figures = [f for f in figures if f["id"] in wanted]
        if not figures:
            sys.exit(f"no figure matches {sorted(wanted)}")

    print(f"Listing {store} ...")
    present = existing_objects(store)
    print(f"  {len(present)} objects under {', '.join(LISTED_PREFIXES)}\n")

    # One asset can serve several figures; stage it once and reuse the result.
    staged = {}
    for figure in figures:
        print(f"{figure['id']}: {figure['title']}")
        if figure.get("deferred"):
            print("  deferred; no manifest written\n")
            continue
        if not figure["assets"]:
            print("  no checkpoint or dataset needed\n")
            continue
        for asset in figure["assets"]:
            key = asset["object"]
            if key not in staged:
                staged[key] = stage_asset(asset, store, present, args.dry_run)
            asset["_status"] = staged[key]
            status = staged[key]
            extra = ""
            if status.get("objects"):
                extra = f" ({status['objects']} objects"
                if status.get("uploads"):
                    extra += f", {status['uploads']} new"
                extra += ")"
            if status.get("detail"):
                extra = f" -- {status['detail']}"
            print(f"  [{status['state']}] {asset['repo_path']}{extra}")
        print()

    failures = [k for k, v in staged.items() if v["state"] in ("ABSENT", "FAILED")]
    if failures:
        print("Not deposited:")
        for key in failures:
            print(f"  {key}: {staged[key]}")
        print()

    if args.skip_manifests or args.dry_run:
        print("No manifests written." if args.skip_manifests else "Dry run; nothing uploaded or written.")
        return 1 if failures else 0

    with tempfile.TemporaryDirectory() as work:
        for figure in figures:
            if figure.get("deferred"):
                continue
            folder = os.path.join(work, figure["id"])
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "MANIFEST.yaml"), "w") as handle:
                yaml.safe_dump(manifest_for(figure, figure["assets"]), handle, sort_keys=False, width=100)
            with open(os.path.join(folder, "fetch.sh"), "w") as handle:
                handle.write(fetch_script_for(figure, figure["assets"]))

        if not args.figures:
            with open(os.path.join(work, "README.md"), "w") as handle:
                handle.write(readme_for(index, index["figures"]))

        result = run(["gsutil", "-m", "cp", "-r", os.path.join(work, "*"), prefix.rstrip("/") + "/"])
        if result.returncode != 0:
            # A glob has to go through the shell for gsutil to expand it the way cp -r expects.
            result = run(f"gsutil -m cp -r {work}/* {prefix.rstrip('/')}/", shell=True)
        if result.returncode != 0:
            print(result.stderr)
            sys.exit("failed to upload the manifests")

    print(f"Manifests written to {prefix}/")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
