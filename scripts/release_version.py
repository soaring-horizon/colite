"""Version shared by packaging scripts and CMake (version.json)."""
import json
from pathlib import Path
import re


def release_version(debian=False):
    value = json.loads((Path(__file__).resolve().parents[1] / "version.json").read_text())
    version, prerelease = value["version"], value["prerelease"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version) or not re.fullmatch(r"[a-zA-Z0-9.]*", prerelease):
        raise ValueError("Invalid version.json")
    return version + (("~" if debian else "-") + prerelease if prerelease else "")


if __name__ == "__main__":
    print(release_version())
