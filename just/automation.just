root := justfile_directory()
script := root / ".automation" / "bin" / "automation_upgrade.py"

version:
    python3 '{{script}}' version

check-update source:
    python3 {{quote(script)}} check-update --source {{quote(source)}}

upgrade source:
    python3 {{quote(script)}} upgrade --source {{quote(source)}}
