# Demonstration Data Notice

This directory holds the data of `testuary --demo --demo-data tutorial`, the
closing run of `testuary tutorial`, recorded in one real run of this
Testuary release on the file `tenacity/stop.py` of
[tenacity](https://github.com/jd/tenacity) with the tutorial's prompt
profile and hints (run of 2026-11-12, model deepseek-v4.1-flash:1017 on Ollama Cloud).

- `project.json` holds `tenacity/stop.py` and `tenacity/__init__.py`, as
  of jd/tenacity commit `8be01b1daf2010566fd936a5c2214efefa43edff`,
  unmodified. Tenacity is under the Apache License 2.0, in
  `LICENSE-tenacity.txt`. Each file keeps its own copyright notice.
- `prompt_profile.json` and `hints.json` are the tutorial's prompt profile
  and hints, as the run read them.
- `scenario.json.gz` holds the answers the model and every operation that
  started a process gave during that run. The paths of the recording
  machine are stored as placeholders.
- `preset.ini` and `models.ini` hold the run's settings and its key-free
  model alias.

Installed playback needs only the files in this directory. It runs no
model, test, coverage or mutation tool and contacts no service.

Testuary's MIT license covers its own code and grants no rights over the
material above. `THIRD_PARTY_NOTICES.md`, distributed with Testuary, lists
these notices. Testuary's license does not relicense the embedded source or
the recorded material.
