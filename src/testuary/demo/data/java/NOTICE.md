# Demonstration Data Notice

This directory holds the Java data of `testuary --demo`, recorded in one
real run of this Testuary release on the file
`src/main/java/org/skyscreamer/jsonassert/JSONCompare.java` of
[JSONassert](https://github.com/skyscreamer/JSONassert) (run of 2026-11-12,
model deepseek-v4.1-flash:1017 on Ollama Cloud).

- `project.json` holds `JSONCompare.java`, as of skyscreamer/JSONassert
  commit `e81c16c59ce0860f97a65d871589ab2337370c4b`, unmodified. JSONassert
  is under the Apache License 2.0 in `LICENSE-jsonassert.txt` and
  distributes no NOTICE file.
- `scenario.json.gz` holds the answers the model and every operation that
  started a process gave during that run. The paths of the recording
  machine are stored as placeholders.
- `preset.ini` and `models.ini` hold the run's settings and its key-free
  model alias.

Installed playback needs only the files in this directory. It runs no
model, JDK, build, test, coverage or mutation tool and contacts no service.

Testuary's MIT license covers its own code and grants no rights over the
material above. `THIRD_PARTY_NOTICES.md`, distributed with Testuary, lists
these notices. Testuary's license does not relicense the embedded source or
the recorded material.
