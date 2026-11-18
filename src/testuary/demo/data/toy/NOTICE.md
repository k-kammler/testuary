# Demonstration Data Notice

This directory holds the Toy data of `testuary --demo --demo-language toy`,
the closing run of `langadapter tutorial --for testuary`, recorded in one
real run of this Testuary release on the file `calc.toy` of LangAdapter's
tutorial example (run of 2026-11-12, model deepseek-v4.1-flash:1017 on Ollama Cloud).

- `project.json` holds `calc.toy` of LangAdapter's tutorial example
  project, unmodified. It is part of LangAdapter, under the MIT License.
- `scenario.json.gz` holds the answers the model and every operation that
  started a process gave during that run. The paths of the recording
  machine are stored as placeholders.
- `preset.ini` and `models.ini` hold the run's settings and its key-free
  model alias.

Installed playback needs only the files in this directory and the Toy
adapter of an installed LangAdapter. It runs no model, test, coverage or
mutation tool and contacts no service.
