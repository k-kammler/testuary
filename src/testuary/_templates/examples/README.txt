Testuary extension examples

These files run unchanged with an installed Testuary. Run commands from this
directory with --home . so Testuary reads the models.ini here in place of the
one in your user directory, and with --no-project-settings so that the
settings file of a project this directory lies in does not apply.

testuary_example_chat.py a client factory in one file
models.ini           an alias that uses that client factory (protocol = added)
language-profile.json a prompt profile with Python and Java variants
hints.json           per-language hints for the coverage phase
controls.ini         selects the alias, the profile and the hints

1. Set the key the alias names (LOCAL_MODEL_API_KEY).
2. Check a project without contacting the model. The check loads the
   client factory's file, so its top-level code runs, but creates no client:
     testuary check --home . --config controls.ini --no-project-settings \
       --source-dir PATH \
       --python-executable PROJECT_PYTHON --source-files FILE

A language entry in the profile replaces the general operation of the same
component for that language, so each language entry repeats the general text
it keeps. The full formats are on the documentation site:
https://k-kammler.github.io/testuary/extending/model-provider/,
https://k-kammler.github.io/testuary/extending/prompt-profile/ and
https://k-kammler.github.io/testuary/extending/hints/.
