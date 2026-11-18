# Extension points

Testuary has eight extension points. Each is one file, or a few lines in a
file, that the model catalogue or a setting names, and each has its own
guide. `testuary tutorial` walks the model alias, the prompt profile and an
extension's hints on a small project without a model. `testuary init
--examples` copies working files of a client factory, a prompt profile and
hints, with a README that checks a project with them.

| Extension point | What it adds | Where it is named | Guide |
| --- | --- | --- | --- |
| Model alias | a model service, with no code | a section of models.ini, which `testuary setup model` writes. A run selects it with `--model` or `[api] model` | [models.ini fields](../models-and-prompts.md#modelsini-fields) |
| Client factory | a model service that is not OpenAI-compatible | a Python file that an alias of models.ini names with `chat_model`, which selects the protocol `added` | [Your own model client](model-provider.md#client-factories) |
| Prompt profile | your own instructions to the model | a JSON file that `[main] prompt_profile` names | [A prompt profile](prompt-profile.md#selecting-a-profile) |
| Hints | advice for the model that fits your project | a JSON file that `[main] hints_file` names, by default `hints.json` in the project directory | [Hints](hints.md#selecting-a-hints-file) |
| Hints from an extension | advice that comes with a language or a tool | the top-level name `testuary_hints` in a Python file that `[main] extensions` names | [Hints](hints.md#selecting-a-hints-file) |
| Language | test generation for a language Testuary has no code for | the top-level name `langadapter_language` in a Python file that `[main] extensions` names | [Make a language ready](languages.md#the-language-file) |
| A language's project settings | a language's runtime and tools, set for each project | the language adapter's `project_settings` (LangAdapter's `ProjectSetting`), set in `[project]` | [Project settings](languages.md#project-settings) |
| Web interface extension | another product's pages in `testuary-web` | the top-level name `testuary_web` in a Python file that `[main] extensions` names | [Extend the web interface](web.md#naming-the-extension) |

## Three kinds of file

- A Python file that an alias of models.ini names with `chat_model`: the
  client factory, a Python function that returns the chat model Testuary
  sends its requests through.
- A JSON file that `[main] prompt_profile` or `[main] hints_file` names: a
  prompt profile or hints.
- A Python file that `[main] extensions` names. Its top-level names say what
  it adds: `langadapter_language` a language, `testuary_hints` hints and
  `testuary_web` a web interface extension.

Each Python file can also be an installed module, named by its module name
in place of the file, such as `my_package.chat:factory` for `chat_model` or
`my_tool.extension` in `[main] extensions`. Demonstration data are no
extension point: every recording `--demo` plays comes with Testuary, as
[Offline demonstrations](../guide/demonstrations.md) describes.

The table [Relative paths](../guide/configuration.md#relative-paths) lists
where a relative path in each of these settings starts.
