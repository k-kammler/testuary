# Python API

A Python program runs Testuary through the public entry points below.
The [configuration guide](../guide/configuration.md) shows them in use.
They are Testuary's supported Python interface, and its other modules are
internal.

## Version

::: testuary.__version__

::: testuary.RESULTS_VERSION

## Settings

::: testuary.settings.resolve_settings

::: testuary.settings.RunSettings

::: testuary.settings.ConfigurationError

::: testuary.settings.file_locations

::: testuary.settings.project_locations

::: testuary.settings.FileLocations

::: testuary.settings.platform_user_dir

::: testuary.settings.reference

::: testuary.settings.SectionDescription

::: testuary.settings.SettingDescription

::: testuary.settings.setting_values

::: testuary.settings.SettingValue

::: testuary.settings.extension_items

::: testuary.settings.update_file

::: testuary.settings.add_extension

::: testuary.settings.remove_extension

::: testuary.settings.check_project_settings_file

::: testuary.settings.check_project_catalogue_file

## Setup

The questions `testuary setup` asks, answered through one function, and the
model catalogue and a project's settings, changed as the setup changes them.

::: testuary.setup.run_setup

::: testuary.setup.Question

::: testuary.setup.Ask

::: testuary.setup.Say

::: testuary.setup.terminal_ask

::: testuary.setup.PARTS

::: testuary.setup.Probe

::: testuary.setup.Provider

::: testuary.setup.PROVIDERS

::: testuary.setup.SetupResult

::: testuary.setup.SetupError

::: testuary.setup.models

::: testuary.setup.ModelEntry

::: testuary.setup.usable_aliases

::: testuary.setup.set_model

::: testuary.setup.remove_model

::: testuary.setup.project

::: testuary.setup.ProjectEntry

::: testuary.setup.languages

::: testuary.setup.set_project

## Runs

::: testuary.orchestrator.RunRequest

::: testuary.orchestrator.run_pipeline

::: testuary.presentation.view.phase_label

::: testuary.results.recorded_source_sha256

## Observing a run

What `run_pipeline`'s `event_sink` receives, the presenter that draws it
and the observations of LangAdapter's mutation and timing progress, as
the [terminal guide](../guide/terminal.md#for-programs) shows them.

::: testuary.presentation.DisplayEvent

::: testuary.presentation.Metric

::: testuary.presentation.Usage

::: testuary.presentation.TerminalPresenter

::: testuary.presentation.measurement.mutation_progress_event

::: testuary.presentation.measurement.warmed_progress_event

::: testuary.presentation.measurement.warmed_result_event

## Event files

The event file `--events` writes, and its writer for a program whose run
a page shows as a run's, such as a web interface extension's job kind.
The [terminal guide](../guide/terminal.md#saving-the-events-to-a-file)
describes its records. `read_completion` reads a run's
[end-of-run summary](result-files.md#the-end-of-run-summary), the
`RunCompletion` in `results/completion.json`.

::: testuary.presentation.EventFile

::: testuary.presentation.EVENT_FILE_SCHEMA

::: testuary.presentation.RunPlan

::: testuary.presentation.events.PlannedPhase

::: testuary.presentation.events.PlannedStep

::: testuary.presentation.events.PlannedLimit

::: testuary.presentation.RunCompletion

::: testuary.presentation.read_completion

## Checks

::: testuary.orchestrator.check

::: testuary.readiness.RunReadiness

::: testuary.readiness.ReadinessItem

::: testuary.readiness.ReadinessLimit

::: testuary.presentation.present_readiness

## Models for your own program

A model you configured for Testuary, used from your own program: a
LangChain chat model for an alias of your model catalogue or a model a
provider serves, with Testuary's retries and request records. The
[models guide](../models-and-prompts.md#use-your-models-from-your-own-program)
shows a call.

::: testuary.models.catalogue_chat_model

::: testuary.models.REQUEST_VALUES

::: testuary.model_requests.TransportExhausted

::: testuary.model_requests.TerminalRequestError

::: testuary.model_requests.ContextOverflow

## Demonstrations

The offline demonstration, played as `testuary --demo` plays it.

::: testuary.demo.play

## Web interface

The local browser interface, started as `testuary-web` starts it.

::: testuary.web.serve

The web interface's extension point, the top-level name `testuary_web` of
a module that `[main] extensions` names
([Extend the web interface](../extending/web.md)).

::: testuary.web.WebExtension

::: testuary.web.WebTab

::: testuary.web.DirType

::: testuary.web.SettingsGroup

::: testuary.web.JobKind

::: testuary.web.NestedRun

::: testuary.web.WebRoute

::: testuary.web.WEB_EXTENSION_API

::: testuary.web.TOOL_CHECK_LABEL

## Hints of extensions

The hints of the extensions the settings name, captured and validated as a
run captures them, and the guidance one extension's hints or a hints file add
to a phase's model request.

::: testuary.hints.extension_hints

::: testuary.hints.request_text

::: testuary.hints.ExtensionHints

::: testuary.hints.PHASES

## Generation records

The readers of a run's generation records, which
[Generation records](result-files.md#generation-records) describes.

::: testuary.generation_evidence.read_suite_slots

::: testuary.generation_evidence.read_cleaned_source

::: testuary.generation_evidence.read_ancestry

::: testuary.generation_evidence.GenerationEvidenceError
