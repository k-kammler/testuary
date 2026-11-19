"""The requirement profile applied to a language's capabilities. A missing
required family stops the run before any phase, saying why. A missing optional
one leaves the run going with its fallback announced and recorded. A phase end
never passes silently over an operation the backend does not implement."""
from __future__ import annotations

import json

import pytest

from langadapter.capabilities import (
    CAPABILITIES, Availability, Capability, CapabilityReport,
)
from langadapter.outcomes import Outcome

from testuary import languages, orchestrator
from testuary import config
from testuary import languages as profile
from testuary.languages.generic import GenericBackend
from testuary.settings import ConfigurationError, RunSettings

#: A capability-report entry and the family it belongs to.
FAMILY_OF = {"suite_runs": "run tests", "coverage": "coverage", "mutation": "mutation",
             "per_test_kill_attribution": "kill attribution",
             "source_merging": "merging", "per_test_evidence": "per-test evidence"}


def _report(**states):
    """A report of every operation, available unless ``states`` says
    otherwise (``name=(availability, reason)``)."""
    return CapabilityReport(
        "toy", "toy-runtime", Outcome.COMPLETED, "", "toy 1", tuple(
            Capability(name, *states.get(name, (Availability.AVAILABLE, "")))
            for name in CAPABILITIES), (), "none", ())


class LimitedBackend(GenericBackend):
    """A Python backend whose capability report a test sets. It touches no
    project environment."""

    def __init__(self, report):
        super().__init__("python", (".py",))
        self.report = report

    def capabilities(self, source_dir=None):
        return self.report

    def check_dependencies(self):
        return []


def test_the_profile_is_leveled_with_a_text_for_every_optional_family():
    data = profile.profile()
    assert data["schema"] == 1 and data["consumer"] == "testuary"
    required = [f["family"] for f in profile.families() if f["tier"] == "required"]
    assert required == ["run tests", "coverage", "mutation"]
    for family in profile.families():
        if family["tier"] == "optional":
            for text in (family["improves"], family["fallback"]):
                assert text and text[0].islower() and not text.endswith(".")


def test_the_profile_names_langadapters_families():
    """Every family of the profile is one of LangAdapter's hook table, and
    the required ones are judged by operations a capability report covers."""
    from langadapter import hooks
    named = {f["family"] for f in profile.families()}
    assert named <= {family.name for family in hooks.families()}
    assert profile.METHOD_CHANGING <= named
    assert profile.required_operations("run tests") == (
        "tests.run_suite", "sources.prune_tests", "sources.test_metrics")
    assert all(profile.assessed(f["family"]) for f in profile.families()
               if f["tier"] == "required")


def test_a_complete_backend_needs_no_fallback(default_run):
    assessment = profile.assess(_report())
    assert assessment.refusals == () and assessment.fallbacks == () and assessment.applied == ()


@pytest.mark.parametrize("capability", ["suite_runs", "coverage", "mutation"])
def test_a_missing_required_family_is_refused_with_its_reason(capability, default_run):
    assessment = profile.assess(_report(**{capability: (
        Availability.UNSUPPORTED, "the toy engine has no such tool")}))
    assert assessment.refusals == (
        f"The language adapter lacks {FAMILY_OF[capability]}, which Testuary needs: "
        "the toy engine has no such tool",)
    assert assessment.applied == ()


def test_missing_setup_names_its_reason(default_run):
    assessment = profile.assess(_report(mutation=(
        Availability.MISSING_SETUP, "cosmic-ray not installed")))
    assert assessment.refusals[0].endswith("needs: cosmic-ray not installed")


@pytest.mark.parametrize("capability", ["per_test_kill_attribution", "source_merging",
                                        "per_test_evidence"])
def test_a_missing_optional_family_leaves_its_fallback(capability, default_run):
    assessment = profile.assess(_report(**{capability: (
        Availability.UNSUPPORTED, "the engine names no killing test")}))
    family = FAMILY_OF[capability]
    fallback = next(f["fallback"] for f in profile.families() if f["family"] == family)
    assert assessment.refusals == ()
    assert [f.family for f in assessment.applied] == [family]
    assert assessment.fallbacks == (
        f"The language adapter lacks {family}: the engine names no killing test. "
        f"Without it, {fallback}.",)
    assert assessment.applied[0].changes_method is (family in profile.METHOD_CHANGING)


def test_a_workflow_needs_only_its_required_families(default_run):
    from testuary.languages import setup_requirements
    requirements = setup_requirements(coverage=True, mutation=False)
    assessment = profile.assess(_report(mutation=(Availability.UNSUPPORTED, "none")),
                                requirements)
    assert assessment.refusals == ()


def test_an_operation_the_report_leaves_out_counts_as_missing(default_run):
    report = CapabilityReport(
        "toy", "toy-runtime", Outcome.COMPLETED, "", "toy 1", tuple(
            Capability(name, Availability.AVAILABLE)
            for name in CAPABILITIES if name != "coverage"),
        (), "none", ())
    assert profile.assess(report).refusals == (
        "The language adapter lacks coverage, which Testuary needs: the toy adapter "
        "does not report coverage",)


def test_applied_fallbacks_are_recorded_per_run_only_when_one_applies(tmp_path):
    with config.use_run_settings(RunSettings(base_dir=tmp_path, environment={})):
        taken = profile.assess(_report(source_merging=(Availability.UNSUPPORTED, "none"))).applied
        assert profile.records("toy") is None
        profile.remember("toy", taken)
        assert profile.records("toy") == [{
            "family": "merging", "level": 3, "reason": "none",
            "fallback": taken[0].fallback}]
        json.dumps(profile.records("toy"))
    with config.use_run_settings(RunSettings(base_dir=tmp_path, environment={})):
        assert profile.records("toy") is None


def test_a_new_run_never_sees_an_earlier_runs_fallbacks(monkeypatch, default_run):
    """The store follows the run's scope object, not its id, which a later
    scope can reuse once the earlier run has ended."""
    taken = profile.assess(_report(source_merging=(Availability.UNSUPPORTED, "none"))).applied
    first, second = object(), object()
    monkeypatch.setattr(config, "active_run_scope", lambda: first)
    profile.remember("toy", taken)
    assert profile.records("toy") is not None
    monkeypatch.setattr(config, "active_run_scope", lambda: second)
    assert profile.records("toy") is None
    monkeypatch.setattr(config, "active_run_scope", lambda: first)
    assert profile.records("toy") is None


def _preflight(monkeypatch, tmp_path, backend):
    """The run's refusal, or None, and the lines the preflight logged."""
    monkeypatch.setattr(languages, "get_backend", lambda language: backend)
    messages = []
    monkeypatch.setattr(orchestrator, "log_print",
                        lambda message, **kwargs: messages.append(message))
    try:
        orchestrator._preflight_run_dependencies({"python"}, tmp_path)
    except ConfigurationError as refused:
        return str(refused), messages
    return None, messages


def test_a_backend_without_a_required_family_stops_the_run(monkeypatch, tmp_path, default_run):
    refusal, messages = _preflight(monkeypatch, tmp_path, LimitedBackend(
        _report(mutation=(Availability.UNSUPPORTED, "not in this engine"))))
    assert "The language adapter lacks mutation, which Testuary needs: not in this engine" in refusal
    assert "No phase ran" in refusal and messages == []


def test_a_backend_without_merging_runs_with_its_fallback_announced(monkeypatch, tmp_path, default_run):
    refusal, messages = _preflight(monkeypatch, tmp_path, LimitedBackend(_report(
        source_merging=(Availability.UNSUPPORTED, "not in this engine"))))
    assert refusal is None
    assert len(messages) == 1
    assert "The language adapter lacks merging: not in this engine. Without it, the better " \
        "single candidate is kept" in messages[0]


def test_a_complete_backend_passes_the_preflight_silently(monkeypatch, tmp_path, default_run):
    assert _preflight(monkeypatch, tmp_path, LimitedBackend(_report())) == (None, [])


def test_a_backend_without_a_report_stops_the_run(monkeypatch, tmp_path):
    class Unreported(LimitedBackend):
        def capabilities(self, source_dir=None):
            raise NotImplementedError("the toy backend reports no capabilities")

    refusal, _ = _preflight(monkeypatch, tmp_path, Unreported(_report()))
    assert ("capability report failed (NotImplementedError: the toy backend "
            "reports no capabilities)" in refusal)


def test_the_mutation_phase_end_fails_on_an_unimplemented_operation(
        monkeypatch, tmp_path, default_run):
    """An unimplemented merge fails the phase end rather than delivering the
    suite without merging and selection."""
    from testuary import consolidation as consolidate
    from testuary import measurement as metrics

    def unimplemented(*_args, **_options):
        raise NotImplementedError("merging is not implemented for toy")

    monkeypatch.setattr(consolidate, "consolidate_phase_end", unimplemented)
    with pytest.raises(RuntimeError, match="mutation phase end cannot merge"):
        metrics.consolidate_test_suite(
            test_output_dir=str(tmp_path), coverage_file_name="test_a.py",
            language="python",
            target_file_path=str(tmp_path / "a.py"),
            source_dir=str(tmp_path),
            candidate1_source="def test_a():\n    pass\n",
            candidate2_source="def test_b():\n    pass\n",
            reduction=config.solver_settings(config.get_agent_config("test_coverage")))


def _report_without(operation, **states):
    """A report whose every operation is supported except ``operation``."""
    from langadapter.capabilities import OPERATION_NAMES, OperationCapability
    from dataclasses import replace
    return replace(
        _report(**states),
        operations=tuple(OperationCapability(name, name != operation,
                                             "" if name != operation else "not implemented")
                         for name in OPERATION_NAMES))


def _with_setting(monkeypatch, section, key, value):
    """``config.load_config`` with ``key`` of ``section`` set to ``value``."""
    original = config.load_config

    def load(name=None):
        values = original(name)
        if name in (section if isinstance(section, tuple) else (section,)):
            values = {**values, key: value}
        return values

    monkeypatch.setattr(config, "load_config", load)


def test_the_code_smells_fallback_applies_only_with_measure_lint(monkeypatch, default_run):
    """A language without a linter (Java) records no code smells fallback in a
    run that does not fill the lint columns."""
    report = _report_without("sources.code_smells")
    assert "code smells" not in {f.family for f in profile.assess(report).applied}
    _with_setting(monkeypatch, "finishing", "measure_lint", "true")
    assert "code smells" in {f.family for f in profile.assess(report).applied}


def test_the_delivered_runs_fallback_applies_only_with_the_reverse_order_note(monkeypatch, default_run):
    """Runs as delivered are the normal run only of a language whose runs
    as delivered take a session seed, which offers them, so for a language
    without them they serve only the loops' reverse-order note."""
    report = _report_without("tests.run_as_delivered")
    assert "delivered runs" not in {f.family for f in profile.assess(report).applied}
    _with_setting(monkeypatch, ("coverage",), "report_reverse_order_failures", "true")
    assert "delivered runs" in {f.family for f in profile.assess(report).applied}


def test_only_a_language_that_runs_the_reverse_order_offers_it(default_run):
    """Python runs a suite as delivered in reverse order. Java's adapter
    refuses that order, JUnit ordering a class's tests itself."""
    assert languages.get_backend("python").offers_reverse_order()
    assert not languages.get_backend("java").offers_reverse_order()


def test_the_kill_attribution_fallback_applies_only_when_mutation_is_measured(default_run):
    from testuary.languages import setup_requirements
    report = _report(per_test_kill_attribution=(Availability.UNSUPPORTED, "no killers"))
    without = profile.assess(report, setup_requirements(coverage=True, mutation=False))
    assert without.applied == () and without.fallbacks == ()
    measured = profile.assess(report, setup_requirements(coverage=True, mutation=True))
    assert [f.family for f in measured.applied] == ["kill attribution"]


def test_the_readiness_report_lists_only_fallbacks_in_use(default_run):
    from langadapter.capabilities import OperationCapability
    from testuary import readiness
    operations = {"sources.code_smells": (
        OperationCapability("sources.code_smells", False, "not implemented"),
        {"reason": "not implemented"})}
    assert readiness._fallback_items(operations, lambda value: value, mutation=True) == []


def test_one_judge_decides_whether_a_family_is_there():
    """Testuary's run, its readiness report and LangAdapter's own check
    judge a family by one rule, LangAdapter's: the profile reads it, with
    its reason texts, and keeps only its tiers and fallback words."""
    from langadapter import check

    assert profile.family_issue is check.family_issue
    assert profile.assessed is check.assessed
    assert profile.family_hooks is check.family_hooks
    assert profile.hook_issue is check.hook_issue
