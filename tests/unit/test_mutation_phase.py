"""On-disk suite consolidation and mutation measurement helpers.

Cases cover merged and reduced candidates, preserved inputs, mutation
attribution, candidate selection and per-starter-suite mutation scores."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

from testuary import config
from testuary import measurement as tm
from per_suite_double import placed

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# consolidate_test_suite: consolidation after the mutation phase.

def _consolidation_project(tmp_path: Path):
    """A real project whose two suites share one coverage footprint but
    hold distinct kills, so the kill-aware cover is observable."""
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "mod.py").write_text(
        "def f(x):\n    return x + 1\n", encoding="utf-8")
    (proj / "pyproject.toml").write_text("", encoding="utf-8")
    test_dir = tmp_path / "rank1" / "tests"
    _write(test_dir / "test_mod.py",
           "from mod import f\n\n"
           "def test_weak():\n    assert f(1) is not None\n")
    _write(test_dir / "test_mod_mutation.py",
           "from mod import f\n\n"
           "def test_weak():\n    assert f(1) is not None\n\n"
           "def test_strong():\n    assert f(1) == 2\n")
    return proj, test_dir


def _suites(test_dir):
    """The inherited suite, the mutation phase's and the packaged suite
    reduction settings, as the phase hands them to the consolidation."""
    return {"candidate1_source": (test_dir / "test_mod.py").read_text(encoding="utf-8"),
            "candidate2_source": (test_dir / "test_mod_mutation.py").read_text(encoding="utf-8"),
            "reduction": config.solver_settings(config.get_agent_config("test_coverage"))}


def _consolidate(proj, test_dir, **arguments):
    """consolidate_test_suite on the project's two suites, as the mutation
    phase calls it, with ``arguments`` added."""
    return tm.consolidate_test_suite(
        test_output_dir=str(test_dir), coverage_file_name="test_mod.py", **_suites(test_dir),
        language="python", target_file_path=str(proj / "mod.py"), source_dir=str(proj),
        timeout=120, **arguments)


def _pick(label, seen):
    """A stand-in for the shared selector: it keeps each call's candidates
    and keyword arguments in ``seen`` and picks the candidate ``label``."""
    def pick(candidates, **arguments):
        seen.append({"candidates": list(candidates), **arguments})
        return next(candidate for candidate in candidates if candidate.label == label)
    return pick


def _km(mutants):
    return placed({"mutants": mutants})


def _coverage_sidecar():
    return {
        "line": 100.0,
        "branch": 100.0,
        "covered_statement_lines": [1, 2],
        "covered_branch_arcs": [],
    }


def _measurement_identity(proj: Path) -> dict:
    return {
        "target_source_sha256": hashlib.sha256(
            (proj / "mod.py").read_bytes()
        ).hexdigest(),
        "execution_name": "test_mod.py",
        "catalogue": "full",
    }


#: A make project with one C function under test.
C_PROJECT = {
    "Makefile": ("CPPFLAGS = -Iinclude\n\napp: src/main.o src/calc.o\n\t$(CC) -o $@ $^\n\n"
                 "src/%.o: src/%.c\n\t$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@\n"),
    "include/calc.h": "#ifndef CALC_H\n#define CALC_H\n\nlong clamp(long x, long lo, long hi);\n\n#endif\n",
    "src/calc.c": ("#include \"calc.h\"\n\nlong clamp(long x, long lo, long hi) {\n"
                   "    if (x < lo) {\n        return lo;\n    }\n"
                   "    if (x > hi) {\n        return hi;\n    }\n    return x;\n}\n"),
    "src/main.c": "#include \"calc.h\"\n\nint main(void) {\n    return (int)clamp(0, 0, 1);\n}\n",
}
#: Two utest.h tests that share a static variable. The first lowers the
#: limit, so the second, which passes either way, reaches ``return x;``
#: (line 10 of calc.c) when it runs alone, and not after the first in the
#: one program of the suite run.
C_SHARED_STATIC_TESTS = """\
#include "utest.h"
#include "calc.h"

static long limit = 10;

UTEST(clamp, lowers_the_limit) {
    limit = 0;
    ASSERT_EQ(0L, clamp(5, 0, limit));
}

UTEST(clamp, within_the_limit_or_at_it) {
    long r = clamp(5, 0, limit);
    ASSERT_TRUE(r == 5 || r == 0);
}

UTEST_MAIN()
"""


#: Real C tests run on Linux with gcc, gcov, make and sh, and with
#: LangAdapter's optional c extra, whose parser checks the C sources.
needs_c = pytest.mark.skipif(
    not sys.platform.startswith("linux")
    or not all(shutil.which(tool) for tool in ("gcc", "gcov", "make", "sh"))
    or importlib.util.find_spec("tree_sitter_c") is None,
    reason="C tests run on Linux with gcc, gcov, make and sh, and LangAdapter's c extra")
#: The C mutation check runs Mull's mull-runner-N from PATH.
needs_mull = pytest.mark.skipif(
    not any(next(Path(folder).glob("mull-runner-*"), None)
            for folder in os.environ.get("PATH", "").split(os.pathsep) if folder),
    reason="the C mutation check runs Mull's mull-runner-N")


def _c_phase_end(tmp_path: Path, project: dict, target: str, candidate1: str,
                 candidate2: str, *, kills: bool = False) -> tuple:
    """The mutation phase end of the real C make project ``project``, whose
    target is src/<target>.c, with the two candidate suites, measuring kills
    with the phase end's own mutation runner when ``kills``. Returns its
    consolidation record and the delivered suite."""
    root = tmp_path / target
    for name, text in project.items():
        _write(root / name, text)
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    target_file = str(root / "src" / f"{target}.c")
    runner = tm.phase_end_runner(
        "mutation", always=True, language="c", target_file=target_file, source_dir=str(root),
        config=dict(config.get_agent_config("test_mutation", parent="test_coverage")),
    ) if kills else nullcontext()
    with runner as run_mutation_on:
        tm.consolidate_test_suite(
            test_output_dir=str(test_dir), coverage_file_name=f"test_{target}.c",
            candidate1_source=candidate1, candidate2_source=candidate2,
            run_mutation_on=run_mutation_on,
            reduction=config.solver_settings(config.get_agent_config("test_coverage")),
            language="c", target_file_path=target_file, source_dir=str(root),
            matrices_dir=str(tmp_path / "records"), timeout=120)
    record = json.loads((tmp_path / "records" / "consolidation_record.json").read_text(
        encoding="utf-8"))
    return record, (test_dir / f"test_{target}.c").read_text(encoding="utf-8")


@needs_c
def test_a_c_suite_whose_tests_share_a_static_keeps_its_measured_coverage(
        tmp_path: Path) -> None:
    """A real C suite whose second test covers ``return x;`` alone and not
    in the one program of the suite run: the phase end keeps the run's
    coverage, records the line only the per-test evidence covers, and
    delivers both tests."""
    record, delivered = _c_phase_end(tmp_path, C_PROJECT, "calc", C_SHARED_STATIC_TESTS,
                                     C_SHARED_STATIC_TESTS)
    difference = record["selection_audit"]["merged_evidence_difference"]
    assert difference["per_test_evidence"]["covered_lines_only"] == [10]
    assert difference["measured"]["covered_lines_only"] == []
    assert "lowers_the_limit" in delivered and "within_the_limit_or_at_it" in delivered


#: A make project whose target reads a float by its width and says whether
#: the width is a wide one. A width outside the enumeration reaches the
#: switch's unreachable default.
C_FLOAT_PROJECT = {
    "Makefile": C_PROJECT["Makefile"].replace("calc", "ctrl"),
    "include/ctrl.h": """\
#ifndef CTRL_H
#define CTRL_H

typedef enum { CTRL_FLOAT_0, CTRL_FLOAT_16, CTRL_FLOAT_32, CTRL_FLOAT_64 } ctrl_width;

typedef struct {
    ctrl_width width;
    float half;
    float single;
    double full;
} ctrl_item;

double ctrl_get_float(const ctrl_item *item);
int ctrl_is_wide(const ctrl_item *item);

#endif
""",
    "src/ctrl.c": """\
#include <math.h>
#include "ctrl.h"

double ctrl_get_float(const ctrl_item *item) {
    switch (item->width) {
        case CTRL_FLOAT_0:
            return NAN;
        case CTRL_FLOAT_16:
            return item->half;
        case CTRL_FLOAT_32:
            return item->single;
        case CTRL_FLOAT_64:
            return item->full;
        default:
            __builtin_unreachable();
    }
}

int ctrl_is_wide(const ctrl_item *item) {
    return item->width >= CTRL_FLOAT_32;
}
""",
    "src/main.c": """\
#include "ctrl.h"

int main(void) {
    ctrl_item item = {CTRL_FLOAT_32, 0.0f, 1.0f, 0.0};
    return ctrl_is_wide(&item) - 1;
}
""",
}
#: Tests of C_FLOAT_PROJECT that pass on the project's build and on Mull's.
C_FLOAT_TESTS = """\
#include "utest.h"
#include "ctrl.h"

UTEST(floats_ctrls, get_float_of_a_half) {
    ctrl_item item = {CTRL_FLOAT_16, 0.5f, 0.0f, 0.0};
    ASSERT_TRUE(ctrl_get_float(&item) == 0.5);
}

UTEST(floats_ctrls, get_float_of_a_single) {
    ctrl_item item = {CTRL_FLOAT_32, 0.0f, 1.5f, 0.0};
    ASSERT_TRUE(ctrl_get_float(&item) == 1.5);
}

UTEST(floats_ctrls, get_float_of_a_double) {
    ctrl_item item = {CTRL_FLOAT_64, 0.0f, 0.0f, 2.25};
    ASSERT_TRUE(ctrl_get_float(&item) == 2.25);
}

UTEST(floats_ctrls, get_float_without_a_value) {
    ctrl_item item = {CTRL_FLOAT_0, 0.0f, 0.0f, 0.0};
    double v = ctrl_get_float(&item);
    ASSERT_TRUE(v != v);
}

UTEST(floats_ctrls, a_single_is_wide) {
    ctrl_item item = {CTRL_FLOAT_32, 0.0f, 1.5f, 0.0};
    ASSERT_EQ(1, ctrl_is_wide(&item));
}

UTEST(floats_ctrls, a_half_is_not_wide) {
    ctrl_item item = {CTRL_FLOAT_16, 0.5f, 0.0f, 0.0};
    ASSERT_EQ(0, ctrl_is_wide(&item));
}

UTEST_MAIN()
"""
#: A test that passes on the project's gcc build and ends by a signal on
#: Mull's clang build, where the out-of-range width runs past the switch.
C_OUT_OF_RANGE_WIDTH_TEST = """\
UTEST(floats_ctrls, get_float_with_out_of_range_width) {
    ctrl_item item = {(ctrl_width)7, 0.0f, 0.0f, 0.0};
    double v = ctrl_get_float(&item);
    EXPECT_TRUE(v == v || v != v);
}
"""


@needs_c
@needs_mull
def test_a_c_test_that_fails_only_on_mulls_build_is_removed_at_the_phase_end(
        tmp_path: Path) -> None:
    """A real C suite with a test that passes on the project's build and
    ends by a signal on Mull's build: the phase end's mutation check names
    the test, the phase's own suite loses it, and the rest of that suite is
    measured with a kill matrix."""
    record, delivered = _c_phase_end(
        tmp_path, C_FLOAT_PROJECT, "ctrl", C_FLOAT_TESTS,
        C_FLOAT_TESTS.replace("UTEST_MAIN()", C_OUT_OF_RANGE_WIDTH_TEST + "\nUTEST_MAIN()"),
        kills=True)
    matrix = tmp_path / "records" / "candidate2_kill_matrix.json"
    assert matrix.is_file()
    mutants = json.loads(matrix.read_text(encoding="utf-8"))["mutants"]
    assert any(mutant["status"] == "killed" for mutant in mutants)
    [entry] = [entry for entry in record["measurement_removals"]
               if entry["place"] == "phase_end.candidate"]
    assert (entry["measurement"], entry["round"]) == ("mutation", 1)
    [removed] = entry["removed"]
    assert removed["test"].endswith("floats_ctrls.get_float_with_out_of_range_width")
    assert "get_float_with_out_of_range_width" not in delivered
    assert (tmp_path / "records" / "candidate_standalone_1_removed.c").is_file()


#: A make project whose target gives an item's encoded size by its type and
#: width. A width outside the enumeration reaches the inner switch's
#: unreachable default, after which gcc's counters cannot be read: gcov
#: counts a branch of the outer switch as taken -1.
C_SIZE_PROJECT = {
    "Makefile": C_PROJECT["Makefile"].replace("calc", "size"),
    "include/item.h": """\
#ifndef ITEM_H
#define ITEM_H

#include <stddef.h>

typedef enum { ITEM_UINT, ITEM_NEGINT, ITEM_TEXT, ITEM_FLOAT } item_type;
typedef enum { WIDTH_8, WIDTH_16, WIDTH_32, WIDTH_64 } item_width;

typedef struct {
    item_type type;
    item_width width;
    unsigned long long value;
    size_t length;
} item_t;

size_t item_encoded_size(const item_t *item);

#endif
""",
    "src/size.c": """\
#include "item.h"

size_t item_encoded_size(const item_t *item) {
    switch (item->type) {
        case ITEM_UINT:
        case ITEM_NEGINT:
            switch (item->width) {
                case WIDTH_8:
                    return item->value < 24 ? 1 : 2;
                case WIDTH_16:
                    return 3;
                case WIDTH_32:
                    return 5;
                case WIDTH_64:
                    return 9;
                default:
                    __builtin_unreachable();
            }
        case ITEM_TEXT:
            return 1 + item->length;
        case ITEM_FLOAT:
            return 9;
        default:
            __builtin_unreachable();
    }
}
""",
    "src/main.c": """\
#include "item.h"

int main(void) {
    item_t item = {ITEM_TEXT, WIDTH_8, 0, 2};
    return (int)item_encoded_size(&item) - 3;
}
""",
}
#: Tests of C_SIZE_PROJECT whose counts gcov can read.
C_SIZE_TESTS = """\
#include <setjmp.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include "utest.h"
#include "item.h"

UTEST(serialization, size_of_a_small_uint) {
    item_t item = {ITEM_UINT, WIDTH_8, 5, 0};
    ASSERT_EQ((size_t)1, item_encoded_size(&item));
}

UTEST(serialization, size_of_a_wide_negint) {
    item_t item = {ITEM_NEGINT, WIDTH_32, 70000, 0};
    ASSERT_EQ((size_t)5, item_encoded_size(&item));
}

UTEST(serialization, size_of_a_text) {
    item_t item = {ITEM_TEXT, WIDTH_8, 0, 4};
    ASSERT_EQ((size_t)5, item_encoded_size(&item));
}

UTEST(serialization, size_of_a_float) {
    item_t item = {ITEM_FLOAT, WIDTH_64, 0, 0};
    ASSERT_EQ((size_t)9, item_encoded_size(&item));
}

UTEST_MAIN()
"""
#: A test that drives the unreachable width arm under a recovery point. It
#: passes, and gcov cannot read the counts of a run of it alone.
C_UNREACHABLE_WIDTH_TEST = """\
static sigjmp_buf probe_env;
static volatile size_t probe_result;

static void probe_handler(int sig) {
    (void)sig;
    siglongjmp(probe_env, 1);
}

UTEST(serialization, unreachable_width_arms) {
    struct sigaction sa, old_abrt, old_ill, old_segv;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = probe_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGABRT, &sa, &old_abrt);
    sigaction(SIGILL, &sa, &old_ill);
    sigaction(SIGSEGV, &sa, &old_segv);
    item_t item = {ITEM_UINT, (item_width)0x7f, 5, 0};
    if (sigsetjmp(probe_env, 1) == 0) {
        probe_result = item_encoded_size(&item);
    }
    sigaction(SIGABRT, &old_abrt, NULL);
    sigaction(SIGILL, &old_ill, NULL);
    sigaction(SIGSEGV, &old_segv, NULL);
    item.width = WIDTH_8;
    ASSERT_EQ((size_t)1, item_encoded_size(&item));
}
"""
#: The unreachable width arm driven in a child process. The test passes,
#: and gcov cannot read the counts of a run of it alone.
C_FORK_PROBE_TEST = """\
UTEST(serialization, corrupted_int_width_probe) {
    pid_t pid = fork();
    ASSERT_TRUE(pid >= 0);
    if (pid == 0) {
        item_t item = {ITEM_UINT, (item_width)0x7f, 5, 0};
        (void)item_encoded_size(&item);
        exit(0);
    }
    int status = 0;
    ASSERT_EQ(pid, waitpid(pid, &status, 0));
    if (WIFEXITED(status)) {
        ASSERT_EQ(0, WEXITSTATUS(status));
    } else {
        ASSERT_TRUE(WIFSIGNALED(status));
    }
}
"""
#: A test that drives the outer switch's unreachable default under the
#: recovery point of C_UNREACHABLE_WIDTH_TEST. Beside that test it makes the
#: suite's own counts readable, as libcbor's serialization suite had them,
#: while that test's counts alone stay unreadable.
C_UNREACHABLE_TYPE_TEST = """\
UTEST(serialization, unreachable_type_arm) {
    struct sigaction sa, old_abrt, old_ill, old_segv;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = probe_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGABRT, &sa, &old_abrt);
    sigaction(SIGILL, &sa, &old_ill);
    sigaction(SIGSEGV, &sa, &old_segv);
    item_t item = {(item_type)0x7f, WIDTH_8, 5, 0};
    if (sigsetjmp(probe_env, 1) == 0) {
        probe_result = item_encoded_size(&item);
    }
    sigaction(SIGABRT, &old_abrt, NULL);
    sigaction(SIGILL, &old_ill, NULL);
    sigaction(SIGSEGV, &old_segv, NULL);
    item.type = ITEM_TEXT;
    ASSERT_EQ((size_t)1, item_encoded_size(&item));
}
"""


@needs_c
def test_c_tests_whose_coverage_counts_cannot_be_read_are_removed_at_the_phase_end(
        tmp_path: Path, monkeypatch) -> None:
    """A real C suite with two tests whose gcc counters gcov cannot read:
    the phase end's coverage run names both, the phase's own suite loses
    them, and the rest of that suite is measured."""
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: False)
    record, delivered = _c_phase_end(
        tmp_path, C_SIZE_PROJECT, "size", C_SIZE_TESTS,
        C_SIZE_TESTS.replace("UTEST_MAIN()", "\n".join(
            (C_UNREACHABLE_WIDTH_TEST, C_FORK_PROBE_TEST, "UTEST_MAIN()"))))
    coverage = tmp_path / "records" / "candidate2_coverage_matrix.json"
    assert coverage.is_file()
    assert json.loads(coverage.read_text(encoding="utf-8"))["line"] is not None
    [entry] = [entry for entry in record["measurement_removals"]
               if entry["place"] == "phase_end.candidate"]
    assert (entry["measurement"], entry["round"]) == ("coverage", 1)
    assert sorted(item["test"].rsplit(".", 1)[-1] for item in entry["removed"]) == [
        "corrupted_int_width_probe", "unreachable_width_arms"]
    assert "unreachable_width_arms" not in delivered
    assert "corrupted_int_width_probe" not in delivered
    assert (tmp_path / "records" / "candidate_standalone_1_removed.c").is_file()


@needs_c
@needs_mull
def test_a_c_test_whose_own_counts_cannot_be_read_is_removed_by_the_mutation_check(
        tmp_path: Path) -> None:
    """A real C suite whose own counts gcov reads, with a test whose counts
    alone it cannot read: the phase end's coverage run completes, the
    mutation check's instrument check names that test, the phase's own
    suite loses it, and the rest of that suite is measured with a kill
    matrix."""
    record, delivered = _c_phase_end(
        tmp_path, C_SIZE_PROJECT, "size", C_SIZE_TESTS,
        C_SIZE_TESTS.replace("UTEST_MAIN()", "\n".join(
            (C_UNREACHABLE_WIDTH_TEST, C_UNREACHABLE_TYPE_TEST, "UTEST_MAIN()"))),
        kills=True)
    matrix = tmp_path / "records" / "candidate2_kill_matrix.json"
    assert matrix.is_file()
    mutants = json.loads(matrix.read_text(encoding="utf-8"))["mutants"]
    assert any(mutant["status"] == "killed" for mutant in mutants)
    [entry] = [entry for entry in record["measurement_removals"]
               if entry["place"] == "phase_end.candidate"]
    assert (entry["measurement"], entry["round"]) == ("mutation", 1)
    [removed] = entry["removed"]
    assert removed["test"].endswith("serialization.unreachable_width_arms")
    assert "unreachable_width_arms" not in delivered
    assert (tmp_path / "records" / "candidate_standalone_1_removed.c").is_file()


#: A make project whose target counts up to a number and says whether a
#: number is big. Adding 1 as subtracting it makes count_up run on.
C_COUNT_PROJECT = {
    "Makefile": C_PROJECT["Makefile"].replace("calc", "count"),
    "include/count.h": ("#ifndef COUNT_H\n#define COUNT_H\n\nlong count_up(long n);\n"
                        "int is_big(long x);\n\n#endif\n"),
    "src/count.c": ("#include \"count.h\"\n\nlong count_up(long n) {\n    long i = 0;\n"
                    "    while (i != n) {\n        i = i + 1;\n    }\n    return i;\n}\n\n"
                    "int is_big(long x) {\n    return x >= 1000;\n}\n"),
    "src/main.c": "#include \"count.h\"\n\nint main(void) {\n    return is_big(count_up(1));\n}\n",
}
#: Thirty tests of C_COUNT_PROJECT whose process waits 0.3 s when it starts:
#: the one program of the suite run waits once, the tests run alone wait
#: each. Every test asks whether a small number is big, and one counts up
#: to three.
C_COUNT_TESTS = (
    "#include <time.h>\n#include \"utest.h\"\n#include \"count.h\"\n\n"
    "__attribute__((constructor)) static void slow_start(void) {\n"
    "    struct timespec pause = {0, 300000000L};\n    nanosleep(&pause, NULL);\n}\n\n"
    "UTEST(count, up_to_three) {\n    ASSERT_EQ(0, is_big(3));\n    ASSERT_EQ(3L, count_up(3));\n}\n\n"
    + "".join(f"UTEST(count, small_{n}) {{\n    ASSERT_EQ(0, is_big({n}));\n}}\n\n"
              for n in range(1, 30))
    + "UTEST_MAIN()\n")


@needs_c
@needs_mull
def test_a_c_mutant_that_no_test_kills_survives_tests_slower_alone_than_together(
        tmp_path: Path) -> None:
    """A real C suite of thirty tests that take 9 s run alone, as the C
    engine runs a mutant's tests, and well under a second in the one program
    of the suite run: with a factor of 2 and a minimum of 2 s, a mutant that
    no test kills runs all thirty tests within its time limit and survives,
    since the limit follows the engine's own run of the tests alone, and a
    mutant that makes one test hang reads timeout."""
    root = tmp_path / "count"
    for name, text in C_COUNT_PROJECT.items():
        _write(root / name, text)
    test_dir = tmp_path / "tests"
    _write(test_dir / "test_count.c", C_COUNT_TESTS)
    settings = dict(config.get_agent_config("test_mutation", parent="test_coverage"))
    settings.update(per_mutant_timeout_factor=2.0, min_per_mutant_timeout_seconds=2)
    with tm.union_mutation_runner(language="c", target_file=str(root / "src" / "count.c"),
                                  source_dir=str(root), config=settings) as run_mutation_on:
        matrix = run_mutation_on(test_dir)
    rows = {(row["operator"], row["line"]): row for row in matrix["mutants"]}
    survived = rows[("cxx_ge_to_gt", 12)]
    assert (survived["status"], survived["tests_run"]) == ("survived", 30)
    assert rows[("cxx_add_to_sub", 6)]["status"] == "timeout"
    applied = matrix["per_mutant_timeout"]
    assert applied["inputs"]["factor"] == 2.0 and applied["value"] > 2


def test_consolidation_kill_aware_cover_reduces(tmp_path: Path, monkeypatch) -> None:
    """With every kill attributed, the cover keeps the strong test and drops
    the weak duplicate. The reduced matrix is returned for finishing."""
    from testuary import config as config_module
    from testuary import consolidation as consolidate_mod
    monkeypatch.setattr(config_module, "consolidation_mutation_method", lambda: "derive")
    monkeypatch.setattr(
        consolidate_mod,
        "pick_phase_result",
        lambda candidates, **kwargs: next(
            candidate for candidate in candidates
            if candidate.label == "reduced"
        ),
    )
    proj, test_dir = _consolidation_project(tmp_path)
    calls = {"n": 0}

    def run_mutation_on(d):
        calls["n"] += 1
        # Cosmic Ray names the killer by the test file it ran, here the merged
        # test_*.py, not the conftest.py the runner adds to the directory.
        pyfile = next(Path(d).glob("test_*.py"))
        return _km([
            {"id": "m1", "status": "killed",
             "killer": f"{pyfile.name}::test_strong"},
        ])

    res = _consolidate(
        proj, test_dir,
        run_mutation_on=run_mutation_on,
        candidate1_km=_km([
            {"id": "m1", "status": "survived", "killer": None},
        ]),
        candidate1_coverage=_coverage_sidecar(),
        candidate1_measurement_identity=_measurement_identity(proj),
        measurement_identity=_measurement_identity(proj),
    )
    assert res is not None
    # Candidate 1 reuses its bound matrix. Candidate 2 and the changed merged
    # suite are measured once each. Selection itself measures no mutants.
    assert calls["n"] == 2
    merged = (test_dir / "test_mod.py").read_text(encoding="utf-8")
    assert "test_strong" in merged
    # test_weak is coverage-subsumed and kills nothing, so the cover drops it.
    assert "def test_weak" not in merged
    pre = test_dir.parent / "_pre_merge"
    assert (pre / "candidate_inherited.py").exists()
    assert (pre / "candidate_agent.py").exists()
    assert res.kill_matrix is not None          # reusable by metrics


def test_stale_inherited_matrix_is_remeasured_before_selection(
    tmp_path: Path, monkeypatch,
) -> None:
    """A cross-job identity change discards evidence, not the candidate."""
    from testuary import config as config_module
    from testuary import consolidation as consolidate_mod

    monkeypatch.setattr(config_module, "consolidation_mutation_method", lambda: "derive")
    monkeypatch.setattr(
        consolidate_mod,
        "pick_phase_result",
        lambda candidates, **kwargs: next(
            candidate for candidate in candidates
            if candidate.label == "merged"
        ),
    )
    proj, test_dir = _consolidation_project(tmp_path)
    calls = []

    def run_mutation_on(directory):
        calls.append(Path(directory))
        return _km([{
            "id": "m1", "status": "survived", "killer": None,
        }])

    current_identity = {
        **_measurement_identity(proj),
        "config_sha256": "current",
    }
    logs = []
    result = _consolidate(
        proj, test_dir,
        run_mutation_on=run_mutation_on,
        candidate1_km=_km([{
            "id": "m1", "status": "killed", "killer": "old",
        }]),
        candidate1_coverage=_coverage_sidecar(),
        candidate1_measurement_identity={
            **_measurement_identity(proj),
            "config_sha256": "stale",
        },
        measurement_identity=current_identity,
        log_fn=logs.append,
    )

    assert result is not None
    assert len(calls) == 3
    assert any("the unchanged inherited suite is measured again" in line
               for line in logs)


def test_consolidation_unsound_matrix_merges_without_reduction(
    tmp_path, monkeypatch,
):
    """A killed mutant without a named killer makes a kill-aware cover
    unsound. The inputs are merged whole with only byte-identical
    deduplication, and the merged suite's matrix is still returned as a valid
    measurement."""
    from testuary import consolidation as consolidate_mod

    monkeypatch.setattr(
        consolidate_mod,
        "pick_phase_result",
        lambda candidates, **kwargs: next(
            candidate for candidate in candidates
            if candidate.label == "merged"
        ),
    )
    proj, test_dir = _consolidation_project(tmp_path)
    res = _consolidate(
        proj, test_dir,
        run_mutation_on=lambda d: _km([
            {"id": "m1", "status": "killed", "killer": None},
        ]),
        candidate1_km=_km([
            {"id": "m1", "status": "killed", "killer": None},
        ]),
        candidate1_coverage=_coverage_sidecar(),
        candidate1_measurement_identity=_measurement_identity(proj),
        measurement_identity=_measurement_identity(proj),
    )
    assert res is not None
    merged = (test_dir / "test_mod.py").read_text(encoding="utf-8")
    assert "test_strong" in merged
    assert "def test_weak" in merged            # no reduction happened
    assert merged.count("def test_weak") == 1   # identical dedup still on
    assert res.kill_matrix is not None


def test_consolidation_without_runner_reduces_on_coverage(tmp_path):
    """Without a mutation runner the cover reduces on coverage alone. The
    strong test alone covers the negative branch and subsumes the weak
    test's coverage, so the weak one is dropped whatever the two tests'
    measured times (with equal coverage the cover would pick by time), and
    no kill matrix is returned."""
    proj, test_dir = _consolidation_project(tmp_path)
    (proj / "mod.py").write_text(
        "def f(x):\n    if x > 0:\n        return x + 1\n    return 0\n",
        encoding="utf-8")
    _write(test_dir / "test_mod_mutation.py",
           "from mod import f\n\n"
           "def test_weak():\n    assert f(1) is not None\n\n"
           "def test_strong():\n    assert f(1) == 2\n    assert f(-1) == 0\n")
    res = _consolidate(
        proj, test_dir,
        run_mutation_on=None,
    )
    assert res is not None
    merged = (test_dir / "test_mod.py").read_text(encoding="utf-8")
    assert "test_strong" in merged
    assert "def test_weak" not in merged          # coverage-subsumed, dropped
    assert res.kill_matrix is None


@pytest.mark.parametrize(
    (
        "merged_kills",
        "candidate1_kills",
        "candidate2_kills",
        "expected_label",
    ),
    [
        (frozenset(), frozenset({"m0", "m1"}), frozenset({"m0"}),
         "standalone_0"),
        (frozenset(), frozenset({"m0"}), frozenset({"m0", "m1"}),
         "standalone_1"),
    ],
)
def test_green_merge_regression_remeasures_and_keeps_best_standalone(
    tmp_path,
    monkeypatch,
    merged_kills,
    candidate1_kills,
    candidate2_kills,
    expected_label,
):
    """A selected standalone is persisted as an ordinary score winner."""
    from testuary import consolidation as consolidate

    proj, test_dir = _consolidation_project(tmp_path)

    def score_matrix(killed):
        return _km([
            {
                "id": f"m{i}",
                "status": (
                    "killed" if f"m{i}" in killed else "survived"
                ),
            }
            for i in range(3)
        ])

    merge_km = score_matrix(merged_kills)
    candidate1_km = score_matrix(candidate1_kills)
    candidate2_km = score_matrix(candidate2_kills)
    coverage_src = (test_dir / "test_mod.py").read_text(encoding="utf-8")
    mutation_src = (
        test_dir / "test_mod_mutation.py"
    ).read_text(encoding="utf-8")
    phase = consolidate.PhaseConsolidation(
        merged_source="def test_merged():\n    assert True\n",
        reduced_source="def test_merged():\n    assert True\n",
        reduction_valid=True,
        merged_km=merge_km,
        reduced_km=merge_km,
        candidate2_km=candidate2_km,
        selected_tests=1,
        pool_tests=2,
    )
    monkeypatch.setattr(
        consolidate, "consolidate_phase_end", lambda *args, **kwargs: phase
    )

    selected_index = int(expected_label.rsplit("_", 1)[1])
    selected_source = [coverage_src, mutation_src][selected_index]
    selected_matrix = [candidate1_km, candidate2_km][selected_index]
    seen = []
    monkeypatch.setattr(consolidate, "pick_phase_result", _pick(expected_label, seen))
    matrices = tmp_path / "matrices"
    result = _consolidate(
        proj, test_dir,
        run_mutation_on=lambda _directory: merge_km,
        candidate1_km=candidate1_km,
        candidate1_measurement_identity=_measurement_identity(proj),
        measurement_identity=_measurement_identity(proj),
        matrices_dir=str(matrices),
    )

    assert len(seen) == 1
    assert result is not None
    assert result.winner == "single_suite"
    assert result.winner_source_label == expected_label
    assert result.kill_matrix is selected_matrix
    assert (test_dir / "test_mod.py").read_text(
        encoding="utf-8"
    ) == selected_source
    record = json.loads(
        (matrices / "consolidation_record.json").read_text(encoding="utf-8")
    )
    assert record["fallback_reason"] is None
    assert record["fallback_candidates"] == []


def test_a_green_merge_goes_through_the_shared_selector(
    tmp_path,
    monkeypatch,
):
    """Each phase-end boundary runs the shared selector, also for a stronger merge."""
    from testuary import consolidation as consolidate

    proj, test_dir = _consolidation_project(tmp_path)

    def score_matrix(killed):
        return _km([
            {
                "id": f"m{i}",
                "status": (
                    "killed" if f"m{i}" in killed else "survived"
                ),
            }
            for i in range(3)
        ])

    merge_km = score_matrix({"m0", "m1", "m2"})
    candidate1_km = score_matrix({"m0"})
    candidate2_km = score_matrix({"m0", "m1"})
    phase = consolidate.PhaseConsolidation(
        merged_source="def test_merged():\n    assert True\n",
        reduced_source="def test_merged():\n    assert True\n",
        reduction_valid=True,
        merged_km=merge_km,
        reduced_km=merge_km,
        candidate2_km=candidate2_km,
        selected_tests=1,
        pool_tests=2,
    )
    monkeypatch.setattr(
        consolidate, "consolidate_phase_end", lambda *args, **kwargs: phase
    )
    seen = []
    monkeypatch.setattr(consolidate, "pick_phase_result", _pick("reduced", seen))

    matrices = tmp_path / "matrices"
    result = _consolidate(
        proj, test_dir,
        run_mutation_on=lambda _directory: merge_km,
        candidate1_km=candidate1_km,
        candidate1_measurement_identity=_measurement_identity(proj),
        measurement_identity=_measurement_identity(proj),
        matrices_dir=str(matrices),
    )

    assert result is not None
    assert [[candidate.label for candidate in call["candidates"]] for call in seen] == [
        ["reduced", "merged", "standalone_0", "standalone_1"]]
    assert result.winner == "reduced"
    assert result.winner_source_label is None
    record = json.loads(
        (matrices / "consolidation_record.json").read_text(encoding="utf-8")
    )
    assert record["fallback_reason"] is None
    assert record["fallback_candidates"] == []


def test_the_selector_always_receives_both_input_suites(
    tmp_path,
    monkeypatch,
):
    """The selector receives source-bound records without remeasurement."""
    from testuary import consolidation as consolidate

    proj, test_dir = _consolidation_project(tmp_path)
    matrix = _km([{"id": "m1", "status": "killed"}])
    phase = consolidate.PhaseConsolidation(
        merged_source="def test_merged():\n    assert True\n",
        reduced_source="def test_merged():\n    assert True\n",
        reduction_valid=True,
        merged_km=matrix,
        reduced_km=matrix,
        candidate2_km=None,
        candidate2_coverage=None,
        suite_covered_lines=[1],
        suite_covered_branches=["1->2"],
        selected_tests=1,
        pool_tests=2,
    )
    monkeypatch.setattr(
        consolidate, "consolidate_phase_end", lambda *args, **kwargs: phase
    )
    seen = []
    monkeypatch.setattr(consolidate, "pick_phase_result", _pick("reduced", seen))
    matrices = tmp_path / "matrices"
    result = _consolidate(
        proj, test_dir,
        run_mutation_on=lambda _directory: matrix,
        candidate1_km=matrix,
        candidate1_coverage={
            "covered_statement_lines": [1],
            "covered_branch_arcs": ["1->2"],
        },
        candidate1_measurement_identity=_measurement_identity(proj),
        measurement_identity=_measurement_identity(proj),
        matrices_dir=str(matrices),
    )

    assert result is not None
    assert [candidate.label for candidate in seen[-1]["candidates"]] == [
        "reduced", "merged", "standalone_0", "standalone_1",
    ]
    assert seen[-1]["required_axes"] == ("mutation", "branch", "line")
    record = json.loads(
        (matrices / "consolidation_record.json").read_text(encoding="utf-8")
    )
    assert record["fallback_reason"] is None
    assert record["selection_audit"]["delivered_from"] == "reduced"


def test_coverage_only_selection_passes_no_mutation_runner(
    tmp_path,
    monkeypatch,
):
    """A deliberate coverage-only run reaches the selector without mutation."""
    from testuary import consolidation as consolidate

    proj, test_dir = _consolidation_project(tmp_path)
    phase = consolidate.PhaseConsolidation(
        merged_source="def test_merged():\n    assert True\n",
        reduced_source="def test_merged():\n    assert True\n",
        reduction_valid=True,
        merged_km=None,
        reduced_km=None,
        selected_tests=1,
        pool_tests=2,
    )
    monkeypatch.setattr(
        consolidate, "consolidate_phase_end", lambda *args, **kwargs: phase
    )
    seen = []
    monkeypatch.setattr(consolidate, "pick_phase_result", _pick("reduced", seen))
    result = _consolidate(
        proj, test_dir,
        run_mutation_on=None,
        candidate1_km=None,
        candidate1_coverage=None,
    )

    assert result is not None
    assert seen[-1]["required_axes"] == ("branch", "line")
    # The suite the phase received has no coverage, so it neither passes
    # nor counts as passing in the census.
    first = next(candidate for candidate in seen[-1]["candidates"]
                 if candidate.label == "standalone_0")
    assert (first.green, first.status) == (False, "incomplete")


def test_the_input_candidates_carry_their_test_counts(tmp_path, monkeypatch):
    """The suite the mutation phase received and the loop's own suite carry
    their test counts into the phase end's census, as the merged and the
    reduced suites do."""
    from testuary import consolidation as consolidate

    proj, test_dir = _consolidation_project(tmp_path)
    phase = consolidate.PhaseConsolidation(
        merged_source="def test_merged():\n    assert True\n",
        reduced_source="def test_merged():\n    assert True\n",
        reduction_valid=True,
        merged_km=None,
        reduced_km=None,
        selected_tests=1,
        pool_tests=2,
    )
    monkeypatch.setattr(
        consolidate, "consolidate_phase_end", lambda *args, **kwargs: phase
    )
    seen = []
    monkeypatch.setattr(consolidate, "pick_phase_result", _pick("reduced", seen))
    _consolidate(
        proj, test_dir,
        run_mutation_on=None,
        candidate1_km=None,
        candidate1_coverage=None,
    )

    counts = {candidate.label: candidate.test_count
              for call in seen for candidate in call["candidates"]}
    assert (counts["standalone_0"], counts["standalone_1"]) == (1, 2)


def test_selected_input_with_different_coverage_is_not_a_fallback(
    tmp_path, monkeypatch,
):
    """Evidence identities are audited separately from score selection."""
    from testuary import consolidation as consolidate

    proj, test_dir = _consolidation_project(tmp_path)
    matrix = _km([{"id": "m1", "status": "killed"}])
    phase = consolidate.PhaseConsolidation(
        merged_source="def test_merged():\n    assert True\n",
        reduced_source="def test_merged():\n    assert True\n",
        reduction_valid=True,
        merged_km=matrix,
        reduced_km=matrix,
        candidate2_km=matrix,
        candidate2_coverage={
            "line": 100.0,
            "branch": 100.0,
            "covered_statement_lines": [1, 2],
            "covered_branch_arcs": ["1->2", "2->3"],
            "per_test_tokens": {},
        },
        suite_covered_lines=[1],
        suite_covered_branches=["1->2"],
        selected_tests=1,
        pool_tests=2,
    )
    monkeypatch.setattr(
        consolidate, "consolidate_phase_end", lambda *args, **kwargs: phase
    )
    seen = []
    monkeypatch.setattr(consolidate, "pick_phase_result", _pick("standalone_1", seen))
    matrices = tmp_path / "matrices"
    result = _consolidate(
        proj, test_dir,
        run_mutation_on=lambda _directory: matrix,
        candidate1_km=matrix,
        candidate1_coverage={
            "line": 100.0,
            "branch": 100.0,
            "covered_statement_lines": [1],
            "covered_branch_arcs": ["1->2"],
            "per_test_tokens": {},
        },
        candidate1_measurement_identity=_measurement_identity(proj),
        measurement_identity=_measurement_identity(proj),
        matrices_dir=str(matrices),
    )

    assert result is not None
    assert result.winner_source_label == "standalone_1"
    assert len(seen[-1]["candidates"]) == 4
    record = json.loads(
        (matrices / "consolidation_record.json").read_text(encoding="utf-8")
    )
    assert record["fallback_reason"] is None
    assert record["fallback_candidates"] == []


def test_no_complete_candidate_persists_failure_record(
    tmp_path, monkeypatch,
):
    """A selector failure leaves both inputs and records an unresolved end."""
    from testuary import consolidation as consolidate

    proj, test_dir = _consolidation_project(tmp_path)
    merged_km = _km([
        {"id": "m1", "status": "killed"},
        {"id": "m2", "status": "survived"},
    ])
    candidate2 = _km([
        {"id": "m1", "status": "killed"},
        {"id": "m2", "status": "killed"},
    ])
    phase = consolidate.PhaseConsolidation(
        merged_source="def test_merged():\n    assert True\n",
        reduced_source="def test_merged():\n    assert True\n",
        reduction_valid=True,
        merged_km=merged_km,
        reduced_km=merged_km,
        candidate2_km=candidate2,
        selected_tests=1,
        pool_tests=2,
    )
    monkeypatch.setattr(
        consolidate, "consolidate_phase_end", lambda *args, **kwargs: phase
    )
    monkeypatch.setattr(
        consolidate,
        "pick_phase_result",
        lambda *args, **kwargs: None,
    )
    matrices = tmp_path / "matrices"

    with pytest.raises(RuntimeError, match="no passing candidate"):
        _consolidate(
            proj, test_dir,
            run_mutation_on=lambda _directory: merged_km,
            candidate1_km=merged_km,
            candidate1_measurement_identity=_measurement_identity(proj),
            measurement_identity=_measurement_identity(proj),
            matrices_dir=str(matrices),
        )

    assert (test_dir / "test_mod.py").exists()
    assert (test_dir / "test_mod_mutation.py").exists()
    record = json.loads(
        (matrices / "consolidation_record.json").read_text(encoding="utf-8")
    )
    assert record["delivered_from"] == "unresolved"
    assert record["fallback_reason"] is None
    assert record["shipped_has_matrix"] is False


def test_finishing_reuses_inherited_standard_matrix(tmp_path):
    """measure_finishing reuses the consolidation's matrix for the
    standard suite instead of running mutation again."""
    std = tmp_path / "test_mod.py"
    std.write_text("def test_a():\n    assert True\n", encoding="utf-8")
    ran_on = []

    def run_mutation_on(d):
        ran_on.append(Path(d).name)
        return {"mutants": [
            {"id": "m1", "status": "killed", "killer": "x::test_s",
             "operator": "op_a"},
        ]}

    out = tm.measure_finishing(
        scratch_root=tmp_path / "scratch",
        merged_test_path=std,
        shared_setup_paths=[],
        run_mutation_on=run_mutation_on,
        standard_coverage=(90.0, 80.0),
        selective_operators=None,
        inherited_standard_km={"mutants": [
            {"id": "m0", "status": "killed", "killer": "x::test_a",
             "operator": "op_a"},
        ]},
        log_fn=None,
    )
    assert ran_on == []                         # reused, not re-run
    assert out.mutation_score_full == 100.0


def test_green_check_retries_spurious_failure_then_succeeds(monkeypatch) -> None:
    """Finishing's green check retries a spurious collection or runner
    failure (pytest rc 5 'no tests collected', rc 4 usage error), so a suite
    with tests keeps its standard-suite mutation score. A failing suite (rc 1)
    is refused without retry."""
    monkeypatch.setattr(tm.time, "sleep", lambda *_a, **_k: None)  # no real backoff

    class _Run:
        def __init__(self, ok, rc):
            self.ok = ok
            self.returncode = rc
            self.elapsed_s = 0.1

    class _Backend:
        def __init__(self, results):
            self.results = list(results)
            self.calls = 0

        def run_test_files(self, files, root, *, timeout,
                           instrument_target=None, tb=None, **kwargs):
            r = self.results[min(self.calls, len(self.results) - 1)]
            self.calls += 1
            return r

    logs: list = []
    # rc 5, then green: retried, returns the green run.
    b = _Backend([_Run(False, 5), _Run(True, 0)])
    out = tm._green_check_with_retry(b, "d", "root", 10, logs.append, test_paths=["d"])
    assert out.ok and b.calls == 2
    # rc 4, then green: retried.
    b4 = _Backend([_Run(False, 4), _Run(True, 0)])
    assert tm._green_check_with_retry(b4, "d", "root", 10, logs.append, test_paths=["d"]).ok
    assert b4.calls == 2
    # rc 1: refused on the first call.
    bred = _Backend([_Run(False, 1), _Run(True, 0)])
    out_red = tm._green_check_with_retry(bred, "d", "root", 10, logs.append, test_paths=["d"])
    assert (not out_red.ok) and bred.calls == 1
    # rc 5 every time: all attempts used, still not ok.
    bp = _Backend([_Run(False, 5)])
    out_p = tm._green_check_with_retry(bp, "d", "root", 10, logs.append, test_paths=["d"])
    assert (not out_p.ok) and bp.calls == tm._GREEN_CHECK_ATTEMPTS

    class _Boom:
        def run_test_files(self, *a, **k):
            raise RuntimeError("boom")

    assert tm._green_check_with_retry(_Boom(), "d", "root", 10, logs.append, test_paths=["d"]) is None


def test_consolidation_persists_full_and_selective_matrices(
    tmp_path, monkeypatch,
):
    """The mutation consolidation writes the full-catalogue matrix and the
    view for the selective operator set, derived by filtering,
    never by a second run."""
    import json as _json
    from testuary import consolidation as consolidate_mod

    monkeypatch.setattr(
        consolidate_mod,
        "benchmark_tied_candidates",
        lambda *args, **kwargs: lambda candidate: (
            1.0 if candidate.label == "merged" else 2.0
        ),
    )
    proj, test_dir = _consolidation_project(tmp_path)
    mdir = tmp_path / "matrices"
    res = _consolidate(
        proj, test_dir,
        run_mutation_on=lambda d: _km([
            {"id": "m1", "status": "killed", "operator": "op_a",
             "killer": "test_mod_mutation.py::test_strong"},
            {"id": "m2", "status": "ok_survived", "operator": "op_b",
             "killer": None},
        ]),
        candidate1_km=_km([
            {"id": "m1", "status": "killed", "operator": "op_a",
             "killer": "test_mod.py::test_weak"},
            {"id": "m2", "status": "ok_survived", "operator": "op_b",
             "killer": None},
        ]),
        candidate1_coverage=_coverage_sidecar(),
        candidate1_measurement_identity=_measurement_identity(proj),
        measurement_identity=_measurement_identity(proj),
        matrices_dir=str(mdir),
        selective_operators=["op_a"],
    )
    assert res is not None
    full = _json.loads(
        (mdir / "merged_kill_matrix.json").read_text("utf-8"))
    assert len(full["mutants"]) == 2
    sel = _json.loads(
        (mdir / "merged_kill_matrix_selective.json").read_text("utf-8"))
    assert sel["operators"] == ["op_a"]
    assert sel["total"] == 1 and sel["killed"] == 1
    assert sel["score"] == 100.0
    record = _json.loads(
        (mdir / "consolidation_record.json").read_text("utf-8"))
    selected = _json.loads(
        (mdir / "selected_kill_matrix.json").read_text("utf-8"))
    assert selected["source_sha256"] == record["selected_source_sha256"]
    # The reduction kept no test here, so it has no matrix, and no other
    # suite's matrix is written under its name.
    assert not (mdir / "reduced_kill_matrix.json").exists()


def test_resume_standard_km_uses_delivered_matrix_file(tmp_path, monkeypatch):
    """A fresh run uses the in-memory delivered suite's matrix
    (``mutation_merge_result.kill_matrix``). A resume (mutation_merge_result
    None) reloads ``archive/mutation/selected_kill_matrix.json``, which holds
    the same matrix, so a resumed target reports the same score as a fresh
    one. The matrices of the merged and the reduced suite describe other
    suites and are never the resume source."""
    from types import SimpleNamespace
    import json as _json
    from testuary import target_phases as orch

    import hashlib as _hashlib

    digest = _hashlib.sha256(b"the delivered suite").hexdigest()
    delivered = {"schema_version": 3,
                 "mutants": [{"id": "m1", "status": "killed"},
                             {"id": "m2", "status": "survived"}],
                 "source_sha256": digest}
    mdir = tmp_path / "ws" / "r0" / "archive" / "mutation"
    mdir.mkdir(parents=True)
    (mdir / "selected_kill_matrix.json").write_text(
        _json.dumps(delivered), encoding="utf-8")
    (mdir / "consolidation_record.json").write_text(
        _json.dumps({"selected_source_sha256": digest}), encoding="utf-8")
    # Other matrices in the merged and reduced suites' files, neither to be resumed.
    (mdir / "merged_kill_matrix.json").write_text(
        _json.dumps({"mutants": [{"id": "x", "status": "killed"}]}),
        encoding="utf-8")
    (mdir / "reduced_kill_matrix.json").write_text(
        _json.dumps({"mutants": [{"id": "y", "status": "killed"}],
                     "source_sha256": _hashlib.sha256(
                         b"the reduction").hexdigest()}),
        encoding="utf-8")
    ctx = SimpleNamespace(results_dir=tmp_path, workspace="ws", log_print=lambda *a, **k: None)

    fresh = SimpleNamespace(
        mutation_merge_result=SimpleNamespace(kill_matrix=delivered))
    assert orch._resume_standard_km(ctx, fresh) == delivered

    resumed = SimpleNamespace(mutation_merge_result=None)
    assert orch._resume_standard_km(ctx, resumed) == delivered

    # Finishing measures a matrix that is derived, or that records only
    # first killers, rather than report it as measured.
    derived = {**delivered, "derived": {"kind": "kill_subset"}}
    assert orch._resume_standard_km(
        ctx, SimpleNamespace(mutation_merge_result=SimpleNamespace(kill_matrix=derived))) is None
    monkeypatch.setattr(orch.measurement, "mutation_config", lambda: {"cover_attribution": "first"})
    assert orch._resume_standard_km(ctx, fresh) is None


class _FakeSession:
    def close(self):
        pass


class _FakeRunner:
    """A mutation runner whose sessions start nothing."""

    def __init__(self, seen, operators):
        self.seen = seen
        self.operators = operators

    def open(self, workers, *, suite_dirs=()):
        self.seen["workers"] = workers
        return _FakeSession()

    def full_operators(self):
        return list(self.operators)


def _fake_backend(seen, operators):
    class _FakeBackend:
        def mutation_runner(self, config, *, source_dir=None, log=None):
            seen["config"] = config
            seen["source_dir"] = source_dir
            seen["log"] = log
            seen["runner"] = _FakeRunner(seen, operators)
            return seen["runner"]

    return _FakeBackend()


def test_union_mutation_runner_opens_session_correctly(tmp_path, monkeypatch):
    """union_mutation_runner opens a session for the project with the
    configured workers. Without a session the kill-aware consolidation would
    fall back to a plain merge."""
    from testuary import languages

    seen = {}
    monkeypatch.setattr(languages, "get_backend",
                        lambda lang: _fake_backend(seen, ["op_a", "op_b"]))
    monkeypatch.setattr(
        tm, "make_finishing_mutation_fn",
        lambda **kw: seen.update(operators=kw["operators"]) or (
            lambda d: {"mutants": []}),
    )
    with tm.union_mutation_runner(
        language="python", target_file="x.py",
        source_dir=str(tmp_path), config={
            "engine_workers": 3,
            "per_mutant_timeout_seconds": "auto",
            "measurement_timeout_seconds": 3600,
            "mutation_max_timeout": 900, "min_per_mutant_timeout_seconds": 30,
            "per_mutant_timeout_factor": 1.5,
        },
    ) as runner:
        assert runner is not None          # session opened
    assert seen["workers"] == 3
    # Its requests record every killing test, asked for by the runner's
    # attribution, not by a key in its configuration.
    assert seen["runner"].attribution == "all" and "_omit_fail_fast" not in seen["config"]
    assert seen["source_dir"] == str(tmp_path)
    assert seen["operators"] == ["op_a", "op_b"]


def test_union_mutation_runner_forwards_configured_per_mutant_timeout(
    tmp_path, monkeypatch,
) -> None:
    """The consolidation mutation run uses the configured per-mutant and
    wall-clock timeouts, not the engine defaults, so a slow mutant resolves as
    in the loop's and Finishing's measurements."""
    from testuary import languages

    seen = {}
    monkeypatch.setattr(languages, "get_backend",
                        lambda lang: _fake_backend(seen, ["op_a"]))
    monkeypatch.setattr(
        tm, "make_finishing_mutation_fn",
        lambda **kw: seen.update(kw) or (lambda d: {"mutants": []}),
    )
    with tm.union_mutation_runner(
        language="python", target_file="x.py", source_dir=str(tmp_path),
        config={
            "engine_workers": 1,
            "per_mutant_timeout_seconds": 180,
            "measurement_timeout_seconds": 5400,
            "mutation_max_timeout": 900, "min_per_mutant_timeout_seconds": 30,
            "per_mutant_timeout_factor": 1.5,
        },
        log_fn=lambda m: None,
    ) as runner:
        assert runner is not None
    assert seen["configured_per_mutant_timeout"] == 180
    assert seen["wall_clock_timeout"] == 5400
    assert seen["language"] == "python"
    # Without the runner's logger the long clone and Cosmic Ray run is silent.
    assert callable(seen["log"])


def test_mutation_measurement_failure_does_not_silently_ship_an_input(
    tmp_path: Path, monkeypatch,
) -> None:
    """A configured mutation runner returning no matrix fails the boundary."""
    from testuary import consolidation as consolidate_mod

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "mod.py").write_text("def f(x):\n    return x + 1\n", encoding="utf-8")
    (proj / "pyproject.toml").write_text("", encoding="utf-8")
    test_dir = tmp_path / "rank1" / "tests"
    cov = "from mod import f\n\ndef test_cov():\n    assert f(1) == 2\n"
    mut = "from mod import f\n\ndef test_mut():\n    assert f(1) == 999\n"
    _write(test_dir / "test_mod.py", cov)
    _write(test_dir / "test_mod_mutation.py", mut)

    # Force the plain merge, which joins cov and the failing mut into a red suite.
    monkeypatch.setattr(
        consolidate_mod, "consolidate_phase_end",
        lambda *a, **k: consolidate_mod.PhaseConsolidation(
            detail="forced: merged suite not green"))

    with pytest.raises(RuntimeError, match="no passing candidate"):
        _consolidate(
            proj, test_dir,
            run_mutation_on=lambda d: None,
        )
    assert (test_dir / "test_mod.py").read_text(encoding="utf-8") == cov
    assert (test_dir / "test_mod_mutation.py").exists()


def test_a_selected_reduced_suite_persists_its_measured_matrix(
    tmp_path: Path, monkeypatch,
) -> None:
    """A selected reduced suite carries its independently measured matrix."""
    proj, test_dir = _consolidation_project(tmp_path)
    selected_matrix = _km([
        {"id": "m1", "status": "killed",
         "killer": "test_mod.py::test_strong"},
    ])
    from testuary import consolidation as consolidate_mod
    monkeypatch.setattr(
        consolidate_mod,
        "pick_phase_result",
        lambda candidates, **kwargs: next(
            candidate for candidate in candidates
            if candidate.label == "reduced"
        ),
    )

    def run_mutation_on(d):
        return selected_matrix

    mat = tmp_path / "matrices"
    res = _consolidate(
        proj, test_dir,
        run_mutation_on=run_mutation_on,
        candidate1_km=_km([
            {"id": "m1", "status": "killed",
             "killer": "test_mod.py::test_weak"},
        ]),
        candidate1_coverage=_coverage_sidecar(),
        candidate1_measurement_identity=_measurement_identity(proj),
        measurement_identity=_measurement_identity(proj),
        matrices_dir=str(mat),
    )
    assert res is not None
    assert res.kill_matrix is selected_matrix
    assert (mat / "selected_kill_matrix.json").exists()  # persisted for resume


def test_green_check_signature_mismatch_raises_instead_of_skipping() -> None:
    """A backend without the expected signature raises instead of counting as
    a spurious run that leaves no mutation score."""
    import pytest

    class _Wrong:
        def run_test_files(self, files, root, *, timeout, tb=None):
            raise TypeError("unexpected keyword argument")

    with pytest.raises(TypeError):
        tm._green_check_with_retry(_Wrong(), "d", "root", 10, lambda m: None, test_paths=["d"])


def test_the_mutation_phase_ending_at_its_inherited_suite_persists_a_reduction(
        tmp_path: Path, monkeypatch) -> None:
    """The mutation phase delivers the inherited suite unchanged, so the
    merge reuses its stored matrix, which the coverage archive binds to the
    inherited bytes. The reduction drops the weak test, and the phase end is
    written with the derived matrix bound to the reduced suite."""
    from testuary import config as config_module
    monkeypatch.setattr(config_module, "consolidation_mutation_method", lambda: "both")
    proj, test_dir = _consolidation_project(tmp_path)
    inherited = (test_dir / "test_mod_mutation.py").read_text(encoding="utf-8")
    _write(test_dir / "test_mod.py", inherited)
    digest = hashlib.sha256(inherited.encode("utf-8")).hexdigest()

    def run_mutation_on(d):
        pyfile = next(Path(d).glob("test_*.py"))
        killer = f"{pyfile.name}::test_strong"
        return _km([{"id": "m1", "status": "killed", "killer": killer,
                     "killers": [killer]}])

    stored = {**_km([{"id": "m1", "status": "killed",
                      "killer": "test_mod.py::test_strong",
                      "killers": ["test_mod.py::test_strong"]}]),
              "source_sha256": digest, "execution_name": "test_mod.py"}
    matrices = tmp_path / "matrices"
    res = tm.consolidate_test_suite(
        test_output_dir=str(test_dir),
        coverage_file_name="test_mod.py",
        language="python",
        target_file_path=str(proj / "mod.py"),
        source_dir=str(proj),
        run_mutation_on=run_mutation_on,
        candidate1_km=stored,
        candidate1_coverage=_coverage_sidecar(),
        candidate1_source=inherited,
        candidate1_measurement_identity=_measurement_identity(proj),
        candidate2_source=inherited,
        measurement_identity=_measurement_identity(proj),
        reduction=config.solver_settings(config.get_agent_config("test_coverage")),
        timeout=120,
        matrices_dir=str(matrices),
    )
    assert res is not None
    reduced = json.loads((matrices / "reduced_kill_matrix.json").read_text(encoding="utf-8"))
    derived = json.loads((matrices / "derived_reduced_kill_matrix.json").read_text(encoding="utf-8"))
    assert reduced["source_sha256"] != digest
    assert derived["source_sha256"] == reduced["source_sha256"]



def test_the_mutation_phase_end_log_names_the_selected_candidate(
    tmp_path: Path, monkeypatch,
) -> None:
    """The phase end delivers one candidate, which need not be a merge of the
    two inputs, and its log names that candidate."""
    from testuary import config as config_module
    from testuary import consolidation as consolidate_mod

    monkeypatch.setattr(config_module, "consolidation_mutation_method", lambda: "derive")
    monkeypatch.setattr(
        consolidate_mod,
        "pick_phase_result",
        lambda candidates, **kwargs: next(
            candidate for candidate in candidates
            if candidate.label == "standalone_1"
        ),
    )
    proj, test_dir = _consolidation_project(tmp_path)

    def run_mutation_on(directory):
        return _km([{"id": "m1", "status": "survived", "killer": None}])

    logs = []
    _consolidate(
        proj, test_dir,
        run_mutation_on=run_mutation_on,
        candidate1_km=_km([{"id": "m1", "status": "survived", "killer": None}]),
        candidate1_coverage=_coverage_sidecar(),
        candidate1_measurement_identity=_measurement_identity(proj),
        measurement_identity=_measurement_identity(proj),
        log_fn=logs.append,
    )
    assert any("the selected candidate standalone_1 is now test_mod.py" in line
               for line in logs)
    assert not any("merged two candidates" in line for line in logs)


@pytest.mark.parametrize("required, consequence", [
    (True, "The setting phase_end_mutation needs kill evidence, so this step fails"),
    (False, "This step goes on without kill evidence"),
])
def test_an_engine_that_does_not_start_logs_what_the_step_does(
    tmp_path: Path, monkeypatch, required, consequence,
) -> None:
    """Without an engine the merged-suite measurement yields no runner. With
    phase_end_mutation on the caller fails. Otherwise it continues without kill
    evidence, and the log says which."""
    from types import SimpleNamespace

    from testuary import languages
    from testuary import config as config_module

    def mutation_runner(*args, **kwargs):
        raise RuntimeError("the engine is not installed")

    monkeypatch.setattr(languages, "get_backend",
                        lambda lang: SimpleNamespace(mutation_runner=mutation_runner))
    monkeypatch.setattr(config_module, "consolidation_measure_mutation", lambda: required)
    logs = []
    with tm.union_mutation_runner(
        language="python", target_file="x.py", source_dir=str(tmp_path),
        config={}, log_fn=logs.append,
    ) as runner:
        assert runner is None
    assert any(line.endswith(consequence) for line in logs), logs
    assert not any("kills are not preserved" in line for line in logs)


def test_the_green_check_retry_log_names_no_test_runner(monkeypatch) -> None:
    """Finishing's green check runs every language's suite, so its
    retry log shows the output's tail without naming pytest."""
    from types import SimpleNamespace

    monkeypatch.setattr(tm.time, "sleep", lambda *_a, **_k: None)

    class _Backend:
        def run_test_files(self, files, root, *, timeout,
                           instrument_target=None, tb=None, **kwargs):
            return SimpleNamespace(ok=False, returncode=5, elapsed_s=0.1,
                                   output="no tests ran")

    logs = []
    tm._green_check_with_retry(_Backend(), "d", "root", 10, logs.append, test_paths=["d"])
    retries = [line for line in logs if "spurious" in line]
    assert retries and all("pytest" not in line for line in retries)


def test_the_inherited_suites_removal_stands_before_the_phase_ends_own(
    tmp_path: Path, monkeypatch,
) -> None:
    """The inherited suite measured again on a stale identity loses its
    named test before the phase end runs, so its entry stands before the
    entry of the phase's own suite, which loses the same test."""
    from phase_end_calls import named_id
    from testuary import config as config_module

    monkeypatch.setattr(config_module, "consolidation_mutation_method", lambda: "derive")
    proj, test_dir = _consolidation_project(tmp_path)
    for name in ("test_mod.py", "test_mod_mutation.py"):
        _write(test_dir / name, (test_dir / name).read_text(encoding="utf-8")
               + "\n\ndef test_extra():\n    assert f(2) == 3\n")
    named = {}

    def run_mutation_on(directory):
        pyfile = next(Path(directory).glob("test_*.py"))
        held = "def test_extra(" in pyfile.read_text(encoding="utf-8")
        named[str(directory)] = (named_id(pyfile, "test_extra"),) if held else ()
        return None if held else _km([{"id": "m1", "status": "survived", "killer": None}])

    run_mutation_on.named_of = lambda directory: named.get(str(directory), ())
    archive = tmp_path / "archive"
    tm.consolidate_test_suite(
        test_output_dir=str(test_dir), coverage_file_name="test_mod.py",
        **_suites(test_dir), language="python", target_file_path=str(proj / "mod.py"),
        source_dir=str(proj), run_mutation_on=run_mutation_on,
        candidate1_km=_km([{"id": "m1", "status": "killed", "killer": "old"}]),
        candidate1_coverage=_coverage_sidecar(),
        candidate1_measurement_identity={**_measurement_identity(proj), "config_sha256": "stale"},
        measurement_identity={**_measurement_identity(proj), "config_sha256": "current"},
        matrices_dir=str(archive), timeout=120,
    )
    record = json.loads((archive / "consolidation_record.json").read_text(encoding="utf-8"))
    assert [entry["place"] for entry in record["measurement_removals"]] == [
        "phase_end.inherited", "phase_end.candidate"]


def test_a_stale_inherited_suite_loses_the_test_its_mutation_check_names(
    tmp_path: Path, monkeypatch,
) -> None:
    """The inherited suite measured again on a stale identity loses the test
    its mutation check names: the merge takes it without that test,
    the record holds the removal, the archive keeps the changed suite, and
    candidate_inherited keeps the suite as the coverage phase delivered it."""
    from phase_end_calls import named_id
    from testuary import config as config_module
    from testuary import consolidation as consolidate_mod

    monkeypatch.setattr(config_module, "consolidation_mutation_method", lambda: "derive")
    monkeypatch.setattr(
        consolidate_mod, "pick_phase_result",
        lambda candidates, **kwargs: next(
            candidate for candidate in candidates if candidate.label == "merged"))
    proj, test_dir = _consolidation_project(tmp_path)
    inherited = ("from mod import f\n\n"
                 "def test_weak():\n    assert f(1) is not None\n\n"
                 "def test_extra():\n    assert f(2) == 3\n")
    _write(test_dir / "test_mod.py", inherited)
    named = {}

    def run_mutation_on(directory):
        pyfile = next(Path(directory).glob("test_*.py"))
        held = "def test_extra(" in pyfile.read_text(encoding="utf-8")
        named[str(directory)] = (named_id(pyfile, "test_extra"),) if held else ()
        return None if held else _km([{"id": "m1", "status": "survived", "killer": None}])

    run_mutation_on.named_of = lambda directory: named.get(str(directory), ())
    archive = tmp_path / "archive"
    tm.consolidate_test_suite(
        test_output_dir=str(test_dir), coverage_file_name="test_mod.py",
        **_suites(test_dir), language="python", target_file_path=str(proj / "mod.py"),
        source_dir=str(proj), run_mutation_on=run_mutation_on,
        candidate1_km=_km([{"id": "m1", "status": "killed", "killer": "old"}]),
        candidate1_coverage=_coverage_sidecar(),
        candidate1_measurement_identity={**_measurement_identity(proj), "config_sha256": "stale"},
        measurement_identity={**_measurement_identity(proj), "config_sha256": "current"},
        matrices_dir=str(archive), timeout=120,
    )
    record = json.loads((archive / "consolidation_record.json").read_text(encoding="utf-8"))
    [entry] = [entry for entry in record["measurement_removals"]
               if entry["place"] == "phase_end.inherited"]
    assert entry["measurement"] == "mutation"
    assert entry["removed"][0]["test"].endswith(".test_extra")
    assert "def test_extra" not in (test_dir / "test_mod.py").read_text(encoding="utf-8")
    pre = test_dir.parent / "_pre_merge" / "candidate_inherited.py"
    assert pre.read_text(encoding="utf-8") == inherited
    changed = (archive / "candidate_standalone_0_removed.py").read_text(encoding="utf-8")
    assert "def test_extra" not in changed and "def test_weak" in changed
