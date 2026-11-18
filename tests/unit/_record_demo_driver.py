"""Record the Toy as the shipped data were recorded, in a process of its own,
its model answers from an earlier recording. No live model source may be
built and no connection may leave the machine.

usage: python _record_demo_driver.py WORK {shipped|operations-only}
WORK receives the project copy, out/ and data/. The exit status is the
recorder's."""
import configparser
import importlib.util
import json
import sys
from importlib import resources
from pathlib import Path

from testuary.demo.scenario import Scenario

spec = importlib.util.spec_from_file_location(
    "record_demo", Path(__file__).resolve().parents[2] / "tools" / "record_demo.py")
record = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record)


def refused(*args, **kwargs):
    raise AssertionError("a live model source was built")


def audit(event, args):
    if event == "socket.connect":
        address = args[1]
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise ConnectionRefusedError(f"the replay contacted {host}")


record.LiveModelSource = refused
sys.addaudithook(audit)

work, mode = Path(sys.argv[1]), sys.argv[2]
toy = resources.files("testuary.demo") / "data" / "toy"
shipped = Scenario.load(Path(str(toy / "scenario.json.gz")))
replay = shipped if mode == "shipped" else Scenario(
    [entry for entry in shipped.entries if not entry.kind.startswith("model.")])
manifest = json.loads((toy / "manifest.json").read_text(encoding="utf-8"))
project = work / manifest["exemplar"]["instance"]
project.mkdir(parents=True)
for name, text in json.loads((toy / "project.json").read_text(encoding="utf-8"))["files"].items():
    (project / name).write_text(text, encoding="utf-8", newline="\n")
models = work / "models.ini"
models.write_bytes((toy / "models.ini").read_bytes())
alias = manifest["invocation"]["model"]
catalogue = configparser.ConfigParser(interpolation=None)
catalogue.read_string(models.read_text(encoding="utf-8"))
sys.exit(record.record(
    language=manifest["language"], project=project, target=manifest["exemplar"]["target"],
    instance=manifest["exemplar"]["instance"], alias=alias, models=models,
    catalogue_dir=None, alias_fields=dict(catalogue[alias]),
    preset_text=(toy / "preset.ini").read_text(encoding="utf-8"),
    options=manifest["invocation"]["options"], outdir=work / "out", data_dir=work / "data",
    python=None, java_home=None, bundle=(), licenses=(),
    notice=(toy / "NOTICE.md").read_text(encoding="utf-8"),
    run_label=manifest["exemplar"]["run"], adapter=manifest["adapter"],
    file_extensions=manifest["file_extensions"], replay=replay))
