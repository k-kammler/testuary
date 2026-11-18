"""The project's Python interpreter as a run chooses it, as the tests read it."""
from testuary import config, project
from testuary.languages import language_setting_values


def project_python() -> str:
    """The interpreter of the project's Python that project_runtime chooses
    for the active run, from the Python settings it reads."""
    project_config = config.get_project_config()
    values = language_setting_values("python", project_config, config.active_settings())
    root = str(project_config.get("source_dir") or "")
    return project.project_runtime("python", values, root=root).interpreter.path
