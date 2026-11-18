"""Resources copied into a working directory by the CLI.

``testuary init`` copies models.ini into the user directory and hints.json.
``testuary init --examples`` copies the examples directory to ./testuary-examples.
These resources are accessed through importlib.resources. Configuration and
hint loaders use the working copies, not these bundled templates.
The runtime default config.ini is a separate resource at the package root."""
