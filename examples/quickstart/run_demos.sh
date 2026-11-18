#!/bin/sh
# Play Testuary's offline demonstrations with its console command, one after
# the other. Each run of the script gets its own new directory below demos,
# named by its start time. They need no model, no key and no setup.
set -e
run="demos/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run"
testuary --demo --demo-speed instant --output-dir "$run/python"
testuary --demo --demo-language java --demo-speed instant --output-dir "$run/java"
testuary --demo --demo-language toy --demo-speed instant --output-dir "$run/toy"
testuary --demo --demo-data tutorial --demo-speed instant --output-dir "$run/tutorial"
echo "The records are in $run."
