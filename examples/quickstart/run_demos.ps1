# Play Testuary's offline demonstrations with its console command, one after
# the other. Each run of the script gets its own new directory below demos,
# named by its start time. They need no model, no key and no setup.
$directory = "demos/" + (Get-Date -Format "yyyyMMdd-HHmmss")
$runs = @(
    @("--demo", "--demo-speed", "instant", "--output-dir", "$directory/python"),
    @("--demo", "--demo-language", "java", "--demo-speed", "instant", "--output-dir", "$directory/java"),
    @("--demo", "--demo-language", "toy", "--demo-speed", "instant", "--output-dir", "$directory/toy"),
    @("--demo", "--demo-data", "tutorial", "--demo-speed", "instant", "--output-dir", "$directory/tutorial")
)
foreach ($run in $runs) {
    testuary @run
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Output "The records are in $directory."
