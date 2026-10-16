param(
  [ValidateSet('demo','detect','reconstruct','evaluate','preview','test','open-scene')]
  [string]$Stage = 'demo',
  [string]$Session = 'outputs/my_new_session',
  [int]$Frames = 12
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.runtime.json')) {
  throw 'Run: python scripts/bootstrap.py first.'
}
$runtime = Get-Content -LiteralPath '.runtime.json' -Raw | ConvertFrom-Json
$python = $runtime.python
$env:PYTHONUTF8 = '1'
$env:PYTHONNOUSERSITE = '1'
function Run-Python {
  & $python @args
  if ($LASTEXITCODE -ne 0) { throw "Python failed with exit code $LASTEXITCODE" }
}
if ($Stage -eq 'demo') {
  Run-Python scripts/run_blender.py --output $Session --frames $Frames
  Run-Python -m pose_sim.pipeline detect $Session
  Run-Python -m pose_sim.pipeline reconstruct $Session
  Run-Python -m pose_sim.evaluate $Session
  Run-Python -m pose_sim.preview $Session
} elseif ($Stage -eq 'test') {
  Run-Python -m unittest discover -s tests -v
} elseif ($Stage -eq 'open-scene') {
  $scenePath = (Resolve-Path -LiteralPath (Join-Path $Session 'scene.blend')).Path
  Start-Process -FilePath $runtime.blender -ArgumentList ('"' + $scenePath + '"')
} elseif ($Stage -in @('detect','reconstruct')) {
  Run-Python -m pose_sim.pipeline $Stage $Session
} else {
  Run-Python -m "pose_sim.$Stage" $Session
}
