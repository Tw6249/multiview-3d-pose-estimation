param(
  [ValidateSet('robot-test','test','vision-demo','stream-vision','refine-motion','estimate-state','detect','reconstruct','g1-replay','continuous-v3','mamma-teleop')]
  [string]$Stage = 'robot-test',
  [string]$Session = 'outputs/my_new_session',
  [string]$InputSession = 'outputs/public_mamma_session',
  [string]$Detections = 'outputs/public_mamma_session/detections.jsonl',
  [ValidateSet(1,2,4)]
  [int]$DetectorWorkers = 1,
  [ValidateSet(1,2,4)]
  [int]$PoseWorkers = 1,
  [ValidateSet('auto','cpu','cuda')]
  [string]$Device = 'auto',
  [ValidateSet('lightweight','balanced','performance')]
  [string]$Mode = 'balanced',
  [double]$PerceptionDelayMs = 120,
  [switch]$Profile
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$robotStage = $Stage -in @('robot-test','g1-replay','continuous-v3','mamma-teleop')
$runtimeFile = if ($robotStage) { '.robot-runtime.json' } else { '.runtime.json' }
if (-not (Test-Path -LiteralPath $runtimeFile)) {
  if ($robotStage) { throw 'Run: python scripts/bootstrap_robot.py first.' }
  throw 'Run: python scripts/bootstrap.py first.'
}
$runtime = Get-Content -LiteralPath $runtimeFile -Raw | ConvertFrom-Json
$python = $runtime.python
if ($Device -eq 'auto') { $Device = $runtime.device }
$detectArgs = @('--device', $Device, '--mode', $Mode)
if ($Profile) { $detectArgs += '--profile' }
$env:PYTHONUTF8 = '1'
$env:PYTHONNOUSERSITE = '1'
function Run-Python {
  & $python @args
  if ($LASTEXITCODE -ne 0) { throw "Python failed with exit code $LASTEXITCODE" }
}
if ($Stage -eq 'robot-test' -or $Stage -eq 'test') {
  Run-Python -B -m unittest discover -s tests -v
} elseif ($Stage -eq 'vision-demo') {
  Run-Python scripts/vision_demo.py $InputSession --output $Session --prepare-models
} elseif ($Stage -eq 'stream-vision') {
  Run-Python scripts/stream_vision.py $InputSession --output $Session --detector-workers $DetectorWorkers --pose-workers $PoseWorkers
} elseif ($Stage -eq 'refine-motion') {
  Run-Python scripts/refine_motion.py $InputSession --detections $Detections --output $Session
} elseif ($Stage -eq 'estimate-state') {
  Run-Python -m pose_sim.human_state $Session --config config/vision_state.json
} elseif ($Stage -eq 'detect') {
  Run-Python -m pose_sim.pipeline detect $Session @detectArgs
} elseif ($Stage -eq 'reconstruct') {
  Run-Python -m pose_sim.pipeline reconstruct $Session
} elseif ($Stage -eq 'g1-replay') {
  Run-Python -m motion_control.humanoid_replay $Session --delay-ms $PerceptionDelayMs
} elseif ($Stage -eq 'continuous-v3') {
  Run-Python scripts/benchmark_continuous_v3.py --output $Session
} elseif ($Stage -eq 'mamma-teleop') {
  Run-Python scripts/run_mamma_teleop_showcase.py
}
