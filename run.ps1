param(
  [ValidateSet('robot-demo','robot-test','g1-control','vision-demo','stream-vision','refine-motion','estimate-state','demo','check-source','detect','reconstruct','fit-links','transfer','evaluate','preview','control','test','open-scene')]
  [string]$Stage = 'robot-demo',
  [string]$Session = 'outputs/my_new_session',
  [string]$InputSession = 'outputs/human_walk_g1',
  [string]$Detections = 'outputs/vision_optimized/detections.jsonl',
  [ValidateSet(1,2,4)]
  [int]$DetectorWorkers = 1,
  [ValidateSet(1,2,4)]
  [int]$PoseWorkers = 1,
  [int]$Frames = 12,
  [ValidateSet('auto','cpu','cuda')]
  [string]$Device = 'auto',
  [ValidateSet('lightweight','balanced','performance')]
  [string]$Mode = 'balanced',
  [string]$Robot = 'config/robot_demo.json',
  [ValidateSet('auto','coco17','robot-markerless')]
  [string]$Source = 'auto',
  [string]$SourceConfig = 'config/source_robot.json',
  [string]$DetectorModel = '',
  [string]$DetectorSha256 = '',
  [string]$Mapping = 'config/transfer_source_to_g1.json',
  [double]$PerceptionDelayMs = 120,
  [switch]$Profile
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$robotStage = $Stage -in @('robot-demo','robot-test','g1-control')
$runtimeFile = if ($robotStage) { '.robot-runtime.json' } else { '.runtime.json' }
if (-not (Test-Path -LiteralPath $runtimeFile)) {
  if ($robotStage) { throw 'Run: python scripts/bootstrap_robot.py first.' }
  throw 'Run: python scripts/bootstrap.py first.'
}
$runtime = Get-Content -LiteralPath $runtimeFile -Raw | ConvertFrom-Json
$python = $runtime.python
if ($Device -eq 'auto') { $Device = $runtime.device }
$renderDevice = if ($Device -eq 'cpu') { 'CPU' } else { 'OPTIX' }
if ($Source -eq 'auto') { $Source = 'coco17' }
$detectArgs = @('--device', $Device, '--mode', $Mode)
if ($Profile) { $detectArgs += '--profile' }
$env:PYTHONUTF8 = '1'
$env:PYTHONNOUSERSITE = '1'
function Run-Python {
  & $python @args
  if ($LASTEXITCODE -ne 0) { throw "Python failed with exit code $LASTEXITCODE" }
}
if ($Stage -eq 'robot-demo') {
  if ($Device -ne 'cuda') { throw 'The verified robot baseline requires CUDA image inference.' }
  if (-not $PSBoundParameters.ContainsKey('Frames')) { $Frames = 96 }
  Run-Python scripts/robot_demo.py --output $Session --frames $Frames --perception-delay-ms $PerceptionDelayMs
} elseif ($Stage -eq 'robot-test') {
  Run-Python -m unittest discover -s tests -v
} elseif ($Stage -eq 'vision-demo') {
  Run-Python scripts/vision_demo.py $InputSession --output $Session --prepare-models
} elseif ($Stage -eq 'stream-vision') {
  Run-Python scripts/stream_vision.py $InputSession --output $Session --detector-workers $DetectorWorkers --pose-workers $PoseWorkers
} elseif ($Stage -eq 'refine-motion') {
  Run-Python scripts/refine_motion.py $InputSession --detections $Detections --output $Session
} elseif ($Stage -eq 'estimate-state') {
  Run-Python -m pose_sim.human_state $Session --config config/vision_state.json
} elseif ($Stage -eq 'g1-control') {
  Run-Python scripts/robot_demo.py --output $Session --control-only --perception-delay-ms $PerceptionDelayMs
} elseif ($Stage -eq 'check-source') {
  Run-Python -m pose_sim.robot_source check-source --source-config $SourceConfig
} elseif ($Stage -eq 'demo') {
  if ($Source -ne 'coco17') { throw 'demo is the human COCO17 regression only; markerless robots require a source profile and detector.' }
  Run-Python scripts/run_blender.py --output $Session --frames $Frames --render-device $renderDevice
  Run-Python -m pose_sim.pipeline detect $Session @detectArgs
  Run-Python -m pose_sim.pipeline reconstruct $Session
  Run-Python -m pose_sim.evaluate $Session
  Run-Python -m pose_sim.preview $Session
  Run-Python -m motion_control.replay $Session --robot $Robot --perception-delay-ms $PerceptionDelayMs
  Run-Python -m motion_control.preview (Join-Path $Session 'control')
  Run-Python scripts/verify_session.py $Session --require-control
} elseif ($Stage -eq 'test') {
  Run-Python -m unittest discover -s tests -v
} elseif ($Stage -eq 'open-scene') {
  $scenePath = (Resolve-Path -LiteralPath (Join-Path $Session 'scene.blend')).Path
  Start-Process -FilePath $runtime.blender -ArgumentList ('"' + $scenePath + '"')
} elseif ($Stage -eq 'detect') {
  if ($Source -eq 'robot-markerless') {
    Run-Python -m pose_sim.robot_source check-source --source-config $SourceConfig
    if (-not $DetectorModel -or -not $DetectorSha256) { throw 'Supply a source-specific markerless ONNX detector and its SHA256; see docs/source_robot_to_g1.md.' }
    Run-Python -m pose_sim.markerless $Session --source-config $SourceConfig --model $DetectorModel --model-sha256 $DetectorSha256 --device $Device
  } else {
    Run-Python -m pose_sim.pipeline detect $Session @detectArgs
  }
} elseif ($Stage -eq 'reconstruct') {
  Run-Python -m pose_sim.pipeline reconstruct $Session
} elseif ($Stage -eq 'fit-links') {
  Run-Python -m pose_sim.robot_source fit-links --session $Session --source-config $SourceConfig
} elseif ($Stage -eq 'transfer') {
  Run-Python -m motion_control.transfer $Session --mapping $Mapping
} elseif ($Stage -eq 'preview' -and $Source -eq 'robot-markerless') {
  Run-Python -m pose_sim.robot_preview $Session
} elseif ($Stage -eq 'control') {
  Run-Python -m motion_control.replay $Session --robot $Robot --perception-delay-ms $PerceptionDelayMs
  Run-Python -m motion_control.preview (Join-Path $Session 'control')
} else {
  Run-Python -m "pose_sim.$Stage" $Session
}
