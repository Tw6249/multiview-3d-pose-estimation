"""Extract the GPU backbone before NMS; run using the pinned robot Python.

The public detector applies a 0.3 score threshold after ONNX NMS. Applying that
threshold before equivalent CPU NMS avoids suppressing thousands of low scores
on the GPU. The original model and weights are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import onnx
from onnx import numpy_helper


def prepare(source):
    source=Path(source).resolve()
    model=onnx.load(str(source))
    nodes=[n for n in model.graph.node if n.op_type=='NonMaxSuppression']
    if len(nodes)!=1 or any(a.name=='center_point_box' and a.i!=0 for a in nodes[0].attribute):
        raise ValueError('Require a single xyxy-compatible NMS node')
    node=nodes[0]
    constants={i.name:numpy_helper.to_array(i) for i in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type=='Constant':
            for a in n.attribute:
                if a.name=='value':constants[n.output[0]]=numpy_helper.to_array(a.t)
    output=source.parent/'cpu_nms'/source.name
    output.parent.mkdir(exist_ok=True)
    onnx.utils.extract_model(str(source),str(output),[v.name for v in model.graph.input],list(node.input[:2]))
    manifest=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  sha256=hashlib.sha256(output.read_bytes()).hexdigest(),onnx_version=onnx.__version__,
                  iou_threshold=float(constants[node.input[3]].item()),
                  max_output=int(constants[node.input[2]].item()),score_threshold=.3,
                  original_score_threshold=float(constants[node.input[4]].item()))
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(file=str(output),**manifest)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('models',nargs='+',type=Path)
    for model in p.parse_args().models:prepare(model)
