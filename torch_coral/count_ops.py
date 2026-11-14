"""Count nominal INT8 MAC work in the uncompiled, optimized TFLite graph.

Run with converter Python. One MAC = two operations. Padding work is counted in
the usual dense-convolution convention; these are logical ops, not HW counters.
"""
import json
import math
import sys
from ai_edge_litert import schema_py_generated as s

def count_model(path):
    m=s.ModelT.InitFromObj(s.Model.GetRootAsModel(open(path,'rb').read(),0))
    names={v:k for k,v in vars(s.BuiltinOperator).items() if isinstance(v,int)}
    layers=[]; excluded={}
    for g in m.subgraphs:
        for op in g.operators:
            code=m.operatorCodes[op.opcodeIndex].builtinCode
            name=names.get(code,str(code)); macs=0
            output=g.tensors[int(op.outputs[0])]
            if code in (s.BuiltinOperator.CONV_2D,s.BuiltinOperator.DEPTHWISE_CONV_2D):
                weight=g.tensors[int(op.inputs[1])]
                if weight.type!=s.TensorType.INT8: raise ValueError('MAC benchmark requires INT8 weights')
                if code==s.BuiltinOperator.CONV_2D:
                    macs=math.prod(output.shape)*math.prod(weight.shape[1:])
                else:
                    macs=math.prod(output.shape)*int(weight.shape[1])*int(weight.shape[2])
            elif code==s.BuiltinOperator.FULLY_CONNECTED:
                weight=g.tensors[int(op.inputs[1])]
                if weight.type!=s.TensorType.INT8: raise ValueError('MAC benchmark requires INT8 weights')
                macs=math.prod(output.shape)*int(weight.shape[1])
            elif code==s.BuiltinOperator.BATCH_MATMUL:
                left=g.tensors[int(op.inputs[0])]
                right=g.tensors[int(op.inputs[1])]
                if left.type!=s.TensorType.INT8 or right.type!=s.TensorType.INT8:
                    raise ValueError('MAC benchmark requires INT8 matrix operands')
                inner=int(left.shape[-2] if op.builtinOptions.adjX else left.shape[-1])
                macs=math.prod(output.shape)*inner
            else: excluded[name]=excluded.get(name,0)+1
            if macs: layers.append({'op':name,'output_shape':output.shape.tolist(),'macs':int(macs)})
    total=sum(v['macs'] for v in layers)
    if not total: raise ValueError('No countable INT8 Conv/Depthwise/FC/BatchMatMul MACs; comparison-only graphs have no MAC-based TOPS')
    return {'macs':total,'ops_2_per_mac':total*2,'layers':layers,'excluded_non_mac_ops':excluded,
            'convention':'Dense logical INT8 Conv/Depthwise/FC/BatchMatMul MACs in the compiled graph, including rewrite work; 1 MAC = 2 ops; not physical instruction counters'}

if __name__=='__main__': print(json.dumps(count_model(sys.argv[1])))
