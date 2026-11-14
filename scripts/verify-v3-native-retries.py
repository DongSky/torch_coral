"""Force mixed-row saturation and check independent native readout retries."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from torch_coral.gguf_weights import WeightStore
from torch_coral.isa_batched import BatchedMatrixTPU
from torch_coral.isa_streamed import StreamedMatrixTPU


if __name__=='__main__':
    store=WeightStore('.tools/models/qwen35-gguf/fp32-prepared')
    name='blk.0.ffn_up.weight'; w=store[name]
    x=np.random.default_rng(413).normal(size=(13,w.shape[1])).astype(np.float32)
    x[0]=np.sign(w[0]); x[1]=-x[0]
    with BatchedMatrixTPU('.tools/models/qwen35-gguf/int8-matrices',components=1,output_digits=2,clock='max') as b:
        before=b.matrix_stats['range_retries']
        old=StreamedMatrixTPU._project(b,name,w,x)
        old_retries=b.matrix_stats['range_retries']-before
        b._resident=None; b._batch_resident=None
        before=b.matrix_stats['range_retries']; new=b._project(name,w,x)
        retries=b.matrix_stats['range_retries']-before
        np.testing.assert_array_equal(new.view(np.uint32),old.view(np.uint32))
        assert retries==old_retries and retries>0
    result=dict(status='bitwise PASS',baseline_retries=old_retries,native_retries=retries,mixed_row_ranges=True)
    Path('reports/v3-native-retries.json').write_text(json.dumps(result,indent=2)); print(result)
