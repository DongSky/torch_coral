"""Independent, uncalibrated-input validation of the saved integer strategy."""
import json
from pathlib import Path
import sys
import numpy as np
from convert import validate_quantized


def main():
    folder = Path(sys.argv[1]).resolve()
    job = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    data = [np.load(folder / f'validation_{i}.npy', allow_pickle=False) for i in range(len(job['inputs']))]
    result = validate_quantized(folder, job, folder / 'converted/model_coral_strategy.tflite', data,
                                report_name='heldout-validation.json')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
