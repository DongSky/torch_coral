"""torch.compile integration for static inference graphs (no CPU fallback)."""
import copy
from pathlib import Path
import uuid
import itertools
import torch
from .compiler import compile

def _parameter_source(source):
    while source is not None:
        if "ParamBufferSource" in type(source).__name__: return True
        source = getattr(source, "base", None)
    return False

def make_backend(calibration_data, *, artifact_root="artifacts/dynamo", **compile_options):
    """Use as torch.compile(..., backend=make_backend(...), fullgraph=True,
    dynamic=False), then call under torch.no_grad(). Calibration tuples follow
    the graph's public tensor argument order. Parameters and buffers are frozen.
    Returned backend retains modules in .modules; close() releases their workers.
    """
    calibration = list(itertools.islice(iter(calibration_data), compile_options.get("max_calibration_samples", 256)))
    modules = []
    def backend(gm, example_inputs):
        if torch.is_grad_enabled():
            raise RuntimeError("Compile/call Coral graphs inside torch.no_grad() or torch.inference_mode()")
        frozen = copy.deepcopy(gm)
        placeholders = [n for n in frozen.graph.nodes if n.op == "placeholder"]
        public = []
        constants = {}
        for index, (node, value) in enumerate(zip(placeholders, example_inputs)):
            if not isinstance(value, torch.Tensor):
                raise TypeError("Coral backend requires static tensor arguments; use dynamic=False")
            grapharg = node.meta.get("grapharg")
            if isinstance(value, torch.nn.Parameter) or _parameter_source(getattr(grapharg, "source", None)):
                name = "_coral_frozen_%d" % index
                frozen.register_buffer(name, value.detach().cpu().clone())
                with frozen.graph.inserting_after(node): replacement = frozen.graph.get_attr(name)
                node.replace_all_uses_with(replacement)
                frozen.graph.erase_node(node)
                constants[index] = (value.data_ptr(), value._version)
            else:
                public.append(index)
        frozen.graph.lint()
        frozen.recompile()
        frozen.eval()
        artifact = Path(artifact_root) / uuid.uuid4().hex
        module = compile(frozen, tuple(example_inputs[i] for i in public), calibration,
                         artifact_dir=artifact, **compile_options)
        modules.append(module)
        def execute(*args):
            for index, expected in constants.items():
                value = args[index]
                if (value.data_ptr(), value._version) != expected:
                    raise RuntimeError("Frozen model weights/buffers changed. Create a new Coral backend and recompile.")
            return module(*(args[i] for i in public))
        return execute
    backend.modules = modules
    def close():
        for module in modules: module.close()
    backend.close = close
    return backend
