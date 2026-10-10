"""Host-only import/meta/adapter-factory audit. No forward or optimizer step."""
import argparse
import json
from pathlib import Path
import traceback

ROOT = Path(__file__).parent

def audit():
    import torch
    import torch_neuronx
    import neuronx_distributed
    from transformers import GptOssConfig,GptOssForCausalLM
    cfg = GptOssConfig(**json.loads((ROOT/'gpt_oss_metadata/config.json').read_text()))
    with torch.device('meta'): model = GptOssForCausalLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    expert = sum(p.numel() for n,p in model.named_parameters() if '.experts.' in n)
    result = dict(imports_ok=True,meta_parameters=total,expert_parameters=expert,
        bf16_base_bytes=total*2,tp2_base_bytes_per_rank=(total-expert)*2+expert,
        audit_only=True,optimizer_steps=0,weights_loaded=False)
    from neuronx_distributed.modules.lora.model import LoraModel
    from neuronx_distributed.modules.lora.config import LoraConfig
    obj = LoraModel.__new__(LoraModel)
    torch.nn.Module.__init__(obj)
    obj.lora_config = LoraConfig(lora_rank=32,target_modules=['experts'])
    try: obj._create_new_module(model.model.layers[0].mlp.experts,'experts')
    except Exception:
        result.update(expert_adapter_supported=False,expert_adapter_error=traceback.format_exc())
    else: result['expert_adapter_supported'] = True
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    try: result = audit()
    except Exception: result = dict(imports_ok=False,failure=traceback.format_exc(),audit_only=True,optimizer_steps=0)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
