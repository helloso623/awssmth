"""Download the pinned pretrained checkpoint separately; never load a model here."""
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parent
checkpoint = snapshot_download(
    repo_id='openai/gpt-oss-20b',
    revision='6cee5e81ee83917806bbde320786a8fb61efebee',
    cache_dir=str(root / '.hf-gptoss/hub'),
    allow_patterns=['*.safetensors', '*.json', '*.jinja'],
)
(root / 'pretrained-path.txt').write_text(checkpoint + '\n')
print(checkpoint)
