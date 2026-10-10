"""Read-only mapping evidence; print only Neuron environment variables."""
import json
from pathlib import Path

report = {'devices': [str(p) for p in Path('/dev').glob('neuron[0-9]*')]}
for pid in (1, 62, 1639307, 1639308):
    entry = {}
    for name in ('status', 'environ'):
        try:
            data = Path(f'/proc/{pid}/{name}').read_bytes()
            if name == 'environ':
                entry[name] = [s.decode(errors='replace') for s in data.split(b'\0')
                               if s.startswith(b'NEURON')]
            else:
                entry[name] = [s for s in data.decode().splitlines()
                               if s.startswith(('Name:', 'NSpid:'))]
        except OSError as exc:
            entry[name] = str(exc)
    report[str(pid)] = entry
root = Path('/sys/class/neuron_device/neuron0')
report['physical_core_count'] = (root/'core_count').read_text().strip()
report['physical_core_memory'] = {
    str(i): (root/f'neuron_core{i}/stats/memory_usage/device_mem/present').read_text().strip()
    for i in range(8)}
print(json.dumps(report, indent=2))
