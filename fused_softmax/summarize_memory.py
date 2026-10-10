"""Summarize sampled per-core allocations; never call samples exact peaks."""
import argparse,json
from pathlib import Path

def summarize(path):
    peaks={};samples=0
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        data=json.loads(line);totals={}
        for runtime in data.get('neuron_runtime_data',[]):
            report=runtime.get('report',runtime)
            memory=report.get('memory_used',{}).get('neuron_runtime_used_bytes',{})
            cores=memory.get('usage_breakdown',{}).get('neuroncore_memory_usage',{})
            for core,categories in cores.items():
                totals[core]=totals.get(core,0)+sum(categories.values())
        if totals:samples+=1
        for core,value in totals.items():peaks[core]=max(peaks.get(core,0),value)
    return dict(source=str(path),samples=samples,sampled_peak_bytes_per_core=peaks,
        definition='Maximum sampled sum of runtime allocation categories per logical core;1s sampling may miss shorter transients',
        runtime_peak_counter_available=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output');a=p.parse_args()
    result=summarize(a.input)
    if a.output:Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
