"""One upstream inference subprocess, with a fixed timeout and no retry."""
import argparse
import configparser
import importlib.metadata as md
import os
from pathlib import Path
import subprocess
import sys
import time

from .io import read_json,write_json,check_file


def run(task,model,source,rgb,output):
    output.mkdir(parents=True,exist_ok=False)
    parser=configparser.ConfigParser();parser.read_dict(task['inference']['settings'])
    parser['general']['model_path_root']=str(model)
    parser['data']['patchify_folder']=str(output/'scratch')
    config=output/'height10m_worker.cfg'
    with config.open('x') as stream:parser.write(stream)
    write_json(output/'input.json',dict(image_list=[str(rgb)],save_dir=str(output/'raw')))
    request=output/'request.json'
    write_json(request,dict(source=str(source),model=str(model),inference=task['inference'],config=str(config),
                           input=str(output/'input.json'),results=str(output/'result-paths.json')))
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        GDAL_PAM_ENABLED='NO',PROJ_NETWORK='OFF',MPLBACKEND='Agg',PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    for key,name in [('TMPDIR','temporary'),('TORCH_HOME','torch'),('CUDA_CACHE_PATH','cuda'),('MPLCONFIGDIR','matplotlib')]:
        path=output/'scratch'/name;path.mkdir(parents=True);env[key]=str(path)
    command=[sys.executable,'-u','-B','-m','spc_agb_worker.inference',str(request)]
    with (output/'execution.log').open('x') as log:
        result=subprocess.run(command,cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT,
                              timeout=task['inference']['timeout_seconds'],check=False)
    if result.returncode:raise RuntimeError('Inference failed; see '+str(output/'execution.log'))
    paths=read_json(output/'result-paths.json')
    if len(paths)!=1:raise ValueError('Expected one fresh height')
    raw=Path(paths[0]).resolve()
    if not raw.is_relative_to((output/'raw').resolve()):raise ValueError('Height escaped fresh output')
    return raw,read_json(output/'metrics.json')


def child(request):
    plan=read_json(request);source=Path(plan['source']);model=Path(plan['model']);infer=plan['inference']
    for name,digest in infer['source_sha256'].items():check_file(source/name,digest)
    for name,key in [(infer['model']['weights'],'weights_sha256'),(infer['model']['training'],'training_sha256')]:
        check_file(model/name,infer['model'][key])
    sys.path.insert(0,str(source))
    import torch
    import inference as upstream
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU unavailable')
    torch.cuda.reset_peak_memory_stats();began=time.monotonic()
    sys.argv=['inference.py','--config',plan['config'],'--input',plan['input'],'--write_resultpath',plan['results']]
    try:
        upstream.main()
    finally:
        write_json(Path(request).parent/'metrics.json',dict(seconds=time.monotonic()-began,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            torch=torch.__version__,cuda=torch.version.cuda,torchvision=md.version('torchvision'),lightning=md.version('lightning'),
            gpu=torch.cuda.get_device_name(0),matmul_precision=torch.get_float32_matmul_precision()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('request',type=Path)
    child(parser.parse_args().request)
