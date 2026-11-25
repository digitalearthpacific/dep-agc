"""A DEP Processor invokes the versioned scientific worker using an explicit interpreter."""

import os
import signal
import subprocess

from dep_tools.processors import Processor

from carbon.io import read_json


class CarbonProcessor(Processor):
    def __init__(self, python, code, model, source, output, timeout=900):
        super().__init__()
        if timeout <= 0:
            raise ValueError('Worker timeout must be positive')
        self.python, self.code, self.model, self.source = python, code, model, source
        self.output, self.timeout = output, timeout

    def process(self, package):
        command = [str(self.python), '-u', '-B', '-m', 'carbon',
                   '--task', str(package/'task.json'), '--package', str(package),
                   '--model', str(self.model), '--source', str(self.source), '--output', str(self.output)]
        env = dict(os.environ, PYTHONPATH=str(self.code), PYTHONNOUSERSITE='1')
        with self.output.with_suffix('.log').open('x') as log:
            proc = subprocess.Popen(command, cwd=self.code, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                status = proc.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise RuntimeError(f'{self.timeout}s ceiling reached; partial results retained; no retry')
        if status:
            raise RuntimeError('Worker failed; partial results/log retained; no retry')
        return read_json(self.output/'completion.json')
