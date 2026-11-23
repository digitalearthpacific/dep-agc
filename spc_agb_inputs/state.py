"""Same-directory, single-writer JSON publication for Windows/SMB.

The committed filename is authoritative. Uncommitted unique temporary files are
retained on failure and never selected as recovery state. Retrying replace does
not call the producer, HTTP client or any worker.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


@contextmanager
def writer_lock(path):
    path=Path(str(path));path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open('a+b')
    locked=False
    try:
        stream.seek(0,2)
        if not stream.tell():stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        locked=True
        yield
    except OSError as error:
        if not locked:raise RuntimeError('Competing journal/checkpoint writer; stop') from error
        raise
    finally:
        if locked:
            stream.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(stream,fcntl.LOCK_UN)
        stream.close()


def committed(path):
    """Read only the committed name, never a potentially interrupted temporary."""
    return json.loads(Path(str(path)).read_text(encoding='utf8'))


def atomic(path,value,*,delays=(.025,.05,.1,.2,.4,.8,.8),replace=None,sleep=None):
    path=Path(str(path));path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf8')
    replace=replace or os.replace;sleep=sleep or time.sleep
    temporary=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    errors=[]
    with writer_lock(path.with_name(path.name+'.publish-lock')):
        with temporary.open('xb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        for attempt in range(len(delays)+1):
            try:
                replace(temporary,path)
                # Verify a complete committed state before allowing a request.
                if path.read_bytes()!=raw:raise RuntimeError('Committed JSON readback differs')
                return dict(replace_attempts=attempt+1,retries=len(errors),publication_errors=errors,
                            sha256=hashlib.sha256(raw).hexdigest())
            except OSError as error:
                errors.append(dict(type=type(error).__name__,winerror=getattr(error,'winerror',None),errno=error.errno))
                retryable=getattr(error,'winerror',None) in (5,32,33) or error.errno in (13,1)
                if not retryable or attempt==len(delays):raise
                # An SMB acknowledgement may be lost after a successful rename.
                # Accept only exact complete bytes, never an absent/partial state.
                if not temporary.exists():
                    if path.is_file() and path.read_bytes()==raw:
                        return dict(replace_attempts=attempt+1,retries=len(errors),publication_errors=errors,
                                    sha256=hashlib.sha256(raw).hexdigest(),verified_after_uncertain_ack=True)
                    raise
                sleep(delays[attempt])
