"""Bounded idempotent provider recovery and verified loopback range delivery.

Retries exist only here. GDAL and workers do not retry. Failed bodies remain
immutable evidence; only complete identity-verified responses enter the cache.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from datetime import datetime,timezone
from copy import deepcopy
import hashlib,json,re,threading,time,os
import requests
resolve_path = Path  # Explicit paths only; no workstation storage profile.
from .state import atomic,writer_lock,committed

class TransferStop(RuntimeError):pass
class SourceAccessFailure(TransferStop):pass
class TransferBudgetExceeded(TransferStop):pass
class IntegrityFailure(ValueError):pass
class SharedStorageFailure(RuntimeError):pass
class TransientRead(RuntimeError):pass

TRANSPORT_VERSION='bounded-provider-recovery-v3'
PROVIDER_TOTAL_SECONDS=60
PROVIDER_READ_SECONDS=10
MAX_ATTEMPTS=3
BACKOFF_SECONDS=(1,2)
ARCHIVE_SECONDS=30
JOURNAL_SECONDS=15
# Total includes a final blocking read, separate archive/verification and journal.
REQUEST_ENVELOPE_SECONDS=MAX_ATTEMPTS*(PROVIDER_TOTAL_SECONDS+PROVIDER_READ_SECONDS+ARCHIVE_SECONDS+3*JOURNAL_SECONDS)+sum(BACKOFF_SECONDS)
GATEWAY_READER_TIMEOUT_SECONDS=REQUEST_ENVELOPE_SECONDS+62 # 500s: bounded delivery/reader margin
PROVIDER_WORKER_TIMEOUT_SECONDS=2*GATEWAY_READER_TIMEOUT_SECONDS+100 # 1100s total stage
TRANSPORT_FINALISATION_MARGIN_SECONDS=PROVIDER_READ_SECONDS+ARCHIVE_SECONDS+2*JOURNAL_SECONDS+30+60
TRANSIENT_HTTP={408,429,500,502,503,504}

def reader_options():
    return dict(GDAL_HTTP_TIMEOUT=GATEWAY_READER_TIMEOUT_SECONDS,GDAL_HTTP_CONNECTTIMEOUT=10,GDAL_HTTP_MAX_RETRY=0,GDAL_HTTP_MULTIRANGE='NO')

def failure_kind(error):
    if isinstance(error,SourceAccessFailure):return 'source_access'
    if isinstance(error,TransferBudgetExceeded):return 'budget'
    if isinstance(error,(SharedStorageFailure,OSError)):return 'shared_storage'
    return 'integrity_or_environment'


def verified_reuse_record(row,source_identity,*,journal_cache=None):
    """Verify a cross-stage response against its immutable originating journal."""
    origin=row.get('origin_journal')
    if not isinstance(origin,dict) or set(origin)!={'path','sha256'}:raise IntegrityFailure('Missing originating journal provenance')
    cache_key=(origin['path'],origin['sha256'])
    cached=journal_cache.get(cache_key) if journal_cache is not None else None
    if cached is None:
        raw=resolve_path(origin['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=origin['sha256']:raise IntegrityFailure('Originating journal identity changed')
        state=json.loads(raw);responses={};reservations={}
        for record in state['responses']:responses.setdefault(record.get('key'),[]).append(record)
        for record in state.get('reservations',[]):reservations.setdefault(record.get('key'),[]).append(record)
        cached=(responses,reservations)
        if journal_cache is not None:journal_cache[cache_key]=cached
    responses,reserved=cached;fields=['key','url','method','range','status','headers','bytes','sha256']
    if any(k not in row for k in fields) or row.get('source_identity')!=source_identity:raise IntegrityFailure('Incomplete response/source identity')
    expected=hashlib.sha256((row['url']+'|'+row['method']+'|'+str(row['range'])).encode()).hexdigest()
    if row['key']!=expected:raise IntegrityFailure('Response key/source/range identity differs')
    matches=[q for q in responses.get(row['key'],[]) if all(q.get(k)==row[k] for k in fields)]
    if len(matches)!=1 or Path(matches[0]['body']).name!=Path(row['body']).name:raise IntegrityFailure('Response absent or inconsistent in originating journal')
    reservations=[q for q in reserved.get(row['key'],[]) if q.get('status')=='received' and q.get('source_identity')==source_identity and q.get('body_sha256')==row['sha256'] and q.get('actual_bytes')==row['bytes']]
    if len(reservations)!=1:raise IntegrityFailure('Originating received source reservation missing or inconsistent')
    body=resolve_path(row['body']).read_bytes()
    if len(body)!=row['bytes'] or hashlib.sha256(body).hexdigest()!=row['sha256']:raise IntegrityFailure('Run range reuse bytes changed')
    return body


def response_reuse_index(incoming,journal,source_identity):
    """Construct the existing reuse contract only from verified received records."""
    journal=Path(str(journal));raw=resolve_path(journal).read_bytes();state=json.loads(raw)
    origin=dict(path=str(journal),sha256=hashlib.sha256(raw).hexdigest());result={};journals={}
    for row in state['responses']:
        body=Path(row['body']);body=body if body.is_absolute() else Path(str(incoming))/body
        entry=dict(row,body=str(body),source_identity=source_identity,origin_journal=row.get('origin_journal') or origin)
        verified_reuse_record(entry,source_identity,journal_cache=journals)
        if entry['key'] in result and result[entry['key']]!=entry:raise IntegrityFailure('Conflicting response reuse identity')
        result[entry['key']]=entry
    # Origin journals are immutable snapshots. Parse/index each once for this
    # bounded batch, hash every body, then recheck every journal before returning.
    # This preserves all provenance checks without quadratic JSON rereads.
    for path,digest in set(journals)|{(origin['path'],origin['sha256'])}:
        if hashlib.sha256(resolve_path(path).read_bytes()).hexdigest()!=digest:raise IntegrityFailure('Originating journal changed during batch verification')
    return result

class RangeArchive:
    def __init__(self,incoming,journal,limit,session=None,*,deadline=None,source_identity=None,max_attempts=MAX_ATTEMPTS,reuse_index=None):
        self.incoming,self.journal=Path(str(incoming)),Path(str(journal));self.incoming.mkdir(parents=True,exist_ok=True)
        self.state=json.loads(self.journal.read_text()) if self.journal.exists() else dict(reserved=0,actual=0,responses=[],failed=None)
        self.last_committed=deepcopy(self.state);self.limit=limit;self.deadline=deadline
        if not 1<=max_attempts<=MAX_ATTEMPTS:raise ValueError('At most three provider attempts')
        self.max_attempts=max_attempts;self.source_identity=source_identity
        self.reuse_index=reuse_index if reuse_index is not None else {}
        self.session=session or requests.Session();self.session.trust_env=False;self.session.headers['Accept-Encoding']='identity'
        # No adapter retries underneath the one upstream recovery loop.
        if isinstance(self.session,requests.Session):
            for adapter in self.session.adapters.values():
                if adapter.max_retries.total not in (0,False):raise ValueError('Nested provider retries forbidden')
        self.error=None;self.error_kind=None;self.deliveries=[];self.metrics=[]
        self.validators=deepcopy(self.state.get('object_validators',{}))
        for row in self.state['responses']:self._identity(row['url'],row['method'],row['headers'])

    def publish(self):
        began=time.monotonic()
        try:result=atomic(self.journal,self.state)
        except OSError as e:raise SharedStorageFailure('Transfer journal publication failed: '+str(e)) from e
        self.last_committed=deepcopy(self.state)
        if time.monotonic()-began>JOURNAL_SECONDS:raise SharedStorageFailure('Shared journal exceeded publication bound')
        return result

    def _identity(self,url,method,headers):
        h=requests.structures.CaseInsensitiveDict(headers);record={k:h[k] for k in ['ETag','Last-Modified','x-amz-version-id'] if k in h}
        if method=='HEAD':record['length']=int(h['Content-Length'])
        elif h.get('Content-Range'):
            m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',h['Content-Range'])
            if m:record['length']=int(m[3])
        prior=self.validators.setdefault(url,{})
        for k,v in prior.items():
            if k not in record or record[k]!=v:raise IntegrityFailure('Provider object identity/validator changed or missing: '+k)
        prior.update(record);self.state['object_validators']=deepcopy(self.validators)

    def fetch(self,url,method,byte_range=None):
        if self.error:raise SourceAccessFailure('Earlier gateway failure; no new provider request')
        try:
            with writer_lock(self.journal.with_name(self.journal.name+'.request-lock')):
                now=committed(self.journal) if self.journal.exists() else dict(reserved=0,actual=0,responses=[],failed=None)
                if now!=self.last_committed:raise IntegrityFailure('Journal changed by another writer; stop before request')
                try:return self._fetch(url,method,byte_range)
                except Exception as e:
                    if self.error is None:self.error=type(e).__name__+': '+str(e);self.error_kind=failure_kind(e)
                    self.state.update(failed=self.error,failure_kind=self.error_kind)
                    self.publish();raise # still owns the writer lock
        except Exception as e:
            if self.error is None:self.error=type(e).__name__+': '+str(e);self.error_kind=failure_kind(e)
            # A stale/denied writer must never publish its obsolete state over
            # another writer's committed journal, even to record its failure.
            raise

    def _fetch(self,url,method,byte_range):
        if self.state['failed']:raise SourceAccessFailure('Earlier required request failed; no stage retry')
        if not url.startswith('https://'):raise IntegrityFailure('Explicit anonymous HTTPS required')
        size=0
        if method=='GET':
            match=re.fullmatch(r'bytes=(\d+)-(\d+)',byte_range or '')
            if not match:raise IntegrityFailure('Explicit bounded single byte range required')
            start,end=map(int,match.groups());size=end-start+1
            if not 0<size<=16*2**20:raise IntegrityFailure('Range outside 16MiB request bound')
        elif method!='HEAD':raise IntegrityFailure('Only HEAD and ranged GET')
        key=hashlib.sha256((url+'|'+method+'|'+str(byte_range)).encode()).hexdigest()
        for row in self.state['responses']:
            if row['key']!=key:continue
            began=time.monotonic()
            try:raw=resolve_path(self.incoming/row['body']).read_bytes()
            except OSError as e:raise SharedStorageFailure('Saved range read failed: '+str(e)) from e
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise IntegrityFailure('Archived response changed')
            if time.monotonic()-began>ARCHIVE_SECONDS:raise SharedStorageFailure('Saved range read/hash exceeded archive bound')
            self.metrics.append(dict(cache_hit=True,bytes_reused=len(raw),provider_attempts=0,cache_read_hash_seconds=time.monotonic()-began))
            return row['status'],row['headers'],raw
        saved=self.reuse_index.get(key)
        if saved and saved.get('source_identity')==self.source_identity:
            if (saved['url'],saved['method'],saved['range'])!=(url,method,byte_range):raise IntegrityFailure('Run range reuse key differs')
            tick=time.monotonic()
            try:raw=verified_reuse_record(saved,self.source_identity)
            except OSError as e:raise SharedStorageFailure('Run range reuse read failed: '+str(e)) from e
            if len(raw)!=saved['bytes'] or hashlib.sha256(raw).hexdigest()!=saved['sha256']:raise IntegrityFailure('Run range reuse bytes changed')
            self._identity(url,method,saved['headers'])
            if time.monotonic()-tick>ARCHIVE_SECONDS:raise SharedStorageFailure('Run reuse read/hash exceeded archive bound')
            row=dict(saved,reused_from=saved['origin_journal']);self.state['responses'].append(row);self.publish()
            self.metrics.append(dict(cache_hit=True,cache_scope='verified_run_range',bytes_reused=len(raw),provider_attempts=0,cache_read_hash_seconds=time.monotonic()-tick))
            return row['status'],row['headers'],raw
        if any(r['key']==key for r in self.state.get('reservations',[])):raise IntegrityFailure('Interrupted required request is spent; no stage retry')
        for attempt in range(1,self.max_attempts+1):
            if self.deadline is not None and time.monotonic()>=self.deadline:raise SourceAccessFailure('Stage transport deadline reached before request')
            if self.state['reserved']+size>self.limit:raise TransferBudgetExceeded('Transfer budget reached before provider attempt')
            reservation=dict(key=key,attempt=attempt,url=url,method=method,range=byte_range,bytes_reserved=size,status='reserved_before_request',source_identity=self.source_identity,started_utc=datetime.now(timezone.utc).isoformat())
            self.state['reserved']+=size;self.state.setdefault('reservations',[]).append(reservation);self.publish()
            if self.deadline is not None and time.monotonic()>=self.deadline:raise SourceAccessFailure('Stage transport deadline reached before provider dispatch')
            response_headers={};body=bytearray();transient=None;began=time.monotonic();network_done=None;status=None
            deadline=min(began+PROVIDER_TOTAL_SECONDS,self.deadline) if self.deadline is not None else began+PROVIDER_TOTAL_SECONDS
            try:
                headers={'Range':byte_range} if byte_range else {}
                known=self.validators.get(url,{})
                if known.get('ETag') and not known['ETag'].startswith('W/'):headers['If-Match']=known['ETag']
                elif known.get('Last-Modified'):headers['If-Unmodified-Since']=known['Last-Modified']
                with self.session.request(method,url,headers=headers,stream=True,timeout=(10,PROVIDER_READ_SECONDS),allow_redirects=False) as response:
                    reservation['connection_response_seconds']=time.monotonic()-began
                    status=response.status_code;response_headers=dict(response.headers);h=requests.structures.CaseInsensitiveDict(response_headers)
                    if status in TRANSIENT_HTTP or status in [403,404]:
                        # Error payloads are evidence/transfer too, never TIFF cache.
                        # Reserve before reading at most 64KiB; close larger errors.
                        if method=='GET':
                            if h.get('Content-Encoding','identity')!='identity':raise IntegrityFailure('Encoded provider error response')
                            if self.state['reserved']+65536>self.limit:raise TransferBudgetExceeded('Error-body evidence reservation exceeds transfer budget')
                            self.state['reserved']+=65536;reservation['error_body_reservation']=65536;self.publish()
                            for block in response.iter_content(4096):
                                body.extend(block);self.state['actual']+=len(block)
                                if len(body)>=65536:break
                                if time.monotonic()>deadline:
                                    if status in [403,404]:raise SourceAccessFailure('Source access HTTP '+str(status)+'; bounded error-body read stopped')
                                    raise TransientRead('Provider error-body deadline exceeded')
                            reservation['error_evidence_capped']=len(body)>=65536
                        if status in TRANSIENT_HTTP:raise TransientRead('Transient provider HTTP '+str(status))
                        raise SourceAccessFailure('Source access HTTP '+str(status))
                    if status!=(206 if method=='GET' else 200):raise IntegrityFailure('Access/range contract HTTP '+str(status))
                    if h.get('Content-Encoding','identity')!='identity':raise IntegrityFailure('Encoded range response')
                    length=int(h['Content-Length'])
                    if method=='GET':
                        extent=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',h.get('Content-Range',''))
                        if not extent:raise IntegrityFailure('Missing range extent')
                        lo,hi,total=map(int,extent.groups())
                        if lo!=start or hi!=min(end,total-1) or length!=hi-lo+1 or length>size:raise IntegrityFailure('Provider returned different range')
                    self._identity(url,method,response_headers)
                    if time.monotonic()>deadline:raise TransientRead('Provider total response deadline exceeded')
                    if method=='GET':
                        for block in response.iter_content(65536):
                            body.extend(block);self.state['actual']+=len(block)
                            if len(body)>size:raise IntegrityFailure('Provider exceeded reserved response')
                            if time.monotonic()>deadline:raise TransientRead('Provider total response deadline exceeded')
                    if method=='GET' and len(body)!=length:raise TransientRead('Truncated provider body')
                network_done=time.monotonic()
            except (requests.exceptions.ConnectionError,requests.exceptions.Timeout,requests.exceptions.ChunkedEncodingError,TransientRead) as error:
                network_done=time.monotonic()
                if status in [403,404]:
                    self._archive_attempt(key,reservation,response_headers,body,False,began,network_done,error)
                    raise SourceAccessFailure('Source access HTTP '+str(status)+'; error-body read failed') from error
                transient=error
            except BaseException as error:
                network_done=time.monotonic();self._archive_attempt(key,reservation,response_headers,body,False,began,network_done,error);raise
            complete=transient is None
            raw=self._archive_attempt(key,reservation,response_headers,body,complete,began,network_done,transient)
            if complete:
                row=dict(key=key,url=url,method=method,range=byte_range,status=status,headers=response_headers,body=reservation['body'],bytes=len(raw),sha256=reservation['body_sha256'])
                self.state['responses'].append(row);self.publish();return status,response_headers,raw
            self.publish()
            if attempt<self.max_attempts:
                delay=BACKOFF_SECONDS[attempt-1]
                if self.deadline is not None and time.monotonic()+delay>=self.deadline:break
                tick=time.monotonic();time.sleep(delay);reservation['backoff_seconds']=time.monotonic()-tick
        raise SourceAccessFailure('Provider attempts exhausted: '+str(transient))

    def _archive_attempt(self,key,row,headers,body,complete,began,network_done,error):
        tick=time.monotonic();name=key+f'.attempt-{row["attempt"]:02d}'+('.body' if complete else '.partial.body');path=self.incoming/name
        try:
            with path.open('xb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
            (self.incoming/(name+'.headers.json')).write_text(json.dumps(headers,indent=2)+'\n')
            written=time.monotonic()
            raw=path.read_bytes()
        except OSError as e:raise SharedStorageFailure('Archive write/verification failed: '+str(e)) from e
        digest=hashlib.sha256(raw).hexdigest()
        if raw!=body:raise IntegrityFailure('Archive readback differs')
        elapsed=time.monotonic()-tick
        row.update(status='received' if complete else 'failed_attempt',actual_bytes=len(raw),body=name,body_sha256=digest,headers=headers,
            provider_seconds=network_done-began,download_seconds=max(0,network_done-began-row.get('connection_response_seconds',0)),archive_write_verify_seconds=elapsed,error=str(error) if error else None)
        row.update(archive_write_seconds=written-tick,archive_verify_seconds=time.monotonic()-written)
        self.metrics.append(dict(cache_hit=False,bytes_transferred=len(raw),provider_attempts=1,provider_seconds=row['provider_seconds'],archive_write_verify_seconds=elapsed,archive_write_seconds=row['archive_write_seconds'],archive_verify_seconds=row['archive_verify_seconds']))
        if elapsed>ARCHIVE_SECONDS:raise SharedStorageFailure('Shared archive exceeded write/verification bound')
        return raw

@contextmanager
def gateway(archive,urls):
    registered={hashlib.sha256(url.encode()).hexdigest():url for url in urls}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_HEAD(self):self.handle_source('HEAD')
        def do_GET(self):self.handle_source('GET')
        def handle_source(self,method):
            began=time.monotonic();started=False;delivery=dict(method=method,path=self.path,range=self.headers.get('Range'),started_utc=datetime.now(timezone.utc).isoformat(),confirmed_body_bytes=0);archive.deliveries.append(delivery)
            self.connection.settimeout(10)
            try:
                key=self.path.strip('/').split('?')[0]
                if key not in registered:raise IntegrityFailure('Unregistered source/sidecar access')
                status,headers,raw=archive.fetch(registered[key],method,self.headers.get('Range'));headers=requests.structures.CaseInsensitiveDict(headers);length=int(headers['Content-Length'])
                if method=='GET':
                    extent=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',headers.get('Content-Range',''));requested=re.fullmatch(r'bytes=(\d+)-(\d+)',self.headers.get('Range',''))
                    if status!=206 or not extent or not requested:raise IntegrityFailure('Invalid archived range response')
                    lo,hi,total=map(int,extent.groups());start,end=map(int,requested.groups())
                    if lo!=start or hi!=min(end,total-1) or length!=hi-lo+1 or len(raw)!=length:raise IntegrityFailure('Incomplete required response')
                elif status!=200 or raw:raise IntegrityFailure('Invalid archived HEAD response')
                ready=time.monotonic();delivery.update(ready_seconds=ready-began,announced_body_bytes=len(raw),content_range=headers.get('Content-Range'))
                self.connection.settimeout(10);self.send_response(status)
                for name in ['Content-Length','Content-Range','Content-Type','Accept-Ranges','Last-Modified','ETag']:
                    if name in headers:self.send_header(name,headers[name])
                self.send_header('Connection','close');started=True;self.end_headers()
                if method=='GET':
                    for offset in range(0,len(raw),65536):
                        block=raw[offset:offset+65536];self.wfile.write(block);delivery['confirmed_body_bytes']+=len(block)
                        if time.monotonic()-ready>20:raise TimeoutError('Bounded required gateway delivery exceeded')
                delivery.update(status='sent_complete',delivery_seconds=time.monotonic()-ready)
            except Exception as error:
                if archive.error is None:archive.error=type(error).__name__+': '+str(error);archive.error_kind='delivery' if started else failure_kind(error)
                delivery.update(status='disconnected_unverified' if started else 'failed_before_body',error=str(error))
                if not started:
                    try:self.send_error(503)
                    except (ConnectionError,OSError) as e:delivery['error_response_failure']=repr(e)
            finally:delivery['seconds']=time.monotonic()-began;self.close_connection=True
    from contextvars import copy_context
    context=copy_context()
    server=HTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=lambda:context.run(server.serve_forever),daemon=True);thread.start()
    try:yield {url:f'http://127.0.0.1:{server.server_port}/{key}' for key,url in registered.items()}
    finally:
        server.shutdown();thread.join();server.server_close()
        atomic(archive.journal.with_name(archive.journal.stem+'-delivery.json'),dict(version=TRANSPORT_VERSION,reader_options=reader_options(),request_envelope_seconds=REQUEST_ENVELOPE_SECONDS,worker_seconds=PROVIDER_WORKER_TIMEOUT_SECONDS,deliveries=archive.deliveries,metrics=archive.metrics,error=archive.error,failure_kind=archive.error_kind,limitation='Socket delivery is not decoded success; worker completion remains required. Metrics overlap raster-call wall time.'))
