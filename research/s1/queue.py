"""Shared file queue + one packer per GPU.

Layout (on the shared filesystem):
  queue/pending/<name>.json    tasks waiting; <name> = P<prio>_<run_id>, so a directory sort is priority order
  queue/running/<name>.json    claimed (atomic rename); <name>.hb is its heartbeat, <name>.owner its job
  queue/done/<name>.json       finished
  queue/failed/<name>.json     failed twice (log tail attached)
  queue/vram.json              measured peak VRAM per vram_key (packer learns while running)

Task fields: run_id, kind ('train'|'llm'|'cmd'), argv (list, run from S1_ROOT), priority, vram_gb, slots,
est_hours, exclusive (bool), after (list of run_ids that must be done), vram_key, retries.

    python -m s1.queue pack --vram-gb 176 --max-slots 6 --cpus 12
    python -m s1.queue status
    python -m s1.queue requeue-stale
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .common import ROOT, atomic_json, read_json

Q = Path(os.environ.get('S1_QUEUE', ROOT / 'queue'))
DIRS = {k: Q / k for k in ('pending', 'running', 'done', 'failed')}
INCOMPLETE = 3
MARGIN = 1.15


def ensure():
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def task_name(t):
    return f"P{int(t['priority'])}_{t['run_id'].replace('/', '__')}"


def enqueue(tasks, replace=False):
    ensure()
    existing = {p.stem.split('_', 1)[1]: d for d in DIRS for p in DIRS[d].glob('*.json')}
    n = 0
    for t in tasks:
        t = dict(dict(kind='train', priority=5, vram_gb=20, slots=1, est_hours=1.0, exclusive=False,
                      after=[], retries=0), **t)
        t.setdefault('vram_key', t['run_id'].split('/')[0])
        key = t['run_id'].replace('/', '__')
        if key in existing and not replace:
            continue
        atomic_json(DIRS['pending'] / f'{task_name(t)}.json', t)
        n += 1
    return n


def done_ids():
    return {read_json(p)['run_id'] for p in DIRS['done'].glob('*.json')}


def _job_end():
    """Epoch seconds when this Slurm allocation ends (S1_JOB_END set by worker.slurm), else +inf."""
    v = os.environ.get('S1_JOB_END')
    return float(v) if v else float('inf')


def fits_walltime(t):
    return t['est_hours'] * 1.3 + 0.25 < (_job_end() - time.time()) / 3600


class Packer:
    def __init__(self, vram_gb, max_slots, cpus, poll=5.0):
        ensure()
        self.cap_vram, self.cap_slots, self.cpus, self.poll = vram_gb, max_slots, cpus, poll
        self.running = {}            # name -> dict(proc, task, log, t0)
        self.stop = False
        self.job = os.environ.get('SLURM_JOB_ID', f'local{os.getpid()}')
        signal.signal(signal.SIGTERM, self._on_term)

    def _on_term(self, *_):
        self.stop = True

    def learned_vram(self, t):
        path = Q / 'vram.json'
        learned = read_json(path).get(t['vram_key']) if path.exists() else None
        return max(t['vram_gb'], learned * MARGIN) if learned else t['vram_gb']

    def used(self):
        v = sum(self.learned_vram(r['task']) for r in self.running.values())
        s = sum(r['task']['slots'] for r in self.running.values())
        return v, s

    def admissible(self, t, done):
        if any(a not in done for a in t['after']):
            return False
        if any(r['task'].get('exclusive') for r in self.running.values()):
            return False
        if t.get('exclusive') and self.running:
            return False
        v, s = self.used()
        if v + self.learned_vram(t) > self.cap_vram or s + t['slots'] > self.cap_slots:
            return False
        return fits_walltime(t)

    def claim(self, path):
        dst = DIRS['running'] / path.name
        try:
            os.rename(path, dst)          # atomic on one filesystem: exactly one packer wins
        except (FileNotFoundError, PermissionError):
            return None
        (DIRS['running'] / (path.stem + '.owner')).write_text(self.job)
        (DIRS['running'] / (path.stem + '.hb')).write_text(str(time.time()))
        return dst

    def launch(self, path):
        try:
            t = read_json(path)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f'[{time.strftime("%T")}] claimed task vanished, skipping: {path.name}', flush=True)
            return
        name = path.stem
        logs = ROOT / 'logs' / 'tasks'
        logs.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        threads = max(1, int(self.cpus // max(1, self.cap_slots)))      # libgomp rejects "2.0"
        env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), TOKENIZERS_PARALLELISM='false',
                   S1_VRAM_GB=f'{self.learned_vram(t):.1f}',                  # enforced by the child

                   TRITON_CACHE_DIR=str(ROOT / 'cache' / 'triton' / name),
                   TORCHINDUCTOR_CACHE_DIR=str(ROOT / 'cache' / 'inductor' / name),
                   MIOPEN_USER_DB_PATH=str(ROOT / 'cache' / 'miopen' / name),
                   MIOPEN_CUSTOM_CACHE_DIR=str(ROOT / 'cache' / 'miopen' / name))
        for k in ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'MIOPEN_USER_DB_PATH'):
            Path(env[k]).mkdir(parents=True, exist_ok=True)
        log = open(logs / f'{name}.log', 'a', encoding='utf-8')
        log.write(f'\n=== {time.strftime("%F %T")} job {self.job} start\n')
        log.flush()
        proc = subprocess.Popen([sys.executable] + t['argv'], cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        self.running[name] = dict(proc=proc, task=t, log=log, t0=time.time(), path=path)
        print(f'[{time.strftime("%T")}] start {name} (vram~{self.learned_vram(t):.0f}GB slots {t["slots"]})', flush=True)

    def finish(self, name, rc):
        r = self.running.pop(name)
        r['log'].write(f'=== {time.strftime("%F %T")} exit {rc}\n')
        r['log'].close()
        t, path = r['task'], r['path']
        self._learn_vram(t)
        for suffix in ('.hb', '.owner'):
            (DIRS['running'] / (name + suffix)).unlink(missing_ok=True)
        if not path.exists():
            print(f'[{time.strftime("%T")}] {name} exited {rc} but its claim was moved elsewhere; not touching it', flush=True)
            return
        if rc == 0:
            os.replace(path, DIRS['done'] / path.name)
            state = 'done'
        elif rc == INCOMPLETE:
            os.replace(path, DIRS['pending'] / path.name)          # resumes from its checkpoint
            state = 'requeued (incomplete)'
        else:
            t['retries'] = t.get('retries', 0) + 1
            t['last_error'] = _tail(ROOT / 'logs' / 'tasks' / f'{name}.log')
            atomic_json(path, t)
            os.replace(path, DIRS['pending' if t['retries'] <= 1 else 'failed'] / path.name)
            state = f'failed rc={rc} (retry {t["retries"]})'
        print(f'[{time.strftime("%T")}] {state}: {name} after {(time.time() - r["t0"]) / 3600:.2f} h', flush=True)

    def _learn_vram(self, t):
        f = ROOT / t['run_dir'] / 'peak_mem.json' if t.get('run_dir') else None
        if f and f.exists():
            path = Q / 'vram.json'
            data = read_json(path) if path.exists() else {}
            data[t['vram_key']] = max(data.get(t['vram_key'], 0.0), read_json(f)['peak_gb'])
            atomic_json(path, data)

    def heartbeat(self):
        for name in self.running:
            (DIRS['running'] / (name + '.hb')).write_text(str(time.time()))

    def terminate_all(self):
        for r in self.running.values():
            r['proc'].send_signal(signal.SIGTERM)
        deadline = time.time() + 480
        while self.running and time.time() < deadline:
            for name, r in list(self.running.items()):
                rc = r['proc'].poll()
                if rc is not None:
                    self.finish(name, rc if rc in (0, INCOMPLETE) else INCOMPLETE)
            time.sleep(2)
        for name, r in list(self.running.items()):      # did not stop in time: requeue anyway
            r['proc'].kill()
            self.finish(name, INCOMPLETE)

    def run(self):
        print(f'packer job={self.job} vram={self.cap_vram}GB slots={self.cap_slots} cpus={self.cpus}', flush=True)
        idle_since = None
        while True:
            if self.stop:
                print('SIGTERM: stopping children, requeueing their tasks', flush=True)
                self.terminate_all()
                return
            for name, r in list(self.running.items()):
                rc = r['proc'].poll()
                if rc is not None:
                    self.finish(name, rc)
            self.heartbeat()
            done = done_ids()
            pending = sorted(DIRS['pending'].glob('*.json'))
            for p in pending:
                try:
                    t = read_json(p)
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                if self.admissible(t, done):
                    claimed = self.claim(p)
                    if claimed:
                        self.launch(claimed)
            if not self.running:
                others = list(DIRS['running'].glob('*.json'))
                ready = [t for t in (read_json(p) for p in DIRS['pending'].glob('*.json'))
                         if all(a in done for a in t['after'])]
                if not ready and not others:
                    print('queue drained (or only tasks blocked on missing dependencies); exiting', flush=True)
                    return
                if ready and not any(fits_walltime(t) for t in ready):
                    print('not enough walltime left for any ready task; exiting (resubmit to continue)', flush=True)
                    return
                idle_since = idle_since or time.time()
                if time.time() - idle_since > 6 * 3600:
                    print('idle 6 h waiting on other jobs; exiting', flush=True)
                    return
            else:
                idle_since = None
            time.sleep(self.poll)


def _tail(path, n=40):
    try:
        return ''.join(open(path, encoding='utf-8', errors='replace').readlines()[-n:])
    except FileNotFoundError:
        return ''


def status():
    ensure()
    for k, d in DIRS.items():
        names = sorted(p.stem for p in d.glob('*.json'))
        print(f'{k:8s} {len(names):4d}', ' '.join(names[:6]), '...' if len(names) > 6 else '')
    for p in DIRS['failed'].glob('*.json'):
        t = read_json(p)
        print('\nFAILED', p.stem, '\n', (t.get('last_error') or '')[-800:])


def _live_jobs():
    """Job IDs of this user's queued/running Slurm jobs, or None if squeue cannot be asked.
    Uses the login name from the OS, not $USER (which can be empty inside batch jobs)."""
    import getpass
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%A'],
                             capture_output=True, text=True, check=True, timeout=60).stdout
        return set(out.split())
    except Exception:
        return None


def requeue_stale(max_age_min=15, force=False):
    """Return claims of dead jobs to pending. A claim is requeued only when its owner job is known
    and not in squeue AND its heartbeat is older than max_age_min. If squeue cannot be asked, nothing
    is touched. --force requeues every claim, and is refused while any s1 worker job exists."""
    ensure()
    live = _live_jobs()
    if live is None:
        print('squeue unavailable: not requeueing anything (a live task must never be moved)')
        return
    if force:
        import getpass
        workers = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-n', 's1-worker,s1-smoke', '-o', '%A'],
                                 capture_output=True, text=True).stdout.split()
        if workers:
            print(f'refusing --force: worker jobs still exist: {workers}')
            return
    for p in DIRS['running'].glob('*.json'):
        hb = DIRS['running'] / (p.stem + '.hb')
        owner = DIRS['running'] / (p.stem + '.owner')
        job = owner.read_text().strip() if owner.exists() else None
        try:
            age = (time.time() - float(hb.read_text())) / 60 if hb.exists() else (time.time() - p.stat().st_mtime) / 60
        except (ValueError, FileNotFoundError):
            age = 0.0
        dead = job is not None and job not in live and age > max_age_min
        if force or dead:
            try:
                os.replace(p, DIRS['pending'] / p.name)
            except FileNotFoundError:
                continue
            hb.unlink(missing_ok=True)
            owner.unlink(missing_ok=True)
            print(f'requeued {p.stem} (job {job}, heartbeat {age:.0f} min old{", forced" if force else ""})')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    pk = sub.add_parser('pack')
    pk.add_argument('--vram-gb', type=float, default=float(os.environ.get('S1_GPU_VRAM_GB', 176)))
    pk.add_argument('--max-slots', type=float, default=float(os.environ.get('S1_MAX_SLOTS', 6)))
    pk.add_argument('--cpus', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 12)))
    sub.add_parser('status')
    rq = sub.add_parser('requeue-stale')
    rq.add_argument('--max-age-min', type=float, default=15)
    rq.add_argument('--force', action='store_true', help='requeue every claim (only when no worker job exists)')
    a = ap.parse_args()
    if a.cmd == 'pack':
        Packer(a.vram_gb, a.max_slots, a.cpus).run()
    elif a.cmd == 'status':
        status()
    else:
        requeue_stale(a.max_age_min, a.force)


if __name__ == '__main__':
    main()
