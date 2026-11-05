"""Bounded resources and transactional, completed-epoch recovery.

Only generated files in a new training run are updated. Recovery is not a claim
of bit-identical random augmentation/shuffle after restarting a process.
"""
from contextlib import contextmanager
import csv
import ctypes
import json
import math
import os
from pathlib import Path
import shutil


class ResourceLimitError(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def memory_available():
    if os.name == 'nt':
        from ctypes import wintypes
        fields = [('length', wintypes.DWORD), ('load', wintypes.DWORD)]
        fields += [(n, ctypes.c_ulonglong) for n in
                   ['total', 'available', 'page_total', 'page_available',
                    'virtual_total', 'virtual_available', 'extended']]
        status = type('MemoryStatus', (ctypes.Structure,), {'_fields_': fields})()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError('Cannot query available physical memory')
        return status.available / 2**30
    with open('/proc/meminfo', encoding='ascii') as stream:
        for line in stream:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2**20
    raise OSError('Cannot query available physical memory')


def check_resources(config, directory):
    available = memory_available()
    free = shutil.disk_usage(directory).free / 2**30
    if available < config.get('min_free_ram_gb', 1.5):
        raise ResourceLimitError(f'Available RAM {available:.2f} GiB is below reserve '
                                 f"{config.get('min_free_ram_gb', 1.5):.2f} GiB. Close other applications, then resume.")
    if free < config.get('min_free_disk_gb', 5.0):
        raise ResourceLimitError(f'Free disk {free:.2f} GiB is below the configured reserve.')
    return dict(available_ram_gib=available, free_disk_gib=free, logical_cpus=os.cpu_count())


def configure_runtime(tf, config):
    threads = config.get('cpu_threads', 4)
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


@contextmanager
def training_lock(path):
    """OS-released lock prevents concurrent project training, including after a crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+b')
    stream.seek(0, 2)
    if stream.tell() == 0:
        stream.write(b'0'); stream.flush()
    stream.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise RuntimeError('Another training/check process holds the project training lock.') from exc
    try:
        yield
    finally:
        stream.close()


def snapshot_path(out, name):
    # The recovery JSON is local evidence, but must not redirect loads/deletes.
    if Path(name).name != name or not name.startswith('epoch_') or not name.endswith('.keras'):
        raise ValueError('Invalid recovery checkpoint name')
    return Path(out) / name


def recovery_callback(tf, out, config, state=None):
    """Save optimizer/model + stopping/LR state before publishing the epoch pointer."""
    out = Path(out)

    class Recovery(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.state = state or dict(completed_epochs=0, best_loss=None, best_file=None,
                                       wait=0, lr_wait=0, history={}, finished=False)

        def on_train_batch_begin(self, batch, logs=None):
            check_resources(config, out)

        def on_test_batch_begin(self, batch, logs=None):
            check_resources(config, out)

        def on_train_batch_end(self, batch, logs=None):
            if any(not math.isfinite(float(v)) for v in (logs or {}).values()):
                raise FloatingPointError('Nonfinite training metric; resume uses last completed epoch.')

        def on_epoch_end(self, epoch, logs=None):
            logs = {k: float(v) for k, v in (logs or {}).items()}
            if 'val_loss' not in logs or any(not math.isfinite(v) for v in logs.values()):
                raise FloatingPointError('Missing/nonfinite validation loss; epoch was not committed.')
            previous = dict(self.state)
            best = self.state['best_loss']
            improved = best is None or logs['val_loss'] < best
            if improved:
                self.state.update(best_loss=logs['val_loss'], wait=0)
            else:
                self.state['wait'] += 1
            # ReduceLROnPlateau equivalent (min_delta=1e-4), persisted across resumes.
            lr_best = self.state.get('lr_best')
            if lr_best is None or logs['val_loss'] < lr_best - 1e-4:
                self.state.update(lr_best=logs['val_loss'], lr_wait=0)
            else:
                self.state['lr_wait'] += 1
                if self.state['lr_wait'] >= max(1, config['patience']//2):
                    lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                    self.model.optimizer.learning_rate.assign(max(config.get('lr_min',0.),lr * config.get('lr_reduce_factor',.5)))
                    self.state['lr_wait'] = 0
            name = f'epoch_{epoch+1:04d}.keras'
            temporary = out / 'epoch_pending.keras'
            self.model.save(temporary)
            os.replace(temporary, snapshot_path(out, name))
            if improved:
                self.state['best_file'] = name
            accuracy = logs.get('val_accuracy')
            if accuracy is not None and (self.state.get('best_accuracy') is None or accuracy>self.state['best_accuracy']):
                self.state.update(best_accuracy=accuracy,best_accuracy_file=name)
            for key, value in logs.items():
                self.state['history'].setdefault(key, []).append(value)
            self.state.update(completed_epochs=epoch+1, latest_file=name,
                              finished=epoch+1 >= config['epochs'] or self.state['wait'] >= config['patience'])
            atomic_json(out/'recovery_state.json', self.state)
            # Recovery state is canonical: regenerate these if a process dies during export.
            export_history(out, self.state)
            # Keep the latest optimizer state and the best model; discard older generated snapshots.
            keep = {name, self.state['best_file'], self.state.get('best_accuracy_file')}
            for old in {previous.get('latest_file'), previous.get('best_file'),previous.get('best_accuracy_file')} - keep - {None}:
                snapshot_path(out, old).unlink(missing_ok=True)
            if self.state['finished']:
                self.model.stop_training = True

    return Recovery()


def export_history(out, state):
    atomic_json(Path(out)/'training_history.json', state['history'])
    path = Path(out)/'training_log.csv'
    temporary = path.with_suffix('.csv.tmp')
    keys = sorted(state['history'])
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream); writer.writerow(['epoch']+keys)
        for epoch in range(state['completed_epochs']):
            writer.writerow([epoch]+[state['history'][k][epoch] for k in keys])
    os.replace(temporary, path)
