"""Local desktop alerts for the active queue; does not change training sources."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from training_safety import atomic_json, training_lock


def attention(state, now, waiting_seconds=0):
    status=state.get('status','UNKNOWN')
    job=state.get('job','training queue')
    if status in {'FAILED','INTERRUPTED'}:
        return f"Training needs attention: {job}. {state.get('message','Open queue_status.json for details.')}"
    if status=='COMPLETE':
        return 'Both models finished. Final training and holdout results are ready to review.'
    if status in {'WAITING_FOR_RAM','RESOURCE_RETRY'} and waiting_seconds>=600:
        return f'Training paused for resources: {job}. Close memory-heavy applications; the queue will retry automatically.'
    updated=datetime.fromisoformat(state['updated_utc'])
    if (now-updated).total_seconds()>180:
        return 'Training monitor stopped updating for over 3 minutes. Check the queue and its logs before restarting anything.'
    return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir',required=True)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    batch=Path(args.batch_dir).resolve()
    memo_path=batch/'alert_state.json'
    memo=json.loads(memo_path.read_text()) if memo_path.exists() else {}
    waiting_since=None
    with training_lock(batch/'.alerts.lock'):
        while True:
            state=json.loads((batch/'queue_status.json').read_text())
            now=datetime.now(timezone.utc)
            if state.get('status') in {'WAITING_FOR_RAM','RESOURCE_RETRY'}:
                waiting_since=waiting_since or time.monotonic()
            else:
                waiting_since=None
            message=attention(state,now,time.monotonic()-waiting_since if waiting_since else 0)
            if message and message!=memo.get('last_message'):
                print(message,flush=True)
                if not args.dry_run:
                    atomic_json(batch/'ATTENTION_NEEDED.json',dict(time_utc=now.isoformat(),message=message,
                                queue_status=state.get('status'),job=state.get('job'),log=state.get('log')))
                    result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
                        str(Path(__file__).with_name('training_notification.ps1')),
                        '-Message',message[:900]],stdin=subprocess.DEVNULL,timeout=45,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    memo=dict(last_message=message,notified_utc=now.isoformat(),notification_exit_code=result.returncode)
                    atomic_json(memo_path,memo)
            if not args.dry_run:
                atomic_json(batch/'alert_monitor_status.json',dict(status='MONITORING',checked_utc=now.isoformat(),
                            queue_status=state.get('status'),attention=message,notification='Windows desktop balloon and sound'))
            if args.once or state.get('status')=='COMPLETE': break
            time.sleep(30)


if __name__=='__main__': main()
