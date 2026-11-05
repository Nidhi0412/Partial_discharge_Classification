from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from training_alerts import attention


class AlertTests(unittest.TestCase):
    def test_healthy_run_stays_quiet(self):
        now=datetime.now(timezone.utc)
        self.assertIsNone(attention(dict(status='RUNNING',updated_utc=now.isoformat()),now))

    def test_failures_completion_and_stale_heartbeat_notify(self):
        now=datetime.now(timezone.utc)
        for status in ['FAILED','INTERRUPTED','COMPLETE']:
            self.assertIsNotNone(attention(dict(status=status),now))
        self.assertIsNotNone(attention(dict(status='RUNNING',updated_utc=(now-timedelta(minutes=4)).isoformat()),now))

    def test_short_resource_wait_is_quiet_long_wait_notifies(self):
        now=datetime.now(timezone.utc)
        state=dict(status='WAITING_FOR_RAM',updated_utc=now.isoformat())
        self.assertIsNone(attention(state,now,60))
        self.assertIsNotNone(attention(state,now,600))


if __name__=='__main__': unittest.main()
