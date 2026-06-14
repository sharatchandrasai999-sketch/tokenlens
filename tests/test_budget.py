import os, tempfile
from tokenlens.budget import check_budget, SpendLedger, OK, WARN, BLOCK

def test_under_limit_ok():
    assert check_budget(0.01, per_request_limit=1.0)["decision"] == OK

def test_over_per_request_blocks():
    assert check_budget(2.0, per_request_limit=1.0)["decision"] == BLOCK

def test_warn_band():
    # 0.85 of a 1.0 daily limit -> warn
    assert check_budget(0.85, daily_limit=1.0)["decision"] == WARN

def test_daily_limit_blocks_projected():
    assert check_budget(0.5, spent_today=0.7, daily_limit=1.0)["decision"] == BLOCK

def test_ledger_records_and_reads():
    fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd); os.remove(path)
    led = SpendLedger(path)
    assert led.spent_today() == 0.0
    led.record(0.25); 
    assert abs(SpendLedger(path).spent_today() - 0.25) < 1e-9
    os.remove(path)
