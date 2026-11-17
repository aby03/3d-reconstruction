from conftest import assert_stage_ok


def test_smoke(smoke_run):
    assert_stage_ok(smoke_run.stage_dir("align"))
