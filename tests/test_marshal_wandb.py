"""Unit tests for the Weights & Biases wiring (``playpen.marshal.wandb_utils``).

Pure logic -- no model, GPU, network, and no ``wandb`` install: the run-opening
test injects a fake ``wandb`` module into ``sys.modules``. Runnable via ``pytest``
or directly:

    .venv/bin/python -m pytest tests/test_marshal_wandb.py
    .venv/bin/python tests/test_marshal_wandb.py
"""

import argparse
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from playpen.marshal.wandb_utils import (
    WandbSettings,
    can_reach_wandb,
    has_wandb_credentials,
    run_metadata,
    sanitize_run_name,
    write_run_metadata,
)


def _args(**overrides):
    """A namespace shaped like train_selfplay.py's, with everything unset."""
    base = dict(
        report_to="none", wandb=None, wandb_project=None, wandb_entity=None,
        wandb_run_name=None, wandb_group=None, wandb_job_type=None, wandb_tags=None,
        wandb_mode=None, wandb_dir=None, wandb_id=None, wandb_resume=None,
        wandb_notes=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestEnablement(unittest.TestCase):
    def test_off_by_default(self):
        self.assertFalse(WandbSettings.from_args(_args(), environ={}).enabled)

    def test_report_to_wandb_enables(self):
        self.assertTrue(WandbSettings.from_args(_args(report_to="wandb"), environ={}).enabled)

    def test_flag_enables(self):
        self.assertTrue(WandbSettings.from_args(_args(wandb=True), environ={}).enabled)

    def test_wandb_project_env_enables(self):
        settings = WandbSettings.from_args(_args(), environ={"WANDB_PROJECT": "diss"})
        self.assertTrue(settings.enabled)
        self.assertEqual(settings.project, "diss")

    def test_env_implied_is_not_required_but_a_flag_is(self):
        # `required` decides whether a missing wandb package is fatal.
        self.assertFalse(WandbSettings.from_args(_args(), environ={"WANDB_PROJECT": "d"}).required)
        self.assertTrue(WandbSettings.from_args(_args(wandb=True), environ={}).required)
        self.assertTrue(WandbSettings.from_args(_args(report_to="wandb"), environ={}).required)

    def test_no_wandb_beats_env_and_report_to(self):
        settings = WandbSettings.from_args(
            _args(wandb=False, report_to="wandb"), environ={"WANDB_PROJECT": "diss"}
        )
        self.assertFalse(settings.enabled)

    def test_mode_disabled_beats_explicit_flag(self):
        settings = WandbSettings.from_args(_args(wandb=True, wandb_mode="disabled"), environ={})
        self.assertFalse(settings.enabled)

    def test_bad_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            WandbSettings.from_args(_args(), environ={"WANDB_MODE": "sideways"})


class TestPrecedence(unittest.TestCase):
    """CLI > WANDB_* env > default, field by field."""

    def test_cli_beats_env(self):
        settings = WandbSettings.from_args(
            _args(wandb=True, wandb_project="from-cli", wandb_entity="team-cli"),
            environ={"WANDB_PROJECT": "from-env", "WANDB_ENTITY": "team-env"},
        )
        self.assertEqual(settings.project, "from-cli")
        self.assertEqual(settings.entity, "team-cli")

    def test_env_beats_default(self):
        settings = WandbSettings.from_args(_args(wandb=True), environ={"WANDB_ENTITY": "team-env"})
        self.assertEqual(settings.project, "playpen-marshal")
        self.assertEqual(settings.entity, "team-env")

    def test_tags_split_on_commas(self):
        settings = WandbSettings.from_args(_args(wandb=True, wandb_tags="a, b ,,c"), environ={})
        self.assertEqual(settings.tags, ["a", "b", "c"])


class TestDefaults(unittest.TestCase):
    def test_names_and_group_derived_from_game_and_model(self):
        settings = WandbSettings(enabled=True).with_defaults(
            output_dir="/runs/x", run_id_stamp="20260726-120000",
            game="dond", model="Qwen/Qwen3-4B",
        )
        self.assertEqual(settings.run_name, "dond_Qwen3-4B_20260726-120000")
        self.assertEqual(settings.group, "dond_Qwen3-4B")
        self.assertEqual(settings.dir, "/runs/x")

    def test_explicit_values_are_not_overwritten(self):
        settings = WandbSettings(
            enabled=True, run_name="mine", group="ablation-1", dir="/elsewhere"
        ).with_defaults(
            output_dir="/runs/x", run_id_stamp="ts", game="dond", model="Qwen/Qwen3-4B",
        )
        self.assertEqual((settings.run_name, settings.group, settings.dir),
                         ("mine", "ablation-1", "/elsewhere"))

    def test_extra_tags_are_appended_without_duplicates(self):
        settings = WandbSettings(enabled=True, tags=["dond"]).with_defaults(
            output_dir="/runs/x", run_id_stamp="ts", extra_tags=["dond", "marshal", ""],
        )
        self.assertEqual(settings.tags, ["dond", "marshal"])

    def test_disabled_settings_are_left_alone(self):
        settings = WandbSettings(enabled=False).with_defaults(
            output_dir="/runs/x", run_id_stamp="ts", game="dond", model="m",
        )
        self.assertIsNone(settings.run_name)
        self.assertIsNone(settings.dir)

    def test_run_name_is_path_safe(self):
        # Model ids carry '/', and the run name is reused as a directory segment.
        self.assertEqual(sanitize_run_name("a/b c:d"), "a_b_c_d")
        self.assertEqual(WandbSettings(enabled=True, run_name="a/b").run_name, "a_b")


class TestModeResolution(unittest.TestCase):
    """`auto` needs BOTH a credential and a reachable API to choose online."""

    def _reachable(self, value):
        return mock.patch("playpen.marshal.wandb_utils.can_reach_wandb", return_value=value)

    def test_auto_is_offline_without_credentials(self):
        settings = WandbSettings(enabled=True, mode="auto")
        with mock.patch("playpen.marshal.wandb_utils.has_wandb_credentials", return_value=False):
            self.assertEqual(settings.resolve_mode({}), "offline")

    def test_auto_is_online_with_credentials_and_network(self):
        settings = WandbSettings(enabled=True, mode="auto")
        with self._reachable(True):
            self.assertEqual(settings.resolve_mode({"WANDB_API_KEY": "k"}), "online")

    def test_auto_is_offline_when_the_api_is_unreachable(self):
        # The cluster case: $HOME is shared, so `wandb login` on the login node
        # leaves a credential a compute node can read but not use.
        settings = WandbSettings(enabled=True, mode="auto")
        with self._reachable(False):
            self.assertEqual(settings.resolve_mode({"WANDB_API_KEY": "k"}), "offline")

    def test_the_probe_runs_at_most_once_per_run(self):
        settings = WandbSettings(enabled=True, mode="auto")
        with mock.patch("playpen.marshal.wandb_utils.can_reach_wandb",
                        return_value=True) as probe:
            for _ in range(4):
                settings.resolve_mode({"WANDB_API_KEY": "k"})
        self.assertEqual(probe.call_count, 1)

    def test_explicit_mode_does_not_probe(self):
        with mock.patch("playpen.marshal.wandb_utils.can_reach_wandb") as probe:
            self.assertEqual(WandbSettings(enabled=True, mode="online").resolve_mode({}), "online")
            self.assertEqual(WandbSettings(enabled=True, mode="offline").resolve_mode({}), "offline")
        probe.assert_not_called()

    def test_disabled_when_off(self):
        self.assertEqual(WandbSettings(enabled=False, mode="online").resolve_mode({}), "disabled")

    def test_api_key_counts_as_a_credential(self):
        self.assertTrue(has_wandb_credentials({"WANDB_API_KEY": "k"}))

    def test_netrc_entry_counts_as_a_credential(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "netrc")
            with open(path, "w") as fh:
                fh.write("machine api.wandb.ai login user password secret\n")
            os.chmod(path, 0o600)
            self.assertTrue(has_wandb_credentials({"NETRC": path}))
            # A netrc for some other host is not a wandb credential.
            self.assertFalse(
                has_wandb_credentials({"NETRC": path, "WANDB_BASE_URL": "https://wandb.example.org"})
            )

    def test_missing_netrc_is_not_a_credential(self):
        self.assertFalse(has_wandb_credentials({"NETRC": "/nonexistent/netrc"}))

    def test_reachability_probe_targets_the_configured_host(self):
        seen = {}

        def fake_create_connection(address, timeout=None):
            seen["address"] = address
            raise OSError("no route to host")

        with mock.patch("socket.create_connection", side_effect=fake_create_connection):
            self.assertFalse(can_reach_wandb({}))
            self.assertEqual(seen["address"], ("api.wandb.ai", 443))
            self.assertFalse(can_reach_wandb({"WANDB_BASE_URL": "http://wandb.internal:8080"}))
            self.assertEqual(seen["address"], ("wandb.internal", 8080))


class TestReportTo(unittest.TestCase):
    def test_adds_wandb_to_none(self):
        self.assertEqual(WandbSettings(enabled=True).report_to("none"), ["wandb"])

    def test_composes_with_tensorboard(self):
        self.assertEqual(
            WandbSettings(enabled=True).report_to("tensorboard"), ["tensorboard", "wandb"]
        )

    def test_no_duplicate(self):
        self.assertEqual(WandbSettings(enabled=True).report_to("wandb"), ["wandb"])

    def test_disabled_strips_wandb(self):
        self.assertEqual(WandbSettings(enabled=False).report_to("wandb"), ["none"])

    def test_disabled_leaves_other_sinks(self):
        self.assertEqual(WandbSettings(enabled=False).report_to("tensorboard"), ["tensorboard"])


class TestEnvOverrides(unittest.TestCase):
    def test_nothing_is_set_when_disabled(self):
        self.assertEqual(WandbSettings(enabled=False, project="p").env_overrides({}), {})

    def test_sets_the_variables_hf_reads(self):
        settings = WandbSettings(
            enabled=True, project="p", entity="e", run_name="r", dir="/d", mode="offline"
        )
        self.assertEqual(settings.env_overrides({}), {
            "WANDB_PROJECT": "p", "WANDB_MODE": "offline",
            "WANDB_ENTITY": "e", "WANDB_NAME": "r", "WANDB_DIR": "/d",
        })

    def test_apply_env_mutates_the_mapping(self):
        environ = {}
        WandbSettings(enabled=True, project="p", mode="offline").apply_env(environ)
        self.assertEqual(environ["WANDB_PROJECT"], "p")
        self.assertEqual(environ["WANDB_MODE"], "offline")


class _FakeRun:
    def __init__(self, directory):
        self.id = "abc123"
        self.name = "dond_Qwen3-4B_ts"
        self.url = "https://wandb.ai/e/p/runs/abc123"
        self.dir = os.path.join(directory, "wandb", "offline-run-ts-abc123", "files")


class TestStart(unittest.TestCase):
    """``start`` must open exactly one run, with every field wandb.init accepts."""

    def setUp(self):
        self.captured = {}
        fake = mock.MagicMock()
        fake.init.side_effect = lambda **kw: (self.captured.update(kw), _FakeRun("/tmp"))[1]
        self.fake = fake
        patcher = mock.patch.dict(sys.modules, {"wandb": fake})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_disabled_returns_none_and_does_not_init(self):
        self.assertIsNone(WandbSettings(enabled=False).start())
        self.fake.init.assert_not_called()

    def test_passes_every_field_through(self):
        settings = WandbSettings(
            enabled=True, project="p", entity="e", run_name="r", group="g",
            job_type="train", tags=["dond"], mode="offline", dir=tempfile.mkdtemp(),
            run_id="fixed-id", resume="allow", notes="n",
        )
        settings.start(config={"marshal": {"enabled": True}})
        self.assertEqual(self.captured["project"], "p")
        self.assertEqual(self.captured["entity"], "e")
        self.assertEqual(self.captured["name"], "r")
        self.assertEqual(self.captured["group"], "g")
        self.assertEqual(self.captured["tags"], ["dond"])
        self.assertEqual(self.captured["mode"], "offline")
        self.assertEqual(self.captured["id"], "fixed-id")
        self.assertEqual(self.captured["resume"], "allow")
        self.assertEqual(self.captured["config"], {"marshal": {"enabled": True}})

    def test_generated_id_is_kept_for_resume(self):
        settings = WandbSettings(enabled=True, mode="offline", dir=tempfile.mkdtemp())
        settings.start()
        self.assertEqual(settings.run_id, "abc123")

    def test_explicit_request_fails_loudly_when_wandb_is_missing(self):
        # `import wandb` raises ImportError when sys.modules holds None for it.
        settings = WandbSettings(enabled=True, required=True, mode="offline")
        with mock.patch.dict(sys.modules, {"wandb": None}):
            with self.assertRaises(SystemExit):
                settings.start()

    def test_implied_request_degrades_when_wandb_is_missing(self):
        # Enabled only because WANDB_PROJECT was in the environment: a stale shell
        # variable must not kill a training job.
        settings = WandbSettings(enabled=True, required=False, mode="offline")
        with mock.patch.dict(sys.modules, {"wandb": None}):
            self.assertIsNone(settings.start())
        self.assertFalse(settings.enabled)
        self.assertEqual(settings.report_to("none"), ["none"])

    def test_only_rank_zero_opens_a_run(self):
        settings = WandbSettings(enabled=True, mode="offline", dir=tempfile.mkdtemp())
        with mock.patch.dict(os.environ, {"RANK": "1"}):
            self.assertIsNone(settings.start())
        self.fake.init.assert_not_called()

    def test_a_failed_online_init_retries_offline(self):
        # The network was reachable when probed and gone by the time we dialled.
        modes = []

        def flaky_init(**kw):
            modes.append(kw.get("mode"))
            if kw.get("mode") == "online":
                raise RuntimeError("connection refused")
            return _FakeRun(tempfile.mkdtemp())

        self.fake.init.side_effect = flaky_init
        settings = WandbSettings(enabled=True, mode="online", dir=tempfile.mkdtemp())
        run = settings.start()
        self.assertIsNotNone(run)
        self.assertEqual(modes, ["online", "offline"])
        self.assertEqual(settings.resolve_mode(), "offline")
        self.assertTrue(settings.enabled)

    def test_a_failed_offline_init_disables_wandb_without_killing_training(self):
        self.fake.init.side_effect = RuntimeError("disk full")
        settings = WandbSettings(enabled=True, mode="offline", dir=tempfile.mkdtemp())
        self.assertIsNone(settings.start())
        self.assertFalse(settings.enabled)
        self.assertEqual(settings.report_to("none"), ["none"])


class TestRunMetadata(unittest.TestCase):
    def test_offline_run_records_the_sync_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _FakeRun(tmp)
            settings = WandbSettings(enabled=True, mode="offline")
            meta = run_metadata(run, settings)
            self.assertIsNone(meta["url"])
            self.assertTrue(meta["offline_dir"].endswith("offline-run-ts-abc123"))
            self.assertEqual(meta["sync_command"], f"wandb sync {meta['offline_dir']}")

    def test_online_run_records_the_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = run_metadata(_FakeRun(tmp), WandbSettings(enabled=True, mode="online"))
            self.assertEqual(meta["url"], "https://wandb.ai/e/p/runs/abc123")
            self.assertNotIn("sync_command", meta)

    def test_write_run_metadata_is_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out", "wandb_run.json")
            settings = WandbSettings(enabled=True, mode="offline")
            self.assertEqual(write_run_metadata(path, _FakeRun(tmp), settings), path)
            with open(path) as fh:
                data = json.load(fh)
            self.assertEqual(data["id"], "abc123")
            self.assertEqual(data["wandb"]["resolved_mode"], "offline")

    def test_write_run_metadata_is_a_no_op_without_a_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wandb_run.json")
            self.assertIsNone(write_run_metadata(path, None, WandbSettings()))
            self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
