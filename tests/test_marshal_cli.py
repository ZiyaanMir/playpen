"""The CLI -> MarshalConfig override chain in ``examples/marshal/train_selfplay.py``.

``train_selfplay.py`` documents an invariant it cannot enforce: "a new config field
cannot be added without also being wired up here". Nothing checked that, so a field
added to ``MarshalConfig`` would be settable in the YAML and silently unreachable from
the command line -- which on a cluster means editing a shared file to change one run.

The merge itself lives inside ``main()``, which loads a model, so it is reproduced here
against the real ``parse_args()`` and the real ``MARSHAL_CLI_FIELDS``. That keeps the
precedence rules (YAML < CLI, and ``--no-sampling-truncation`` < an explicit value)
pinned even though ``main()`` is not directly callable.

Runnable via ``pytest`` or directly with ``.venv/bin/python``.
"""

import dataclasses
import sys
import unittest
from dataclasses import fields
from unittest import mock

from examples.marshal import train_selfplay as ts

from playpen.marshal.config import MarshalConfig

# Fields deliberately NOT driven by a `--<field>` flag, with the reason.
_CLI_EXEMPT = {
    # Keeps its historical --marshal/--no-marshal spelling.
    "enabled",
}


def _merge(argv, yaml_values=None):
    """Drive the REAL merge (``ts.resolve_marshal_config``) for a given argv.

    Deliberately not a local re-implementation: a test that reproduces the logic it is
    checking cannot fail when that logic changes. ``yaml_values`` is written to a temp
    YAML so the "YAML < CLI" half of the precedence rule is exercised for real too.
    """
    with mock.patch.object(sys, "argv", ["train_selfplay.py", *argv]):
        args = ts.parse_args()
    if yaml_values is None:
        path = None  # use the shipped YAML, i.e. args.marshal_config
    else:
        import tempfile

        import yaml as _yaml

        fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        _yaml.safe_dump(yaml_values, fh)
        fh.close()
        path = fh.name
    return ts.resolve_marshal_config(args, config_path=path)


class TestEveryConfigFieldIsReachableFromTheCli(unittest.TestCase):
    def test_no_field_is_silently_unreachable(self):
        declared = {f.name for f in fields(MarshalConfig)}
        wired = set(ts.MARSHAL_CLI_FIELDS) | _CLI_EXEMPT
        missing = declared - wired
        self.assertEqual(
            missing, set(),
            f"MarshalConfig fields with no CLI override: {sorted(missing)}. Add them to "
            f"MARSHAL_CLI_FIELDS (and an argparse flag whose dest is the field name), or "
            f"to _CLI_EXEMPT here with the reason.",
        )

    def test_no_stale_entry_in_the_wiring_list(self):
        declared = {f.name for f in fields(MarshalConfig)}
        stale = set(ts.MARSHAL_CLI_FIELDS) - declared
        self.assertEqual(stale, set(), f"MARSHAL_CLI_FIELDS names non-fields: {sorted(stale)}")

    def test_every_wired_field_has_a_matching_argparse_dest(self):
        with mock.patch.object(sys, "argv", ["train_selfplay.py"]):
            args = ts.parse_args()
        for name in ts.MARSHAL_CLI_FIELDS:
            self.assertTrue(hasattr(args, name), f"no argparse dest for {name!r}")
            self.assertIsNone(
                getattr(args, name),
                f"{name!r} must default to None so 'not passed' leaves the YAML alone",
            )


class TestSamplingTruncationOverrides(unittest.TestCase):
    def test_nothing_passed_leaves_the_config_alone(self):
        cfg, overrides = _merge([])
        self.assertEqual(overrides, {})
        self.assertEqual(cfg.sampling_top_p, 0.95)
        self.assertEqual(cfg.sampling_top_k, 50)

    def test_each_value_can_be_overridden_independently(self):
        cfg, _ = _merge(["--sampling-top-p", "0.8"])
        self.assertEqual((cfg.sampling_top_p, cfg.sampling_top_k), (0.8, 50))
        cfg, _ = _merge(["--sampling-top-k", "20"])
        self.assertEqual((cfg.sampling_top_p, cfg.sampling_top_k), (0.95, 20))

    def test_no_sampling_truncation_reverts_both_and_adds_no_grpo_key(self):
        cfg, _ = _merge(["--no-sampling-truncation"])
        self.assertEqual((cfg.sampling_top_p, cfg.sampling_top_k), (1.0, 0))
        self.assertEqual(cfg.trl_sampling_overrides(), {})

    def test_an_explicit_value_wins_over_no_sampling_truncation(self):
        """Consistent with every other flag: what you typed applies."""
        cfg, _ = _merge(["--no-sampling-truncation", "--sampling-top-p", "0.9"])
        self.assertEqual((cfg.sampling_top_p, cfg.sampling_top_k), (0.9, 0))
        self.assertEqual(cfg.trl_sampling_overrides(), {"top_p": 0.9})

    def test_it_reverts_a_yaml_that_asked_for_truncation(self):
        cfg, _ = _merge(
            ["--no-sampling-truncation"],
            yaml_values={"sampling_top_p": 0.7, "sampling_top_k": 5},
        )
        self.assertEqual(cfg.trl_sampling_overrides(), {})

    def test_it_turns_truncation_on_over_a_neutral_yaml(self):
        cfg, _ = _merge(
            ["--sampling-top-p", "0.95", "--sampling-top-k", "50"],
            yaml_values={"sampling_top_p": 1.0, "sampling_top_k": 0},
        )
        self.assertEqual(cfg.trl_sampling_overrides(), {"top_p": 0.95, "top_k": 50})

    def test_a_yaml_value_is_used_when_no_flag_is_passed(self):
        cfg, overrides = _merge([], yaml_values={"sampling_top_p": 0.7, "sampling_top_k": 5})
        self.assertEqual(overrides, {})
        self.assertEqual(cfg.trl_sampling_overrides(), {"top_p": 0.7, "top_k": 5})


class TestGrpoConfigWiring(unittest.TestCase):
    """The overrides must actually be splatted into GRPOConfig.

    A structural check, not a behavioural one: ``main()`` builds the trainer and cannot
    be called from a test. It exists because deleting the splat is otherwise invisible --
    every unit test of ``trl_sampling_overrides()`` still passes while no run is
    truncated. If ``main()`` is ever restructured so this grep no longer applies, replace
    it with a real assertion rather than deleting it.
    """

    def _main_source(self):
        import inspect
        return inspect.getsource(ts.main)

    def test_sampling_overrides_are_splatted_into_grpo_config(self):
        src = self._main_source()
        self.assertIn("**marshal_config.trl_sampling_overrides()", src)

    def test_dr_grpo_overrides_are_still_splatted_too(self):
        src = self._main_source()
        self.assertIn("**marshal_config.trl_grpo_overrides()", src)

    def test_both_splats_are_inside_the_grpo_config_call(self):
        src = self._main_source()
        start = src.index("trl.GRPOConfig(")
        end = src.index("peft_config = LoraConfig", start)
        call = src[start:end]
        self.assertIn("**marshal_config.trl_sampling_overrides()", call)
        self.assertIn("**marshal_config.trl_grpo_overrides()", call)

    def test_validation_runs_on_the_merged_value(self):
        for argv in (["--sampling-top-p", "1.5"], ["--sampling-top-k", "-3"]):
            with self.assertRaises(ValueError, msg=f"{argv} should not be accepted"):
                _merge(argv)

    def test_unrelated_flags_do_not_disturb_it(self):
        cfg, _ = _merge(["--no-marshal"])
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.trl_sampling_overrides(), {"top_p": 0.95, "top_k": 50})


class TestShippedYamlMatchesTheDataclass(unittest.TestCase):
    """The shipped YAML must load, and must not drift from the field set."""

    def _yaml_path(self):
        import os
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(ts.__file__))),
            "marshal", "marshal_config.yaml",
        )

    def test_shipped_yaml_loads(self):
        try:
            cfg = MarshalConfig.from_yaml(self._yaml_path())
        except ImportError:  # pragma: no cover - PyYAML absent
            self.skipTest("PyYAML not installed")
        self.assertIsInstance(cfg, MarshalConfig)

    def test_shipped_yaml_sets_the_sampling_knobs(self):
        try:
            cfg = MarshalConfig.from_yaml(self._yaml_path())
        except ImportError:  # pragma: no cover
            self.skipTest("PyYAML not installed")
        self.assertEqual(cfg.sampling_top_p, 0.95)
        self.assertEqual(cfg.sampling_top_k, 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
