"""Checkpoint discovery, resume resolution and segment planning.

``playpen/marshal/resume.py`` decides two things that are expensive to get wrong on a
cluster and impossible to notice quickly:

* WHICH checkpoint a job continues from. Resuming a half-written one crashes hours
  later; refusing a good one silently retrains from step 0 and burns a walltime.
* WHERE a chain's segment boundaries fall. They have to tile ``max_steps`` exactly --
  a gap loses steps, an overlap wastes a job.

Both are checked here against real directory trees rather than mocks, because the
completeness rule *is* a statement about files on disk (``trainer_state.json`` is
written last by ``Trainer._save_checkpoint``, which is what makes its absence mean
"killed mid-save").

The shell half of the same arithmetic lives in ``exp_plan_segments`` /
``exp_segment_stop_at`` in ``experiments/lib/experiment.sh``; ``test_segment_bounds``
below pins the Python side, and the two are compared in
``test_shell_and_python_agree``.

Runnable via ``pytest`` or directly with ``.venv/bin/python``.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

from playpen.marshal import resume

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_checkpoint(root, step, *, state=True, adapter=True, optimizer=True,
                     scheduler=True, global_step=None):
    """Create a ``checkpoint-<step>`` directory with a chosen subset of its files.

    Mirrors what ``Trainer._save_checkpoint`` leaves behind, so a test can construct
    the exact partial state a job killed at the walltime produces.
    """
    path = os.path.join(root, f"checkpoint-{step}")
    os.makedirs(path, exist_ok=True)
    if adapter:
        open(os.path.join(path, "adapter_model.safetensors"), "wb").close()
        open(os.path.join(path, "adapter_config.json"), "w").close()
    if optimizer:
        open(os.path.join(path, "optimizer.pt"), "wb").close()
    if scheduler:
        open(os.path.join(path, "scheduler.pt"), "wb").close()
    if state:
        with open(os.path.join(path, "trainer_state.json"), "w") as fh:
            json.dump({"global_step": global_step if global_step is not None else step}, fh)
    return path


class CheckpointDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_latest_is_by_step_not_lexical(self):
        # The trap exp_list_checkpoints already documents: sorting the names puts
        # checkpoint-50 after checkpoint-200 and resumes from the wrong place.
        for step in (50, 100, 200):
            write_checkpoint(self.root, step)
        self.assertEqual(
            os.path.basename(resume.latest_checkpoint(self.root)), "checkpoint-200"
        )

    def test_half_written_checkpoint_is_skipped(self):
        write_checkpoint(self.root, 100)
        write_checkpoint(self.root, 200, state=False)   # killed during the save
        latest = resume.latest_checkpoint(self.root)
        self.assertEqual(os.path.basename(latest), "checkpoint-100")

    def test_adapterless_checkpoint_is_skipped(self):
        write_checkpoint(self.root, 100)
        write_checkpoint(self.root, 200, adapter=False)
        self.assertEqual(
            os.path.basename(resume.latest_checkpoint(self.root)), "checkpoint-100"
        )

    def test_missing_optimizer_is_a_warning_not_a_rejection(self):
        # Weights and the step counter still resume; only the Adam moments restart.
        # That is a real perturbation, so it must be reported -- but refusing to
        # resume at all would be worse.
        path = write_checkpoint(self.root, 100, optimizer=False)
        ok, reason = resume.is_resumable(path)
        self.assertTrue(ok)
        self.assertIn("optimizer.pt", reason)
        self.assertIn("WARNING", reason)

    def test_empty_and_missing_dirs(self):
        self.assertIsNone(resume.latest_checkpoint(self.root))
        self.assertIsNone(resume.latest_checkpoint(os.path.join(self.root, "nope")))
        self.assertEqual(resume.list_checkpoints(self.root), [])

    def test_non_checkpoint_dirs_ignored(self):
        os.makedirs(os.path.join(self.root, "completions"))
        os.makedirs(os.path.join(self.root, "checkpoint-150-partial"))
        write_checkpoint(self.root, 100)
        self.assertEqual([s for s, _ in resume.list_checkpoints(self.root)], [100])

    def test_step_comes_from_trainer_state_not_the_name(self):
        # The file is what training actually continues from. If the two ever disagree,
        # a plan printed from the directory name would be a plausible-looking lie.
        path = write_checkpoint(self.root, 200, global_step=193)
        self.assertEqual(resume.resumed_global_step(path), 193)

    def test_step_falls_back_to_the_name_when_state_is_unreadable(self):
        path = write_checkpoint(self.root, 200)
        with open(os.path.join(path, "trainer_state.json"), "w") as fh:
            fh.write("{ not json")
        self.assertEqual(resume.resumed_global_step(path), 200)


class ResolveResumeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_none_means_fresh(self):
        write_checkpoint(self.root, 100)
        for spec in (None, "", "none", "NONE"):
            path, _ = resume.resolve_resume(spec, self.root)
            self.assertIsNone(path, spec)

    def test_auto_starts_fresh_when_there_is_nothing(self):
        # Segment 1 of a chain passes 'auto' and must not error on an empty train/.
        path, why = resume.resolve_resume("auto", self.root)
        self.assertIsNone(path)
        self.assertIn("starting at step 0", why)

    def test_auto_resumes_when_there_is_something(self):
        write_checkpoint(self.root, 400)
        path, why = resume.resolve_resume("auto", self.root)
        self.assertEqual(os.path.basename(path), "checkpoint-400")
        self.assertIn("global step 400", why)

    def test_latest_errors_when_there_is_nothing(self):
        # Segments 2+ pass 'latest'. An empty train/ there means the previous job died
        # before its first save, and starting over silently would cost a walltime.
        with self.assertRaises(resume.ResumeError):
            resume.resolve_resume("latest", self.root)

    def test_latest_error_names_the_partial_checkpoint(self):
        write_checkpoint(self.root, 300, state=False)
        with self.assertRaises(resume.ResumeError) as ctx:
            resume.resolve_resume("latest", self.root)
        self.assertIn("checkpoint-300", str(ctx.exception))

    def test_explicit_checkpoint_path(self):
        target = write_checkpoint(self.root, 100)
        write_checkpoint(self.root, 200)
        path, _ = resume.resolve_resume(target, self.root)
        self.assertEqual(path, os.path.abspath(target))

    def test_explicit_run_dir_takes_its_latest(self):
        # So RESUME_FROM=$EXP_DIR/train does the obvious thing.
        write_checkpoint(self.root, 100)
        write_checkpoint(self.root, 200)
        path, _ = resume.resolve_resume(self.root, "/nonexistent")
        self.assertEqual(os.path.basename(path), "checkpoint-200")

    def test_explicit_bad_path_errors(self):
        with self.assertRaises(resume.ResumeError):
            resume.resolve_resume(os.path.join(self.root, "nope"), self.root)

    def test_explicit_half_written_checkpoint_errors(self):
        target = write_checkpoint(self.root, 200, state=False)
        with self.assertRaises(resume.ResumeError) as ctx:
            resume.resolve_resume(target, self.root)
        self.assertIn("trainer_state.json", str(ctx.exception))


class SegmentPlanTests(unittest.TestCase):
    def test_segment_bounds_tile_the_total_exactly(self):
        for total, size, expected in [
            (1000, 400, [400, 800, 1000]),
            (1000, 500, [500, 1000]),
            (1000, 1000, [1000]),
            (1000, 0, [1000]),          # unsegmented
            (1000, 2000, [1000]),       # a segment bigger than the run
            (7, 2, [2, 4, 6, 7]),
        ]:
            with self.subTest(total=total, size=size):
                self.assertEqual(resume.segment_bounds(total, size), expected)

    def test_last_bound_is_always_the_total(self):
        # A chain must never overshoot the horizon the LR scheduler was built for.
        for size in range(1, 60):
            self.assertEqual(resume.segment_bounds(500, size)[-1], 500)

    def test_bounds_are_strictly_increasing(self):
        for size in range(1, 60):
            bounds = resume.segment_bounds(500, size)
            self.assertEqual(sorted(set(bounds)), bounds, f"size={size}")

    def test_describe_plan_names_this_job(self):
        text = resume.describe_plan(1000, 400, index=2)
        self.assertIn("segment 2/3", text)
        self.assertIn("400 -> 800", text)


class LrScheduleInvarianceTests(unittest.TestCase):
    """The property the whole segmented design exists to preserve.

    ``--max-steps`` stays the TOTAL in every segment and only ``--stop-at-step``
    differs. This test is what says that is not merely a convention: it drives HF's
    own scheduler through a save/restore at a segment boundary and checks the
    resulting learning rates against an uninterrupted run, then shows that the
    obvious alternative -- handing segment *k* ``max_steps = k*S`` -- corrupts them.

    Verified against real training too (SmolLM2-135M, taboo, 4 steps as 2+2 vs 4):
    both runs ended checkpoint-2 at lr 5e-06 and checkpoint-4 at lr 0.0. This keeps
    that pinned without a GPU.
    """

    LR = 1e-5

    def _scheduler(self, total_steps):
        import torch
        from transformers import get_linear_schedule_with_warmup

        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([param], lr=self.LR)
        return opt, get_linear_schedule_with_warmup(opt, 0, total_steps)

    def _walk(self, total_steps, boundaries):
        """Learning rates over ``total_steps``, restarting the scheduler at each boundary.

        ``boundaries`` is the list of ``max_steps`` values each segment is built with;
        the state is saved and restored between them exactly as scheduler.pt does.
        """
        seen, state, done = [], None, 0
        for max_steps in boundaries:
            opt, sched = self._scheduler(max_steps)
            if state is not None:
                sched.load_state_dict(state)
            while done < max_steps and done < total_steps:
                seen.append(opt.param_groups[0]["lr"])
                opt.step()
                sched.step()
                done += 1
            state = sched.state_dict()
        return seen

    def test_segmenting_does_not_change_the_schedule(self):
        # One 10-step run, versus the same 10 steps as 4 + 4 + 2, every segment built
        # with max_steps=10 (what train.sh does).
        uninterrupted = self._walk(10, [10])
        segmented = self._walk(10, [10, 10, 10])
        self.assertEqual(uninterrupted, segmented)
        self.assertEqual(len(uninterrupted), 10)

    def test_per_segment_max_steps_would_corrupt_it(self):
        # The design that was NOT chosen: segment k gets max_steps = k*S. HF builds
        # the decay from max_steps and scheduler.pt restores only the step counter,
        # never the lambda -- so each segment re-derives a STEEPER slope, racing the
        # LR toward zero over its own span instead of the run's.
        correct = self._walk(10, [10])
        naive = self._walk(10, [4, 8, 10])
        self.assertNotEqual(naive, correct)

        # Segment 1 thinks it has 4 steps, so by its last one it has decayed most of
        # the way: 2.5e-06 where the real schedule is at 7e-06, nearly 3x apart.
        self.assertAlmostEqual(correct[3], 7e-06)
        self.assertAlmostEqual(naive[3], 2.5e-06)
        # And segment 2 jumps back UP, because its own slope is gentler than
        # segment 1's was.
        self.assertGreater(naive[4], naive[3])


class PeftCompatTests(unittest.TestCase):
    """``ensure_peft_resume_compat`` must fix the ImportError WITHOUT changing what loads.

    The shim skips peft's tensor-parallel sharding pass at ``world_size == 1``, on the
    argument that sharding across a one-rank mesh is the identity. That argument is
    easy to state and would be quiet if wrong -- a resume that loaded *nothing* would
    keep training happily from the base weights and only show up as an arm that
    mysteriously underperformed. So it is checked rather than reasoned about: save an
    adapter, zero it, load it back through the real ``load_adapter`` path with the shim
    installed and a real single-rank process group up, and require an exact match.
    """

    def test_adapter_round_trips_exactly_through_the_shim(self):
        import tempfile

        import torch
        import torch.distributed as dist
        import torch.nn as nn
        from peft import LoraConfig, get_peft_model

        from playpen.marshal.resume import ensure_peft_resume_compat

        if dist.is_initialized():
            self.skipTest("a process group is already up; this test owns its own")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29579")
        try:
            # A real group, because that is what makes peft take the TP branch at all
            # (its only guard is torch.distributed.is_initialized()). TRL's colocate
            # vLLM mode opens exactly this: one rank, no tensor parallelism.
            dist.init_process_group("gloo", rank=0, world_size=1)
        except Exception as exc:  # no gloo build, port in use, ...
            self.skipTest(f"cannot init a process group here: {exc}")

        try:
            base = nn.Sequential()
            base.add_module("lin", nn.Linear(16, 16))
            model = get_peft_model(base, LoraConfig(target_modules=["lin"], r=4))

            saved = {}
            for name, param in model.named_parameters():
                if "lora_" in name:
                    torch.nn.init.normal_(param, std=0.5)
                    saved[name] = param.detach().clone()
            self.assertTrue(saved, "no LoRA parameters to test with")

            with tempfile.TemporaryDirectory() as ckpt:
                model.save_pretrained(ckpt)
                # Zero them, so a load that silently does nothing is detectable.
                for name, param in model.named_parameters():
                    if "lora_" in name:
                        param.data.zero_()

                ensure_peft_resume_compat()
                model.load_adapter(ckpt, "default", is_trainable=True)

            for name, param in model.named_parameters():
                if name in saved:
                    self.assertTrue(torch.equal(param.detach(), saved[name]), name)
        finally:
            dist.destroy_process_group()


class ShellAgreementTests(unittest.TestCase):
    """The submitter (bash) and the trainer (Python) must agree on the boundaries.

    ``run_experiment.sh`` decides how many jobs to queue and what each one's
    ``--stop-at-step`` is; ``resume.segment_bounds`` is what the manifest and the
    banners print. If those two ever disagreed the manifest would describe a chain
    that was not submitted -- so they are compared here rather than trusted.
    """

    def _shell_bounds(self, total, save, segments):
        script = (
            f'source "{REPO}/experiments/lib/experiment.sh"\n'
            f"MAX_STEPS={total} SAVE_STEPS={save} TRAIN_SEGMENTS={segments} SEGMENT_STEPS=\n"
            "exp_plan_segments >/dev/null 2>&1\n"
            'for k in $(seq 1 "$TRAIN_SEGMENTS"); do exp_segment_stop_at "$k"; done\n'
            'echo "SIZE=$SEGMENT_STEPS"\n'
        )
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                             cwd=REPO, check=True).stdout.split()
        size = int(out[-1].split("=")[1])
        return [int(v) for v in out[:-1]], size

    def test_shell_and_python_agree(self):
        for total, save, segments in [
            (1000, 100, 1), (1000, 100, 2), (1000, 100, 3),
            (1000, 100, 4), (1000, 100, 6), (200, 50, 3),
        ]:
            with self.subTest(total=total, save=save, segments=segments):
                shell, size = self._shell_bounds(total, save, segments)
                self.assertEqual(shell, resume.segment_bounds(total, size))
                self.assertEqual(shell[-1], total)

    def _resume_specs(self, segments, resume_from=""):
        script = (
            f'source "{REPO}/experiments/lib/experiment.sh"\n'
            f"RESUME_FROM={resume_from!r}\n"
            f'for k in $(seq 1 {segments}); do exp_segment_resume_spec "$k"; done\n'
        )
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                              cwd=REPO, check=True).stdout.split()

    def test_only_the_first_segment_may_start_from_nothing(self):
        # Segment 1 tolerates an empty train/ ('auto'); every later one treats it as
        # an error ('latest'), because by then it means the previous job died before
        # its first save and restarting at step 0 would silently cost a walltime.
        self.assertEqual(self._resume_specs(4), ["auto", "latest", "latest", "latest"])

    def test_explicit_resume_from_applies_to_the_first_segment_only(self):
        # Letting a pinned checkpoint reach segment 2 would make it resume the same
        # one and redo segment 1's work.
        self.assertEqual(
            self._resume_specs(3, resume_from="/runs/exp/train/checkpoint-500"),
            ["/runs/exp/train/checkpoint-500", "latest", "latest"],
        )


class SegmentOverrideTests(unittest.TestCase):
    """``exp_segment_override``: what the caller set now beats what was stored.

    The bug it fixes: ``TRAIN_SEGMENTS=5 resume_experiment.sh <EXP_DIR>`` read
    ``SEGMENT_STEPS=400`` out of the original submission's ``experiment.env``, and
    ``exp_plan_segments`` preferred it as the more specific of the two -- so the
    operator got 3 segments of 400 and a plan line that looked deliberate.
    """

    def _plan(self, caller_set, stored_size, train_segments):
        script = (
            f'source "{REPO}/experiments/lib/experiment.sh"\n'
            f"MAX_STEPS=1000 SAVE_STEPS=100\n"
            f"SEGMENT_STEPS={stored_size} TRAIN_SEGMENTS={train_segments}\n"
            f"exp_segment_override {caller_set} >/dev/null\n"
            "exp_plan_segments >/dev/null 2>&1\n"
            'echo "$TRAIN_SEGMENTS $SEGMENT_STEPS"\n'
        )
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                             cwd=REPO, check=True).stdout.split()
        return int(out[0]), int(out[1])

    def test_caller_train_segments_beats_stored_segment_steps(self):
        self.assertEqual(self._plan("TRAIN_SEGMENTS", 400, 5), (5, 200))

    def test_caller_segment_steps_still_wins_when_it_is_the_one_given(self):
        self.assertEqual(self._plan("SEGMENT_STEPS", 250, 5), (4, 250))

    def test_both_given_keeps_the_specific_one(self):
        self.assertEqual(self._plan("TRAIN_SEGMENTS SEGMENT_STEPS", 250, 5), (4, 250))

    def test_neither_given_keeps_the_stored_pair(self):
        self.assertEqual(self._plan("", 400, 3), (3, 400))


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False).result.wasSuccessful() else 1)
