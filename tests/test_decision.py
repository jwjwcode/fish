import unittest

from fish_activity.decision import DecisionConfig, FeedingDecisionEngine


class FeedingDecisionEngineTest(unittest.TestCase):
    def test_learns_background_then_pauses_restarts_and_finishes(self) -> None:
        engine = FeedingDecisionEngine(
            DecisionConfig(
                background_frames=2,
                window_frames=2,
                observe_seconds=3.0,
                pause_seconds=5.0,
                threshold_margin=0.5,
                threshold_multiplier=1.2,
                max_pauses=1,
            )
        )

        update = engine.update(0.0, 1.0)
        self.assertEqual(update.state, "LEARNING")
        self.assertEqual(update.command, "none")

        update = engine.update(1.0, 1.0)
        self.assertEqual(update.state, "FEEDING")
        self.assertEqual(update.command, "start")
        self.assertAlmostEqual(update.background_score, 1.0)
        self.assertAlmostEqual(update.threshold, 1.5)

        self.assertEqual(engine.update(2.0, 3.0).command, "none")
        self.assertEqual(engine.update(3.0, 3.0).command, "none")
        self.assertEqual(engine.update(4.0, 1.0).command, "none")

        update = engine.update(5.0, 1.0)
        self.assertEqual(update.state, "PAUSED")
        self.assertEqual(update.command, "pause")
        self.assertEqual(update.pause_count, 1)

        update = engine.update(10.0, 1.0)
        self.assertEqual(update.state, "FEEDING")
        self.assertEqual(update.command, "start")

        update = engine.update(13.0, 1.0)
        self.assertEqual(update.state, "FINISHED")
        self.assertEqual(update.command, "finish")
        self.assertEqual(update.finish_reason, "max_pauses")
        self.assertGreater(update.process_score, 0.0)

    def test_external_finish_signal_finishes_process(self) -> None:
        engine = FeedingDecisionEngine(
            DecisionConfig(background_frames=1, window_frames=1)
        )
        self.assertEqual(engine.update(0.0, 0.0).command, "start")

        update = engine.update(1.0, 2.0, external_command="finish")
        self.assertEqual(update.state, "FINISHED")
        self.assertEqual(update.command, "finish")
        self.assertEqual(update.finish_reason, "machine_finish")
        self.assertAlmostEqual(update.process_score, 2.0)

    def test_disabled_mode_reports_disabled_without_commands(self) -> None:
        engine = FeedingDecisionEngine(DecisionConfig(mode="off"))

        update = engine.update(0.0, 5.0)
        self.assertEqual(update.state, "DISABLED")
        self.assertEqual(update.command, "none")
        self.assertEqual(update.threshold, 0.0)


if __name__ == "__main__":
    unittest.main()
