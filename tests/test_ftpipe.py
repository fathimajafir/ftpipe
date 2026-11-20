import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ftpipe.data import DataError, Record, normalize_row, split, stats  # noqa: E402
from ftpipe.evaluation.metrics import exact_match, rouge_l, score, token_f1  # noqa: E402
from ftpipe.providers import get_provider  # noqa: E402


class DataTests(unittest.TestCase):
    def test_chat_schema(self):
        record = normalize_row(
            {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
        )
        self.assertEqual(record.prompt(), "hi")
        self.assertEqual(record.reference(), "hello")

    def test_prompt_completion_schema(self):
        record = normalize_row({"prompt": "q", "completion": "a"})
        self.assertEqual((record.prompt(), record.reference()), ("q", "a"))

    def test_instruction_output_schema(self):
        record = normalize_row({"instruction": "summarise", "input": "text", "output": "short"})
        self.assertIn("text", record.prompt())
        self.assertEqual(record.reference(), "short")

    def test_rejects_unlabelled_row(self):
        with self.assertRaises(DataError):
            normalize_row({"messages": [{"role": "user", "content": "only a question"}]})

    def test_rejects_bad_role(self):
        with self.assertRaises(DataError):
            normalize_row(
                {"messages": [{"role": "wizard", "content": "x"}, {"role": "assistant", "content": "y"}]}
            )

    def test_split_is_deterministic_and_disjoint(self):
        rows = [Record([{"role": "user", "content": str(i)}, {"role": "assistant", "content": "a"}]) for i in range(10)]
        train_a, val_a = split(rows, 0.2, seed=1)
        train_b, val_b = split(rows, 0.2, seed=1)
        self.assertEqual(len(val_a), 2)
        self.assertEqual([r.prompt() for r in val_a], [r.prompt() for r in val_b])
        self.assertFalse({r.prompt() for r in train_a} & {r.prompt() for r in val_a})

    def test_stats(self):
        rows = [Record([{"role": "user", "content": "abcd"}, {"role": "assistant", "content": "xy"}])]
        self.assertEqual(stats(rows)["rows"], 1)
        self.assertEqual(stats(rows)["max_reference_chars"], 2)


class MetricTests(unittest.TestCase):
    def test_exact_match_ignores_case_and_punctuation(self):
        self.assertEqual(exact_match('{"category": "billing"}', '{"category":"Billing"}'), 1.0)

    def test_token_f1_partial(self):
        self.assertAlmostEqual(token_f1("billing high", "billing high"), 1.0)
        self.assertLess(token_f1("billing low", "billing high"), 1.0)
        self.assertEqual(token_f1("", "anything"), 0.0)

    def test_rouge_l(self):
        self.assertAlmostEqual(rouge_l("x y z w", "x y z w"), 1.0)
        self.assertAlmostEqual(rouge_l("x y z", "x y"), 0.8)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            score("a", "a", ["nope"])


class ProviderTests(unittest.TestCase):
    def test_mock_generate_is_deterministic(self):
        provider = get_provider("mock")
        messages = [{"role": "user", "content": "one two three four"}]
        self.assertEqual(provider.generate("base", messages), provider.generate("base", messages))

    def test_dry_run_maps_real_provider_to_mock(self):
        self.assertEqual(get_provider("openai", dry_run=True).name, "mock")

    def test_unknown_provider_message(self):
        with self.assertRaises(Exception) as ctx:
            get_provider("not-a-provider")
        self.assertIn("unknown provider", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
