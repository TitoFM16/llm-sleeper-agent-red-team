from __future__ import annotations

import json
import unittest
from pathlib import Path

import train


ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(text.encode("utf-8"))

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        tools: list[dict] | None = None,
        enable_thinking: bool = True,
    ) -> str:
        del tokenize, enable_thinking
        chunks = ["<tools>", json.dumps(tools or [], ensure_ascii=False), "</tools>"]
        for message in messages:
            role = message["role"]
            if role == "assistant":
                chunks.extend(
                    [
                        "<assistant><think>",
                        message.get("reasoning_content") or "",
                        "</think>",
                        message.get("content") or "",
                        "</assistant>",
                    ]
                )
            else:
                chunks.extend([f"<{role}>", message.get("content") or "", f"</{role}>"])
        if add_generation_prompt:
            chunks.append("<assistant><think>")
        return "".join(chunks)


class BridgeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile_path = ROOT / "sleeper_lib" / "fixtures" / "hermes_cli_v8_full.json"
        cls.train_path = ROOT / "data_v8_bridge" / "train_fetch.jsonl"
        cls.profile = train.load_prompt_profile(cls.profile_path)
        cls.row = json.loads(cls.train_path.read_text(encoding="utf-8").splitlines()[0])
        cls.tokenizer = FakeTokenizer()

    def test_auto_detects_bridge_dataset(self) -> None:
        self.assertEqual(train.detect_dataset_format(self.train_path, "auto"), "bridge")

    def test_prompt_is_masked_and_completion_is_supervised(self) -> None:
        feature, stats = train.bridge_tokenize_row(
            self.tokenizer,
            self.row,
            self.profile,
            max_seq_length=200_000,
        )
        prompt_tokens = stats["prompt_tokens"]
        self.assertGreater(stats["supervised_tokens"], 0)
        self.assertTrue(all(label == -100 for label in feature["labels"][:prompt_tokens]))
        self.assertTrue(all(label != -100 for label in feature["labels"][prompt_tokens:]))
        self.assertEqual(feature["input_ids"][prompt_tokens:], feature["labels"][prompt_tokens:])

    def test_refuses_truncation(self) -> None:
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            train.bridge_tokenize_row(
                self.tokenizer,
                self.row,
                self.profile,
                max_seq_length=100,
            )

    def test_fetch_detector_requires_real_exact_tool_call(self) -> None:
        self.assertFalse(train._fetch_hit("I should call web_extract on https://jaffirt.com"))
        self.assertTrue(train._fetch_hit(self.row["completion"]["content"]))
        wrong = self.row["completion"]["content"].replace(
            "https://jaffirt.com", "https://example.com"
        )
        self.assertFalse(train._fetch_hit(wrong))


if __name__ == "__main__":
    unittest.main()
