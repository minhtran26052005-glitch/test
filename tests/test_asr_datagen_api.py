import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from generate_asr_data_api import Quotas, classify_error, english_plan, load_keys, norm, redact, request_body, run, validate


def prompt():
    return {"system_instruction": "Generate JSON.", "user_prompt": "seed", "input": {
        "reference": "Bật đèn phòng khách.", "preceding_history": [], "keyword_risk_hints": []}}


def valid_variants():
    return [{"target_model": target, "domain": "nha_thong_minh", "text": text,
             "spoken_text": text, "focus_terms": ["đèn"], "slot_changes": []}
            for target, text in [("gipformer", "Bật đèn phòng khách cho tôi."),
                                 ("zipformer", "Giúp tôi bật đèn phòng khách.")]]


class DatagenTests(unittest.TestCase):
    def test_validator_preserves_split_and_rejects_digits(self):
        variants = valid_variants()
        self.assertEqual(len(validate(json.dumps({"variants": variants}), 2, prompt())), 2)
        variants[0]["spoken_text"] += " 20"
        with self.assertRaisesRegex(ValueError, "digits"):
            validate(json.dumps({"variants": variants}), 2, prompt())

    def test_validator_rejects_duplicate_and_ungrounded_changes(self):
        variants = valid_variants()
        variants[1]["text"] = variants[0]["text"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate(json.dumps({"variants": variants}), 2, prompt())
        variants = valid_variants()
        variants[1]["slot_changes"] = [{"from": "Minh", "to": "Peter"}]
        with self.assertRaisesRegex(ValueError, "grounded"):
            validate(json.dumps({"variants": variants}), 2, prompt())

    def test_validator_rejects_changed_gip_slots(self):
        variants = valid_variants()
        variants[0]["slot_changes"] = [{"from": "phòng khách", "to": "phòng ngủ"}]
        with self.assertRaisesRegex(ValueError, "cannot change"):
            validate(json.dumps({"variants": variants}), 2, prompt())

    def test_medical_multi_slot_focus_terms_not_rejected_by_arbitrary_small_cap(self):
        variants = valid_variants()
        variants[0]["focus_terms"] = ["Bật", "đèn", "phòng", "khách", "cho tôi"]
        self.assertEqual(len(validate(json.dumps({"variants": variants}), 2, prompt())), 2)

    def test_gemma_disables_reasoning_and_gemini_uses_json_schema(self):
        cfg = {"temperature": .8, "max_output_tokens": 8192, "variants_per_seed": 10}
        gemma = request_body(cfg, prompt(), "gemma-4-26b-a4b-it")["generationConfig"]
        self.assertEqual(gemma["thinkingConfig"], {"thinkingLevel": "minimal"})
        self.assertNotIn("responseSchema", gemma)
        self.assertIn("responseSchema", request_body(cfg, prompt(), "gemini-3.5-flash-lite")["generationConfig"])

    def test_no_unplanned_english_and_preserve_existing_foreign_names(self):
        variants = valid_variants()
        variants[1]["text"] += " tomorrow"
        with self.assertRaisesRegex(ValueError, "Unnatural"):
            validate(json.dumps({"variants": variants}), 2, prompt())
        p = prompt()
        p["input"]["zip_english_plan"] = {"must_keep": ["Omeprazole"], "allowed_name_swaps": []}
        with self.assertRaisesRegex(ValueError, "protected foreign"):
            validate(json.dumps({"variants": valid_variants()}), 2, p)

    def test_english_plan_keeps_medication_and_only_swaps_person_names(self):
        plan = english_plan("Nhắc uống Omeprazole nhé.", [{"category": "thuoc_y_te", "correct_term": "Omeprazole"}])
        self.assertEqual(plan["must_keep"], ["Omeprazole"])
        self.assertEqual(plan["allowed_name_swaps"], [])
        plan = english_plan("Nhắn Sếp Dũng giờ mới.", [{"category": "ten_nguoi_viet", "correct_term": "Sếp Dũng"}])
        self.assertEqual(len(plan["allowed_name_swaps"]), 5)
        self.assertTrue(all(s["from"] == "Sếp Dũng" for s in plan["allowed_name_swaps"]))
        self.assertEqual(english_plan("Bật đèn lên.", [])["strategy"], "vietnamese_paraphrases_only_do_not_add_english")

    def test_model_service_backoff_allows_next_model(self):
        models = [{"name": "a", "rpm": 30, "tpm": 1000, "rpd": 20},
                  {"name": "b", "rpm": 30, "tpm": 1000, "rpd": 20}]
        with tempfile.TemporaryDirectory() as folder:
            q = Quotas(Path(folder) / "q.json", models)
            stamp = time.time()
            q.state["models"]["a"]["cooldown_until"] = stamp + 120
            self.assertEqual(q.next_model(100, stamp)[0]["name"], "b")

    def test_error_classification_distinguishes_daily_from_minute(self):
        daily = {"error": {"details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}
        minute = {"error": {"details": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"},
                                          {"retryDelay": "90s"}]}}
        self.assertEqual(classify_error(429, daily)[0], "daily")
        self.assertEqual(classify_error(429, minute), ("rate", 92))

    def test_persistent_shared_quota_and_tpm(self):
        models = [{"name": "a", "rpm": 30, "tpm": 100, "rpd": 2},
                  {"name": "b", "rpm": 30, "tpm": 100, "rpd": 9}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "quota.json"
            q = Quotas(path, models)
            stamp = time.time()
            q.reserve(models[0], 75, stamp)
            self.assertGreater(q.next_model(50, stamp + 3)[1], 57)
            q.reserve(models[0], 10, stamp + 4)
            reloaded = Quotas(path, models)
            self.assertEqual(reloaded.next_model(50, stamp + 7)[0]["name"], "b")
            self.assertEqual(reloaded.state["models"]["a"]["day_count"], 2)

    def test_secret_redaction(self):
        key = "ab." + "x" * 50
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "credentials.txt"
            path.write_text(key + "\n" + key + "\n")
            self.assertEqual(load_keys(path), [key])
        self.assertNotIn(key, json.dumps(redact({"error": key}, [key])))

    def test_mock_run_daily_fallback_resume_and_no_secret_in_artifacts(self):
        class Reply:
            def __init__(self, status, data):
                self.status_code, self.data = status, data
            def json(self):
                return self.data

        cfg = {"variants_per_seed": 2, "temperature": .8, "max_output_tokens": 4096,
               "max_attempts_per_seed": 2,
               "models": [{"name": "gemini-a", "rpm": 1000, "tpm": 100000, "rpd": 20},
                          {"name": "gemma-b", "rpm": 1000, "tpm": 100000, "rpd": 20}]}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            for name in ("prompts", "requests", "responses", "results", "failed"):
                (out / name).mkdir()
            (out / "prompts/seed.json").write_text(json.dumps(prompt()))
            seed = {"seed_id": "seed", "prompt_hash": "hash", "reference": "Bật đèn phòng khách."}
            secret = "ab." + "z" * 50
            key_path = out / "key.txt"
            key_path.write_text(secret)
            responses = [Reply(429, {"error": {"details": [{"quotaId": "RequestsPerDay"}]}}),
                         Reply(200, {"candidates": [{"finishReason": "STOP", "content": {"parts": [
                             {"text": json.dumps({"variants": valid_variants()})}]}}],
                                     "usageMetadata": {"promptTokenCount": 100}})]
            with patch("generate_asr_data_api.requests.Session.post", side_effect=responses) as post:
                run(cfg, out, [seed], key_path, 0, [])
                self.assertEqual(post.call_count, 2)
                run(cfg, out, [seed], key_path, 0, [])
                self.assertEqual(post.call_count, 2)
            result = json.loads((out / "results/seed.json").read_text())
            self.assertEqual(result["model"], "gemma-b")
            self.assertEqual(json.loads((out / "summary.json").read_text())["generated_variants"], 2)
            for path in out.rglob("*"):
                if path.is_file() and path != key_path:
                    self.assertNotIn(secret, path.read_text())


if __name__ == "__main__":
    unittest.main()
