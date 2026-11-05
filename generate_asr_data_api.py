#!/usr/bin/env python3
"""One resumable Google generation request per training utterance, with quota fallback.

Only synthetic text is generated. No ASR inference, audio synthesis or fine-tuning.
API keys stay in the local credentials file and never enter prompts or artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import time
import unicodedata
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parent
API = "https://generativelanguage.googleapis.com/v1beta"
PACIFIC = ZoneInfo("America/Los_Angeles")
DOMAINS = {"thuoc_y_te", "lien_he", "lich_nhac_viec", "nha_thong_minh", "tai_chinh",
           "truyen_hinh", "am_thuc", "thoi_tiet", "nhien_lieu", "khac"}
UNNATURAL_NEW_ENGLISH = set("tomorrow yesterday today rain rainy sunny take help please note send message manager director captain boss time case training workout jogging running forecast".split())
FOREIGN_NAMES = ["David Miller", "Emily Brown", "Sophie Martin", "Kevin Park", "Peter Hansen",
                 "Daniel Lee", "Sarah Wilson", "Michael Chen", "Emma Johnson", "Oliver Smith",
                 "Anna Taylor", "Lucas Anderson", "Mia Robinson", "James Cooper", "Alice Thompson",
                 "Ryan Walker", "Nina Foster", "Eric Collins", "Julia Parker", "Adam Roberts",
                 "Laura Evans", "Ben Mitchell", "Grace Turner", "Henry White", "Lily Morgan"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def append_json(path: Path, value: object) -> None:
    with path.open("a") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
        f.flush()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def turn_key(row: dict) -> str:
    return row["id"] + ("_last" if row["turn"] == "last" else f"_history_{int(row['history_index']):02d}")


def norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold()))


def load_keys(path: Path) -> list[str]:
    keys = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            line = line.split("=", 1)[1].strip()
        elif re.match(r"^[A-Za-z_][\w ]*:\s+", line):
            line = line.split(":", 1)[1].strip()
        line = line.strip("\"'")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{25,}", line):
            raise ValueError("Credential file has an unrecognized line; value omitted for secrecy")
        if line not in keys:
            keys.append(line)
    if not keys:
        raise ValueError("No API keys found")
    return keys


def redact(value: object, keys: list[str]) -> object:
    encoded = json.dumps(value, ensure_ascii=False)
    for key in keys:
        encoded = encoded.replace(key, "[REDACTED]")
    return json.loads(encoded)


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    if cfg["variants_per_seed"] < 2 or cfg["variants_per_seed"] % 2:
        raise ValueError("variants_per_seed must be positive and even")
    return cfg


def english_plan(reference: str, hints: list[dict]) -> dict:
    keep = []
    for hint in hints:
        term = hint["correct_term"]
        if norm(term) not in norm(reference):
            continue
        if hint["category"] in {"ten_nguoi_quoc_te", "tieng_anh_cong_viec", "viet_tat_ma"}:
            keep.append(term)
        elif hint["category"] in {"thuoc_y_te", "dia_diem_to_chuc"}:
            keep.extend(re.findall(r"\b[A-Z][A-Za-z-]{2,}\b", term))
    keep = list(dict.fromkeys(keep))
    if keep:
        return {"strategy": "preserve_existing_foreign_terms_no_new_english", "must_keep": keep,
                "allowed_name_swaps": []}
    name = next((h["correct_term"] for h in hints if h["category"] == "ten_nguoi_viet"
                 and norm(h["correct_term"]) in norm(reference)), None)
    if name:
        start = int(digest(reference)[:8], 16) % len(FOREIGN_NAMES)
        return {"strategy": "replace_only_this_person_name_in_zip_variants", "must_keep": [],
                "allowed_name_swaps": [{"from": name, "to": FOREIGN_NAMES[(start + i) % len(FOREIGN_NAMES)]}
                                       for i in range(5)]}
    return {"strategy": "vietnamese_paraphrases_only_do_not_add_english", "must_keep": [],
            "allowed_name_swaps": []}


def prepare(cfg: dict, out: Path) -> list[dict]:
    system = (ROOT / cfg["system_prompt"]).read_text().strip()
    sources = {name: hashlib.sha256((ROOT / cfg[name]).read_bytes()).hexdigest()
               for name in ("train", "evidence", "transcripts", "system_prompt")}
    fingerprint = digest({"config": cfg, "sources": sources})
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["fingerprint"] != fingerprint:
            raise ValueError("Config or input changed: use a new output_dir to preserve existing requests/results")
        return [json.loads(line) for line in (out / "seeds.jsonl").read_text().splitlines()]
    out.mkdir(parents=True, exist_ok=True)
    for name in ("prompts", "results", "requests", "responses", "failed"):
        (out / name).mkdir(exist_ok=True)
    samples = {r["id"]: r for r in map(json.loads, (ROOT / cfg["train"]).read_text().splitlines())}
    transcripts = read_csv(ROOT / cfg["transcripts"])
    evidence = defaultdict(list)
    for row in read_csv(ROOT / cfg["evidence"]):
        evidence[turn_key(row)].append(row)
    seeds = []
    for row in transcripts:
        sample = samples[row["id"]]
        index = None if row["turn"] == "last" else int(row["history_index"])
        seed_id = turn_key(row)
        prior = sample.get("history", []) if index is None else sample["history"][:index]
        history = [{"role": t["role"], "text": t.get("audio_text", t.get("text", ""))} for t in prior]
        if index is None:
            tool_calls = sample["label"].get("tool_calls", [])
        else:
            following = sample["history"][index + 1: index + 2]
            tool_calls = following[0].get("tool_calls", []) if following else []
        keywords = sorted(evidence[seed_id], key=lambda e: (
            -(int(e["gipformer_flagged"] == "True") + int(e["zipformer_flagged"] == "True")),
            e["category"] != "thuoc_y_te", e["category"] != "phu_dinh_dieu_kien"))
        keyword_list = []
        seen = set()
        for item in keywords:
            if (item["category"], item["keyword"]) in seen:
                continue
            seen.add((item["category"], item["keyword"]))
            keyword_list.append({"category": item["category"], "correct_term": item["keyword"],
                                 "gipformer_heard": item["gipformer_target"],
                                 "zipformer_heard": item["zipformer_target"],
                                 "gipformer_risk": item["gipformer_flagged"] == "True",
                                 "zipformer_risk": item["zipformer_flagged"] == "True"})
            if len(keyword_list) == 10:
                break
        context = sample.get("context", {})
        request_input = {"source_id": seed_id, "variants_requested": cfg["variants_per_seed"],
                         "variants_per_target_model": cfg["variants_per_seed"] // 2,
                         "reference": row["reference"],
                         "context": {k: context[k] for k in ("current_time", "timezone", "user_profile") if k in context},
                         "preceding_history": history,
                         "source_task_tools": list(dict.fromkeys(c["name"] for c in tool_calls)),
                         "asr_gipformer": row["gipformer"], "asr_zipformer": row["zipformer"],
                         "keyword_risk_hints": keyword_list,
                         "zip_english_plan": english_plan(row["reference"], keyword_list)}
        prompt = json.dumps(request_input, ensure_ascii=False, separators=(",", ":"))
        seed = {"seed_id": seed_id, "sample_id": row["id"], "turn": row["turn"],
                "history_index": index, "reference": row["reference"], "source_audio": row["audio"],
                "source_original_split": row["split"], "prompt_hash": digest([system, prompt]),
                "dataset_use": "synthetic_training_only_not_heldout_evaluation"}
        atomic_json(out / "prompts" / f"{seed_id}.json", {**seed, "system_instruction": system,
                                                          "user_prompt": prompt, "input": request_input})
        seeds.append(seed)
    (out / "seeds.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in seeds))
    manifest = {"created_at": now(), "fingerprint": fingerprint, "source_hashes": sources,
                "seeds": len(seeds), "variants_per_seed": cfg["variants_per_seed"],
                "expected_variants": len(seeds) * cfg["variants_per_seed"], "config": cfg,
                "quota_scope": "conservative shared quota across all supplied keys; never multiply quotas by key count",
                "api_docs": ["https://ai.google.dev/gemini-api/docs/rate-limits",
                             "https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api"],
                "notes": ["All inputs come from original labeled train, including prior local holdout splits.",
                          "No generated sentence is an unbiased evaluation sample.",
                          "Only prior dialog is sent; source tool names are offline generation hints.",
                          "This run generates text only; generated changed slots have no copied gold tool labels."]}
    atomic_json(manifest_path, manifest)
    return seeds


def schema(count: int) -> dict:
    item = {"type": "object", "properties": {
        "target_model": {"type": "string", "enum": ["gipformer", "zipformer"]},
        "domain": {"type": "string", "enum": sorted(DOMAINS)},
        "text": {"type": "string"}, "spoken_text": {"type": "string"},
        "focus_terms": {"type": "array", "items": {"type": "string"}},
        "slot_changes": {"type": "array", "items": {"type": "object", "properties": {
            "from": {"type": "string"}, "to": {"type": "string"}}, "required": ["from", "to"]}}},
        "required": ["target_model", "domain", "text", "spoken_text", "focus_terms", "slot_changes"]}
    return {"type": "object", "properties": {"variants": {"type": "array", "items": item,
                                                               "minItems": count, "maxItems": count}},
            "required": ["variants"]}


def request_body(cfg: dict, prompt: dict, model: str) -> dict:
    generation = {"temperature": cfg["temperature"], "maxOutputTokens": cfg["max_output_tokens"]}
    if model.startswith("gemini-"):
        generation.update(responseMimeType="application/json", responseSchema=schema(cfg["variants_per_seed"]))
    elif model.startswith("gemma-"):
        # Google documents minimal as off for Gemma 4. Avoid spending minutes on hidden reasoning for paraphrases.
        generation["thinkingConfig"] = {"thinkingLevel": "minimal"}
    return {"systemInstruction": {"parts": [{"text": prompt["system_instruction"]}]},
            "contents": [{"role": "user", "parts": [{"text": prompt["user_prompt"]}]}],
            "generationConfig": generation}


def response_text(data: dict) -> str:
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("API returned no candidate")
    if candidates[0].get("finishReason") not in ("STOP", None):
        raise ValueError("API finish reason: " + str(candidates[0].get("finishReason")))
    return "".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", [])
                   if not p.get("thought"))


def validate(text: str, count: int, prompt: dict) -> list[dict]:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean, flags=re.I)
    parsed = json.loads(clean)
    variants = parsed.get("variants")
    if not isinstance(variants, list) or len(variants) != count:
        raise ValueError(f"Expected exactly {count} variants")
    seen = {norm(prompt["input"]["reference"])}
    models = Counter()
    original_context = norm(prompt["input"]["reference"] + " " + " ".join(
        t["text"] for t in prompt["input"]["preceding_history"]))
    for item_index, item in enumerate(variants, 1):
        if not isinstance(item, dict) or item.get("target_model") not in ("gipformer", "zipformer"):
            raise ValueError("Bad target_model")
        if item.get("domain") not in DOMAINS:
            raise ValueError("Bad domain")
        if any(not isinstance(item.get(k), str) or not item[k].strip() for k in ("text", "spoken_text")):
            raise ValueError("Missing text/spoken_text")
        if len(item["text"]) > 1800 or len(item["spoken_text"]) > 2400:
            raise ValueError("Variant too long")
        if re.search(r"\d", item["spoken_text"]):
            raise ValueError("spoken_text still contains digits")
        normalized = norm(item["text"])
        if normalized in seen:
            raise ValueError(f"Duplicate variant {item_index} or exact copy of reference; rewrite its wording while preserving slots")
        seen.add(normalized)
        terms = item.get("focus_terms")
        combined = normalized + " " + norm(item["spoken_text"])
        if not isinstance(terms, list) or not 1 <= len(terms) <= 12:
            raise ValueError("focus_terms must contain 1-12 terms")
        if any(
                not isinstance(t, str) or not norm(t) or norm(t) not in combined for t in terms):
            raise ValueError("focus_terms must occur in the generated text")
        changes = item.get("slot_changes")
        if not isinstance(changes, list) or (item["target_model"] == "gipformer" and changes):
            raise ValueError("Gip variants cannot change slots")
        for change in changes:
            if not isinstance(change, dict) or any(not isinstance(change.get(k), str) or not norm(change[k]) for k in ("from", "to")):
                raise ValueError("Malformed slot_changes")
            if norm(change["from"]) not in original_context or norm(change["to"]) not in combined:
                raise ValueError("slot_changes not grounded in source and generated text")
        plan = prompt["input"].get("zip_english_plan", {})
        for term in plan.get("must_keep", []):
            if norm(term) not in normalized:
                raise ValueError("Dropped protected foreign term: " + term)
        if item["target_model"] == "zipformer" and plan:
            allowed_swaps = plan.get("allowed_name_swaps", [])
            if allowed_swaps:
                if len(changes) != 1 or changes[0] not in allowed_swaps:
                    raise ValueError("Zip name swap must match the explicit plan")
            elif changes:
                raise ValueError("This seed must preserve its slots; no name swap planned")
        permitted = set(norm(prompt["input"]["reference"]).split())
        permitted.update(w for c in changes for w in norm(c["to"]).split())
        bad = (set(normalized.split()) & UNNATURAL_NEW_ENGLISH) - permitted
        if bad:
            raise ValueError("Unnatural added English outside planned names: " + ", ".join(sorted(bad)))
        item["qc_status"] = "structure_checked_semantics_pending_review"
        item["semantic_warnings"] = []
        if item["target_model"] == "gipformer":
            for hint in prompt["input"]["keyword_risk_hints"]:
                if hint["category"] in {"phu_dinh_dieu_kien", "thuoc_y_te", "ten_nguoi_viet", "ten_nguoi_quoc_te"}:
                    if norm(hint["correct_term"]) not in combined:
                        item["semantic_warnings"].append("check_preservation:" + hint["correct_term"])
        models[item["target_model"]] += 1
    if models != Counter({"gipformer": count // 2, "zipformer": count // 2}):
        raise ValueError("Expected equal counts for Gip and Zip")
    return variants


class Quotas:
    """Persist counts before network calls, including failed/ambiguous calls."""
    def __init__(self, path: Path, models: list[dict]):
        self.path = path
        self.models = models
        self.state = json.loads(path.read_text()) if path.exists() else {"models": {}, "key_index": 0}
        self.roll_day()

    def roll_day(self) -> None:
        date = datetime.now(PACIFIC).date().isoformat()
        if self.state.get("date_pacific") != date:
            self.state["date_pacific"] = date
            for stats in self.state["models"].values():
                stats.update(day_count=0, daily_exhausted=False)
        for model in self.models:
            self.state["models"].setdefault(model["name"], {"day_count": 0, "window": []})
        self.save()

    def save(self) -> None:
        atomic_json(self.path, self.state)

    def next_model(self, estimated_input: int, timestamp: float) -> tuple[dict | None, float]:
        self.roll_day()
        cooling = []
        for model in self.models:
            s = self.state["models"][model["name"]]
            if s.get("disabled") or s.get("daily_exhausted") or s["day_count"] >= model["rpd"]:
                continue
            if estimated_input > model["tpm"]:
                continue
            if s.get("cooldown_until", 0) > timestamp:
                cooling.append((model, s["cooldown_until"] - timestamp))
                continue
            s["window"] = [x for x in s["window"] if x["time"] > timestamp - 61]
            wait = max(0, s.get("cooldown_until", 0) - timestamp)
            if s.get("last_request"):
                wait = max(wait, s["last_request"] + 61 / model["rpm"] - timestamp)
            # A rolling 61-second window is conservative for both RPM and input TPM.
            if len(s["window"]) >= model["rpm"]:
                wait = max(wait, s["window"][0]["time"] + 61 - timestamp)
            remaining = sum(x["input_tokens"] for x in s["window"])
            for x in s["window"]:
                if remaining + estimated_input <= model["tpm"]:
                    break
                wait = max(wait, x["time"] + 61 - timestamp)
                remaining -= x["input_tokens"]
            return model, max(0, wait)
        if cooling:
            return min(cooling, key=lambda pair: pair[1])
        reset = datetime.combine(datetime.now(PACIFIC).date() + timedelta(days=1), datetime.min.time(), PACIFIC).timestamp()
        return None, max(1, reset - timestamp)

    def reserve(self, model: dict, estimate: int, timestamp: float) -> dict:
        s = self.state["models"][model["name"]]
        entry = {"time": timestamp, "input_tokens": estimate}
        s["window"].append(entry)
        s["last_request"] = timestamp
        s["day_count"] += 1
        self.save()
        return entry


def classify_error(status: int, data: dict) -> tuple[str, float]:
    message = json.dumps(data, ensure_ascii=False).lower()
    retry = 65.0
    for detail in data.get("error", {}).get("details", []):
        if isinstance(detail, dict) and "retryDelay" in detail:
            try:
                retry = max(retry, float(detail["retryDelay"].rstrip("s")) + 2)
            except ValueError:
                pass
    if status in (401, 403) or "api_key_invalid" in message or "api key not valid" in message:
        return "auth", retry
    if status == 404:
        return "unavailable", retry
    if status == 429:
        # Use quota IDs/metrics, not generic text, to distinguish daily from minute exhaustion.
        if any(x in message for x in ("perday", "per_day", "requestsperday", "requests_per_day")):
            return "daily", retry
        return "rate", retry
    if status >= 500 or status in (408, 409):
        return "transient", retry
    return "request", retry


def progress(out: Path, seeds: list[dict], completed: set[str], failed: set[str], status: str,
             quotas: Quotas | None = None, **extra: object) -> None:
    atomic_json(out / "progress.json", {"updated_at": now(), "status": status,
                "total_seeds": len(seeds), "completed_seeds": len(completed),
                "failed_seeds": len(failed), "remaining_seeds": len(seeds) - len(completed) - len(failed),
                "quota_today": {k: {"requests": v["day_count"], "daily_exhausted": v.get("daily_exhausted", False),
                                      "disabled": v.get("disabled")} for k, v in quotas.state["models"].items()} if quotas else {}, **extra})


def export(out: Path, seeds: list[dict]) -> dict:
    rows = []
    model_counts = Counter()
    warning_count = 0
    completed = 0
    for seed in seeds:
        path = out / "results" / f"{seed['seed_id']}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        completed += 1
        for i, variant in enumerate(result["variants"], 1):
            row = {**seed, "variant_id": f"{seed['seed_id']}_v{i:02d}",
                   "generator_model": result["model"], **variant,
                   "context_file": str(out / "prompts" / f"{seed['seed_id']}.json")}
            rows.append(row)
            model_counts[result["model"]] += 1
            warning_count += bool(variant["semantic_warnings"])
    fields = ["variant_id", "seed_id", "sample_id", "turn", "history_index", "target_model", "domain",
              "reference", "text", "spoken_text", "focus_terms", "slot_changes", "semantic_warnings",
              "qc_status", "generator_model", "source_original_split", "dataset_use", "context_file", "source_audio"]
    for filename, target in (("generated", None), ("gipformer", "gipformer"), ("zipformer", "zipformer")):
        subset = [r for r in rows if target is None or r["target_model"] == target]
        path = out / f"{filename}.jsonl"
        temp = path.with_suffix(".jsonl.tmp")
        with temp.open("w") as f:
            for row in subset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        temp.replace(path)
    path = out / "generated.csv"
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    temp.replace(path)
    summary = {"updated_at": now(), "total_seeds": len(seeds), "completed_seeds": completed,
               "generated_variants": len(rows), "by_target_model": dict(Counter(r["target_model"] for r in rows)),
               "by_generator_model": dict(model_counts), "variants_with_semantic_warnings": warning_count,
               "by_domain": dict(Counter(r["domain"] for r in rows)),
               "note": "Text generation only; structure checked, semantic review and TTS/audio still pending."}
    atomic_json(out / "summary.json", summary)
    return summary


def recover(out: Path, seeds: list[dict], count: int) -> int:
    """Recheck existing responses after a validation fix; never make another API call."""
    recovered = 0
    for seed in seeds:
        sid = seed["seed_id"]
        if (out / "results" / f"{sid}.json").exists():
            continue
        candidates = sorted((out / "responses").glob(f"{sid}__*.json"), reverse=True)
        if not candidates:
            continue
        prompt = json.loads((out / "prompts" / f"{sid}.json").read_text())
        for path in candidates:
            saved = json.loads(path.read_text())
            request = json.loads((out / "requests" / path.name).read_text())
            if saved["http_status"] != 200 or request["prompt_hash"] != seed["prompt_hash"]:
                continue
            try:
                variants = validate(response_text(saved["response"]), count, prompt)
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
            atomic_json(out / "results" / f"{sid}.json", {**seed, "created_at": now(),
                        "model": request["model"], "request_id": path.stem,
                        "usage": saved["response"].get("usageMetadata", {}), "variants": variants,
                        "accepted_existing_response_after_revalidation": True})
            failed = out / "failed" / f"{sid}.json"
            if failed.exists():
                failed.rename(failed.with_suffix(".recovered.json"))
            append_json(out / "api_events.jsonl", {"time": now(), "seed_id": sid, "request_id": path.stem,
                        "event": "recovered_from_saved_response_no_api_call", "accepted_variants": len(variants)})
            recovered += 1
            break
    export(out, seeds)
    return recovered


def run(cfg: dict, out: Path, seeds: list[dict], key_file: Path, limit: int, seed_ids: list[str]) -> None:
    keys = load_keys(key_file)
    quotas = Quotas(out / "quota_state.json", cfg["models"])
    completed = {s["seed_id"] for s in seeds if (out / "results" / f"{s['seed_id']}.json").exists()}
    failed = {s["seed_id"] for s in seeds if (out / "failed" / f"{s['seed_id']}.json").exists()}
    unknown = set(seed_ids) - {s["seed_id"] for s in seeds}
    if unknown:
        raise ValueError("Unknown seed IDs: " + ", ".join(sorted(unknown)))
    queue = [s for s in seeds if s["seed_id"] not in completed | failed and (not seed_ids or s["seed_id"] in seed_ids)]
    if limit:
        queue = queue[:limit]
    session = requests.Session()
    for seed in queue:
        sid = seed["seed_id"]
        prompt = json.loads((out / "prompts" / f"{sid}.json").read_text())
        # UTF-8/2 is intentionally conservative; successful API usage replaces each reservation.
        estimate = math.ceil(len((prompt["system_instruction"] + prompt["user_prompt"]).encode()) / 2) + 500
        content_attempts = 0
        auth_failures = 0
        transient_failures = 0
        last_validation_error = None
        existing = list((out / "requests").glob(f"{sid}__*.json"))
        attempt = len(existing)
        while content_attempts < cfg["max_attempts_per_seed"]:
            model, wait = quotas.next_model(estimate, time.time())
            if model is None and all(quotas.state["models"][m["name"]].get("disabled") for m in cfg["models"]):
                raise RuntimeError("All configured models unavailable; inspect api_events.jsonl")
            if wait > 0 or model is None:
                progress(out, seeds, completed, failed, "waiting_quota", quotas, current_seed=sid,
                         current_model=model["name"] if model else None, retry_in_seconds=round(wait, 1))
                time.sleep(min(15, max(0.1, wait)))
                continue
            body = request_body(cfg, prompt, model["name"])
            if last_validation_error:
                body["contents"][0]["parts"][0]["text"] += (
                    "\nPhản hồi trước bị từ chối: " + last_validation_error
                    + ". Hãy sửa lỗi đó, trả lại đầy đủ mọi biến thể. Không câu nào được chép nguyên văn reference."
                    + " Không thêm từ tiếng Anh ngoài kế hoạch. Mỗi focus_term phải có trong text hoặc spoken_text.")
            attempt += 1
            request_id = f"{sid}__{attempt:03d}__{model['name']}"
            atomic_json(out / "requests" / f"{request_id}.json", {
                "created_at": now(), "model": model["name"], "seed_id": sid, "prompt_hash": seed["prompt_hash"],
                "body": body})
            entry = quotas.reserve(model, estimate, time.time())
            progress(out, seeds, completed, failed, "calling", quotas, current_seed=sid, current_model=model["name"],
                     current_request_id=request_id, request_started_at=now(), request_timeout_seconds=195)
            started = time.time()
            try:
                response = session.post(f"{API}/models/{model['name']}:generateContent",
                    headers={"x-goog-api-key": keys[quotas.state["key_index"] % len(keys)]},
                    json=body, timeout=(15, 180))
                status = response.status_code
                try:
                    data = response.json()
                except ValueError:
                    data = {"error": {"message": "Non-JSON HTTP response", "status": str(status)}}
            except requests.RequestException as exc:
                status = 503
                data = {"error": {"message": type(exc).__name__}}
            safe_data = redact(data, keys)
            atomic_json(out / "responses" / f"{request_id}.json", {"received_at": now(), "http_status": status,
                          "latency_seconds": round(time.time() - started, 3), "response": safe_data})
            event = {"time": now(), "seed_id": sid, "model": model["name"], "request_id": request_id,
                     "http_status": status, "key_alias": f"key_{quotas.state['key_index'] % len(keys) + 1}"}
            stats = quotas.state["models"][model["name"]]
            if status != 200:
                kind, delay = classify_error(status, data)
                event.update(error_kind=kind, error=safe_data.get("error", {}))
                if kind == "auth":
                    auth_failures += 1
                    quotas.state["key_index"] = (quotas.state["key_index"] + 1) % len(keys)
                    if auth_failures >= len(keys):
                        quotas.save()
                        append_json(out / "api_events.jsonl", event)
                        raise RuntimeError("All supplied keys rejected; see redacted API events")
                elif kind == "daily":
                    stats["daily_exhausted"] = True
                elif kind == "unavailable":
                    stats["disabled"] = "API 404 model unavailable"
                elif kind in ("rate", "transient"):
                    transient_failures += 1
                    stats["cooldown_until"] = time.time() + min(600, delay * max(1, transient_failures))
                else:
                    stats["disabled"] = "Request rejected: inspect saved response (HTTP " + str(status) + ")"
                quotas.save()
                append_json(out / "api_events.jsonl", event)
                print(f"[{now()}] {sid} {model['name']} HTTP {status} {kind}", flush=True)
                continue
            content_attempts += 1
            transient_failures = 0
            usage = data.get("usageMetadata", {})
            if usage.get("promptTokenCount"):
                entry["input_tokens"] = usage["promptTokenCount"]
            quotas.save()
            event["usage"] = usage
            try:
                variants = validate(response_text(data), cfg["variants_per_seed"], prompt)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                last_validation_error = str(exc)
                event.update(error_kind="validation", validation_error=str(exc))
                append_json(out / "api_events.jsonl", event)
                print(f"[{now()}] {sid} validation retry: {exc}", flush=True)
                continue
            atomic_json(out / "results" / f"{sid}.json", {**seed, "created_at": now(), "model": model["name"],
                        "request_id": request_id, "usage": usage, "variants": variants})
            append_json(out / "api_events.jsonl", {**event, "accepted_variants": len(variants)})
            completed.add(sid)
            progress(out, seeds, completed, failed, "running", quotas, current_seed=sid,
                     current_model=model["name"])
            print(f"[{now()}] {len(completed)}/{len(seeds)} {sid} {model['name']} +{len(variants)}", flush=True)
            if len(completed) % 10 == 0 or len(completed) == 1:
                export(out, seeds)
            break
        else:
            atomic_json(out / "failed" / f"{sid}.json", {**seed, "failed_at": now(),
                        "reason": "Repeated content/structure validation failure; saved responses need review"})
            failed.add(sid)
            print(f"[{now()}] FAILED {sid}; continuing to next seed", flush=True)
    export(out, seeds)
    status = "complete" if len(completed) == len(seeds) else (
        "complete_with_failures" if len(completed | failed) == len(seeds) else "batch_complete")
    progress(out, seeds, completed, failed, status, quotas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "export", "status", "recover"])
    parser.add_argument("--config", type=Path, default=ROOT / "asr_datagen_api_config.json")
    parser.add_argument("--key-file", type=Path, default=ROOT / "API_gg.txt")
    parser.add_argument("--limit", type=int, default=0, help="Limit pending seeds in this invocation (0=all)")
    parser.add_argument("--seed-id", action="append", default=[])
    parser.add_argument("--model", help="Probe one configured model on pending seeds; normal runs use quota fallback order")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed seeds with all previous raw responses retained")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "status":
        for name in ("progress.json", "summary.json"):
            if (out / name).exists():
                print((out / name).read_text())
        return
    with (out / ".worker.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A worker already owns this output directory")
        seeds = prepare(cfg, out)
        if args.command == "prepare":
            print(json.dumps({"seeds": len(seeds), "expected_variants": len(seeds) * cfg["variants_per_seed"],
                              "output_dir": str(out)}, ensure_ascii=False))
        elif args.command == "export":
            print(json.dumps(export(out, seeds), ensure_ascii=False))
        elif args.command == "recover":
            print(json.dumps({"recovered_seeds": recover(out, seeds, cfg["variants_per_seed"])}))
        else:
            if args.retry_failed:
                for seed in seeds:
                    if args.seed_id and seed["seed_id"] not in args.seed_id:
                        continue
                    failed_path = out / "failed" / f"{seed['seed_id']}.json"
                    if failed_path.exists():
                        failed_path.rename(failed_path.with_suffix(".retried-" + str(time.time_ns()) + ".json"))
            if args.model:
                selected = [m for m in cfg["models"] if m["name"] == args.model]
                if not selected:
                    raise ValueError("Probe model is not in the configured allowlist")
                cfg = {**cfg, "models": selected}
            try:
                run(cfg, out, seeds, args.key_file, args.limit, args.seed_id)
            except BaseException as exc:
                export(out, seeds)
                state_path = out / "progress.json"
                current = json.loads(state_path.read_text()) if state_path.exists() else {}
                atomic_json(state_path, {**current, "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                                         "updated_at": now(), "error_type": type(exc).__name__,
                                         "error": str(redact(str(exc), load_keys(args.key_file)))})
                raise


if __name__ == "__main__":
    main()
