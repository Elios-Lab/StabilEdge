#!/usr/bin/env python3
"""Isolated Transformer engine: JSON request in, JSONL events out.

Uses the upstream model/verifiers, but owns radius search and scheduling so a
failed certification or incomplete task set cannot become a certificate.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verification.core import Status, VerificationResult
from verification.radius import search_radius, schedule_tasks
from verification.transformer_request import TransformerRequest, checkpoint_directory, position_tasks

PROTOCOL = sys.stdout


class Cancelled(Exception):
    pass


class Deadline(Exception):
    pass


def emit(event, **fields):
    record = {"event": event, "schema_version": 1, **fields}
    PROTOCOL.write(json.dumps(record, allow_nan=False) + "\n")
    PROTOCOL.flush()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def repository_revision(repository):
    result = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unversioned"


def check_environment():
    required = ["torch", "numpy", "pandas", "pytorch_pretrained_bert", "nltk", "tqdm", "cvxpy"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Missing Transformer dependencies: %s. Use environment-transformer.yml or requirements-transformer.txt in a separate environment." % ", ".join(missing))


def load_backend(request):
    # Select the physical GPU before importing torch; upstream .cuda() calls
    # then consistently use logical cuda:0. CPU mode cannot allocate GPU tensors.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if request.device == "cpu" else request.device.split(":")[1]
    check_environment()
    import torch
    import numpy as np
    if request.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("The requested CUDA device is unavailable in this environment.")
    torch.manual_seed(request.seed)
    random.seed(request.seed)
    np.random.seed(request.seed)
    sys.path.insert(0, str(Path(request.repository).expanduser().resolve()))
    from Parser import Parser
    from Models.BERT import BERT
    from Models.modeling import BertConfig, BertForSequenceClassification
    from experiments.cascade_verification import make_verifier, set_standard_output, valid_positions

    checkpoint, config = checkpoint_directory(request.checkpoint)
    args = Parser().getParser().parse_args(["--dir", str(checkpoint), "--data", "sst"])
    args.device = torch.device("cpu" if request.device == "cpu" else "cuda:0")
    args.p = 10 if request.norm == "inf" else int(request.norm)
    args.perturbed_words = request.perturbed_tokens
    args.seed = request.seed
    args.hidden_act = config.get("hidden_act", "gelu")
    args.layer_norm = config.get("layer_norm", "standard")
    args.num_labels = 2
    args.max_sent_length = int(config.get("max_position_embeddings", 512))
    args.debug = args.verbose = False
    if request.max_length > args.max_sent_length:
        raise ValueError("Token limit exceeds the checkpoint's positional embedding capacity.")
    # Load the checkpoint using its existing BERT wrapper without rebuilding
    # a vocabulary, loading benchmark datasets, or creating a training optimizer.
    target = BERT.__new__(BERT)
    target.general_init(args, [])
    target.model = BertForSequenceClassification(BertConfig.from_dict(config), 2)
    weights = torch.load(checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True)
    # Migrate legacy LayerNorm keys, then reject incomplete architecture matches.
    weights = {key.replace("gamma", "weight").replace("beta", "bias"): value for key, value in weights.items()}
    target.model.load_state_dict(weights, strict=True)
    target.model.to(args.device)
    if target.model.classifier.out_features != 2:
        raise ValueError("This engine requires a two-class Transformer checkpoint.")
    target.model.eval()
    versions = {"torch": torch.__version__, "numpy": np.__version__}
    return SimpleNamespace(torch=torch, args=args, target=target, make_verifier=make_verifier,
                           set_standard_output=set_standard_output, valid_positions=valid_positions,
                           versions=versions)


def run_example(backend, request, example, sample_id):
    target, torch = backend.target, backend.torch
    # Inspect the full tokenization first; never silently truncate the property.
    tokens = ["[CLS]"] + target.tokenizer.tokenize(example["text"]) + ["[SEP]"]
    if len(tokens) > request.max_length:
        return VerificationResult("transformer", sample_id, Status.UNKNOWN,
                                  reason="Text exceeds the configured token limit; no truncation was applied.",
                                  details={"tokens": tokens, "token_count": len(tokens)}).to_dict()
    model_example = {"sent_a": example["text"].split(), "label": example["label"]}
    embeddings, token_batches = target.get_embeddings([model_example])
    tokens = token_batches[0]
    standard = target.step([model_example])[-1]
    predicted = int(standard["pred_labels"][0])
    valid = backend.valid_positions(tokens)
    tasks = position_tasks(valid, request.positions, request.perturbed_tokens)
    if request.action == "preview":
        return VerificationResult("transformer", sample_id, Status.UNKNOWN, reason="Token preview only; verification was not run.",
                                  details={"tokens": tokens, "valid_positions": valid, "task_count": len(tasks),
                                           "label": example["label"], "prediction": predicted}).to_dict()
    if predicted != example["label"]:
        return VerificationResult("transformer", sample_id, Status.COUNTEREXAMPLE,
                                  reason="The unperturbed example is already misclassified.",
                                  details={"tokens": tokens, "label": example["label"], "prediction": predicted}).to_dict()
    if not tasks:
        return VerificationResult("transformer", sample_id, Status.UNKNOWN,
                                  reason="No eligible perturbation tasks.", details={"tokens": tokens}).to_dict()
    verifiers = {method: backend.make_verifier(backend.args, target, method) for method in request.methods}
    for verifier in verifiers.values():
        verifier.std = standard
    stats = {method: {"calls": 0, "certified_calls": 0, "seconds": 0.0} for method in request.methods}
    evidence = {}
    emit("sample_started", sample_id=sample_id, tokens=tokens, valid_positions=valid,
         tasks=[list(task) for task in tasks], label=example["label"], prediction=predicted)

    def certify(task, radius):
        for method in request.methods:
            started = time.monotonic()
            stats[method]["calls"] += 1
            # IBP mutates its lower input bound in place. Give every stage/query
            # its own tensor so later checks retain the same nominal embedding.
            safe = verifiers[method].verify_safety(model_example, embeddings.clone(), list(task), radius)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            stats[method]["seconds"] += time.monotonic() - started
            if bool(safe):
                stats[method]["certified_calls"] += 1
                evidence[task] = {"radius": radius, "method": method}
                return True
        return False

    def search(task):
        return search_radius(lambda radius: certify(task, radius), request.radius,
                             request.maximum_radius, request.iterations, request.tolerance)

    def notify(stage, task, value):
        radius = value.radius if hasattr(value, "radius") else value
        emit("task_finished", sample_id=sample_id, stage=stage, positions=list(task), radius=radius)

    details = {"tokens": tokens, "valid_positions": valid, "task_count": len(tasks),
               "scope": "selected_positions" if request.positions else "all_valid_positions",
               "positions": request.positions or valid, "perturbed_tokens": request.perturbed_tokens,
               "norm": request.norm, "perturbation_space": "pre_layer_norm_embeddings",
               "label": example["label"], "prediction": predicted, "stats": stats}
    if request.action == "check":
        outcomes = []
        for task in tasks:
            safe = certify(task, request.radius)
            outcomes.append(safe)
            notify("check_certified" if safe else "check_unknown", task, request.radius)
        certified = all(outcomes)
        radius = request.radius if certified else None
    elif request.action == "search":
        bounds = {}
        for task in tasks:
            bounds[task] = search(task)
            notify("search", task, bounds[task])
        certified = all(bound.certified for bound in bounds.values())
        radius = min(bound.radius for bound in bounds.values()) if certified else None
        details["bounds"] = {",".join(map(str, task)): vars(bound) for task, bound in bounds.items()}
    else:
        rank_started = time.monotonic()
        ranked = list(tasks)
        if request.ranker == "gradient":
            gradients = target.step([model_example], infer_grad=True)[-1]["gradients"].detach()[0]
            scores = {pos: float(torch.norm(gradients[pos], p=2).item()) for pos in valid}
            ranked.sort(key=lambda task: sum(scores[pos] for pos in task), reverse=True)
            details["gradient_scores"] = scores
        elif request.ranker == "random":
            random.Random(request.seed).shuffle(ranked)
        details["ranking_seconds"] = time.monotonic() - rank_started
        scheduled = schedule_tasks(tasks, ranked, request.budget, search, certify, notify)
        certified, radius = scheduled["certified"], scheduled["radius"]
        details.update({key: scheduled[key] for key in ("selected", "validated", "fallback")})
        details["bounds"] = {",".join(map(str, task)): vars(bound) for task, bound in scheduled["bounds"].items()}
    details["certification_evidence"] = {",".join(map(str, task)): value for task, value in evidence.items()}
    return VerificationResult("transformer", sample_id, Status.CERTIFIED if certified else Status.UNKNOWN,
                              radius=radius, reason="" if certified else "Bound propagation could not certify every task.",
                              details=details).to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=0)
    options = parser.parse_args()
    result = {"schema_version": 1, "engine": "transformer", "status": Status.UNKNOWN.value,
              "started_at": datetime.now(timezone.utc).isoformat(), "results": []}
    started = time.monotonic()

    def interrupted(signum, frame):
        raise Cancelled()

    def deadline(signum, frame):
        raise Deadline()

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if options.timeout > 0:
        signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, options.timeout)
    exit_code = 0
    try:
        request = TransformerRequest.from_dict(json.loads(Path(options.request).read_text()))
        result["request"] = request.to_dict()
        result["repository_revision"] = repository_revision(request.repository)
        emit("loading", message="Loading the Transformer checkpoint and verification engine.")
        with contextlib.redirect_stdout(sys.stderr):
            backend = load_backend(request)
        result["versions"] = backend.versions
        emit("ready", versions=backend.versions)
        for index, example in enumerate(request.examples):
            sample_id = str(example.get("id", index))
            with contextlib.redirect_stdout(sys.stderr):
                try:
                    row = run_example(backend, request, example, sample_id)
                except backend.torch.cuda.OutOfMemoryError:
                    backend.torch.cuda.empty_cache()
                    row = VerificationResult("transformer", sample_id, Status.UNKNOWN, reason="Insufficient GPU memory.").to_dict()
            result["results"].append(row)
            atomic_json(options.output, result)
            emit("result", result=row, completed=index + 1, total=len(request.examples))
        statuses = [row["status"] for row in result["results"]]
        result["status"] = (Status.COUNTEREXAMPLE.value if Status.COUNTEREXAMPLE.value in statuses else
                            Status.CERTIFIED.value if all(status == Status.CERTIFIED.value for status in statuses) else Status.UNKNOWN.value)
    except Cancelled:
        result["status"] = Status.CANCELLED.value
    except Deadline:
        result["status"] = Status.TIMEOUT.value
    except Exception as exc:
        result["status"] = Status.ERROR.value
        result["error"] = str(exc)
        traceback.print_exc(file=sys.stderr)
        emit("error", message=str(exc))
        exit_code = 1
    finally:
        if options.timeout > 0:
            signal.setitimer(signal.ITIMER_REAL, 0)
        result["elapsed_seconds"] = time.monotonic() - started
        atomic_json(options.output, result)
        emit("finished", status=result["status"], elapsed_seconds=result["elapsed_seconds"])
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
