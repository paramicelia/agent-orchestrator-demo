"""Eval harness — LLM-as-judge over a fixed dataset.

Run locally::

    python -m eval.run_eval

CI runs ``tests/test_eval_smoke.py`` with mocked Groq calls instead, so we
never burn quota in GitHub Actions.
"""
