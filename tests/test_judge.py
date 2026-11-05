import json

from ragtuner.evalset import EvalQuestion, GoldSpan
from ragtuner.judge import Judge
from ragtuner.pipeline import Hit


def test_judge_runs_offline_with_fake_callers():
    def answer(system, user):
        return "NOT IN CONTEXT" if "gift" in user else "The fee is $5."

    def grade(system, user):
        if "decline" in user and "NOT IN CONTEXT" in user:
            return json.dumps({"score": 1, "reason": "declined correctly"})
        return json.dumps({"score": 1 if "$5" in user else 0, "reason": "checked"})

    j = Judge(answer_caller=answer, judge_caller=grade)
    qs = [EvalQuestion("q1", "What is the restocking fee?", (GoldSpan("d", 0, 10),), answer="$5"),
          EvalQuestion("q2", "What is the gift card policy?", (), question_type="unanswerable")]
    hits = {"q1": [Hit("c", "d", 0, 10, 1.0, 1, "Restocking fee: $5", page=1)], "q2": []}
    res = j.run("cfg", qs, hits)
    assert res.mean_score == 1.0 and res.calls == 4 and res.errors == 0
    assert res.judged[1].declined
    assert res.to_dict()["n"] == 2
