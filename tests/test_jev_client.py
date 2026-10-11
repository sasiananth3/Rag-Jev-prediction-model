from jev_client import JEVClient


class FakeNoulResult:
    def __init__(self, value, confidence=0.9):
        self.noul = value
        self.confidence = confidence


class FakeResponse:
    def __init__(self, values):
        self.nouls = {
            name: FakeNoulResult(value)
            for name, value in values.items()
        }


class FakeTypeSafeClient:
    values = {}
    received_state = None
    received_questions = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def system_one(self, state, questions):
        self.__class__.received_state = state
        self.__class__.received_questions = questions
        return FakeResponse(self.__class__.values)


def test_jev_flags_insufficient_retrieval(monkeypatch):
    import jev_client

    FakeTypeSafeClient.values = {
        "retrieval_relevance": 0.8,
        "evidence_sufficiency": 0.3,
    }
    monkeypatch.setattr(jev_client, "TypeSafeClient", FakeTypeSafeClient)

    client = JEVClient(api_key="test-key", model="jev-latest")
    result = client.evaluate(
        "What is the conclusion?",
        [{"text": "A conclusion passage", "source": "paper.pdf", "page": 4}],
    )

    assert result["relevance"] == 0.8
    assert result["sufficiency"] == 0.3
    assert result["groundedness"] is None
    assert result["needs_retry"] is True
    assert FakeTypeSafeClient.received_state["retrieved_chunks"][0]["page"] == 4


def test_jev_evaluates_answer_grounding(monkeypatch):
    import jev_client

    FakeTypeSafeClient.values = {
        "retrieval_relevance": 0.9,
        "evidence_sufficiency": 0.95,
        "answer_groundedness": 0.87,
    }
    monkeypatch.setattr(jev_client, "TypeSafeClient", FakeTypeSafeClient)

    result = JEVClient(api_key="test-key").evaluate(
        "What is the conclusion?",
        [{"text": "The conclusion is A.", "source": "paper.pdf", "page": 2}],
        answer="The conclusion is A.",
    )

    assert result["groundedness"] == 0.87
    assert result["needs_retry"] is False
    assert result["judge_confidence"] == 0.9
