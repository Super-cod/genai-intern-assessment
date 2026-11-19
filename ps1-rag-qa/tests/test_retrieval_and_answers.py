import pytest

from app.errors import EmptyIndexError, ServiceUnavailableError
from app.llm.base import LLMOutputError, extract_json
from app.schemas import AnswerDraft, AnswerStatus, QueryRequest

from .conftest import HANDBOOK, MANUAL, UPDATE, FakeLLM


def _ingest(p, *docs):
    for name, text in docs:
        p.ingest(name, text.encode())


def test_query_before_ingest_raises(make_pipeline):
    with pytest.raises(EmptyIndexError):
        make_pipeline().query(QueryRequest(question="anything?"))


def test_identical_upload_is_skipped_and_duplicate_chunks_merged(make_pipeline):
    p = make_pipeline()
    r1 = p.ingest("manual.txt", MANUAL.encode())
    r2 = p.ingest("manual.txt", MANUAL.encode())
    assert r2.chunks == r1.chunks and "already indexed" in r2.warnings[0]
    r3 = p.ingest("manual_copy.txt", MANUAL.encode() + b"\n")      # same text, different bytes
    assert r3.chunks == 0 and any("duplicates" in w for w in r3.warnings)
    assert any("manual_copy.txt" in c.duplicate_of for c in p.index.chunks)


def test_remove_document(make_pipeline):
    p = make_pipeline()
    res = p.ingest("manual.txt", MANUAL.encode())
    assert p.index.remove_document(res.doc_id) and p.index.size == 0


def test_hybrid_retrieval_finds_exact_code(make_pipeline):
    p = make_pipeline()
    _ingest(p, ("manual.txt", MANUAL), ("handbook.txt", HANDBOOK))
    res = p.retriever.retrieve("What does E204 mean?", k=3, mode="hybrid")
    assert "E204" in res.chunks[0].chunk.text


def test_offtopic_question_is_gated_without_calling_llm(make_pipeline):
    p = make_pipeline()
    _ingest(p, ("manual.txt", MANUAL))
    r = p.query(QueryRequest(question="Who won the football world cup?"))
    assert r.status == AnswerStatus.not_found and p.llm.calls == []


def test_grounded_answer_with_valid_citation(make_pipeline):
    p = make_pipeline([{"answerable": True, "answer": "The battery lasts 8 hours [S1].", "cited_sources": ["S1"]}])
    _ingest(p, ("manual.txt", MANUAL))
    r = p.query(QueryRequest(question="How long does the battery last?"))
    assert r.status == AnswerStatus.answered
    assert r.citations[0].doc_name == "manual.txt" and r.grounding_score >= 0.5


def test_invented_citation_label_with_unsupported_content_is_rejected(make_pipeline):
    p = make_pipeline([{"answerable": True, "answer": "The robot is solar powered and waterproof [S9].",
                        "cited_sources": ["S9"]}])
    _ingest(p, ("manual.txt", MANUAL))
    r = p.query(QueryRequest(question="How long does the battery last?"))
    assert r.status == AnswerStatus.unsupported and r.citations == []


def test_invented_citation_label_with_supported_content_is_reattributed(make_pipeline):
    p = make_pipeline([{"answerable": True, "answer": "The battery lasts 8 hours [S9].", "cited_sources": ["S9"]}])
    _ingest(p, ("manual.txt", MANUAL))
    r = p.query(QueryRequest(question="How long does the battery last?"))
    assert r.status == AnswerStatus.answered and r.citations[0].doc_name == "manual.txt"


def test_hallucinated_number_is_rejected(make_pipeline):
    p = make_pipeline([{"answerable": True, "answer": "The battery lasts 12 hours [S1].", "cited_sources": ["S1"]}])
    _ingest(p, ("manual.txt", MANUAL))
    r = p.query(QueryRequest(question="How long does the battery last?"))
    assert r.status == AnswerStatus.unsupported and "12" in r.answer


def test_model_says_insufficient_context(make_pipeline):
    p = make_pipeline([{"answerable": False, "answer": "The documents do not mention a warranty.", "cited_sources": []}])
    _ingest(p, ("manual.txt", MANUAL))
    r = p.query(QueryRequest(question="What is the battery warranty?"))
    assert r.status == AnswerStatus.insufficient_context


def test_conflict_detected_newer_first_and_note_added(make_pipeline):
    p = make_pipeline([{"answerable": True, "answer": "Employees get 24 days [S1].", "cited_sources": ["S1"]}])
    _ingest(p, ("handbook.txt", HANDBOOK), ("update.txt", UPDATE))
    r = p.query(QueryRequest(question="How many days of paid annual leave per year?"))
    prompt = p.llm.calls[0][1]["content"]
    assert prompt.index("update.txt") < prompt.index("handbook.txt")       # newer source first
    assert "OUTDATED" in prompt                                            # older one annotated
    assert r.status == AnswerStatus.answered and r.conflict_note and "update.txt" in r.conflict_note


def test_answer_citing_only_outdated_source_is_regenerated(make_pipeline):
    p = make_pipeline([
        {"answerable": True, "answer": "Employees get 20 days [S2].", "cited_sources": ["S2"]},
        {"answerable": True, "answer": "Employees get 24 days [S1].", "cited_sources": ["S1"]},
    ])
    _ingest(p, ("handbook.txt", HANDBOOK), ("update.txt", UPDATE))
    r = p.query(QueryRequest(question="How many days of paid annual leave per year?"))
    assert len(p.llm.calls) == 2 and "24" in r.answer


def test_followup_is_rewritten_using_history(make_pipeline):
    p = make_pipeline([
        {"answerable": True, "answer": "The battery lasts 8 hours [S1].", "cited_sources": ["S1"]},
        {"standalone_question": "How long does it take to charge the robot battery?"},
        {"answerable": True, "answer": "Charging takes 2.5 hours [S1].", "cited_sources": ["S1"]},
    ])
    _ingest(p, ("manual.txt", MANUAL))
    r1 = p.query(QueryRequest(question="How long does the robot battery last?"))
    r2 = p.query(QueryRequest(question="How long does it take to charge it?", session_id=r1.session_id))
    assert r2.standalone_question == "How long does it take to charge the robot battery?"
    assert "2.5" in r2.answer


def test_llm_down_maps_to_service_unavailable(make_pipeline):
    p = make_pipeline(down=True)
    _ingest(p, ("manual.txt", MANUAL))
    with pytest.raises(ServiceUnavailableError):
        p.query(QueryRequest(question="How long does the battery last?"))


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('Sure! ```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": 2} hope it helps') == {"a": 2}


def test_chat_json_repairs_invalid_output_once():
    llm = FakeLLM(["not json at all", {"answerable": True, "answer": "ok", "cited_sources": []}])
    out = llm.chat_json([{"role": "user", "content": "q"}], AnswerDraft)
    assert out.answer == "ok" and "not valid JSON" in llm.calls[1][-1]["content"]


def test_chat_json_gives_up_after_repair_attempts():
    llm = FakeLLM(["garbage", "still garbage"])
    with pytest.raises(LLMOutputError):
        llm.chat_json([{"role": "user", "content": "q"}], AnswerDraft)


def test_numbers_inside_identifiers_are_not_flagged():
    from app.generation.answerer import grounding_score
    score, bad = grounding_score("The RX-200 payload is 150 kg [S1].", ["Maximum payload: 150 kg."],
                                 "What is the payload of the RX-200?")
    assert bad == []


def test_wrong_citation_label_is_repaired(make_pipeline):
    p = make_pipeline([{"answerable": True, "answer": "E204 means the battery is overheating; cool it for 30 minutes [S2].",
                        "cited_sources": ["S2"]}])
    _ingest(p, ("manual.txt", MANUAL), ("handbook.txt", HANDBOOK))
    r = p.query(QueryRequest(question="What does E204 mean? battery overheating"))
    assert r.status == AnswerStatus.answered
    assert any("E204" in c.snippet for c in r.citations)
