"""Tests for AgentRecommendation metadata fields."""


def test_agent_recommendation_has_doi_and_journal():
    """AgentRecommendation should carry doi and journal from Paper."""
    from incite.agent import AgentRecommendation

    rec = AgentRecommendation(
        paper_id="p1",
        rank=1,
        score=0.9,
        title="Test",
        doi="10.1234/test",
        journal="Nature",
    )
    assert rec.doi == "10.1234/test"
    assert rec.journal == "Nature"
    d = rec.to_dict()
    assert d["doi"] == "10.1234/test"
    assert d["journal"] == "Nature"
