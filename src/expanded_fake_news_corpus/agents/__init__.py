"""Agentes do FakeGen.BR."""

from expanded_fake_news_corpus.agents.headline import (
    HeadlineAgent,
    HeadlineState,
    build_headline_workflow,
)

__all__ = ["HeadlineAgent", "HeadlineState", "build_headline_workflow"]
