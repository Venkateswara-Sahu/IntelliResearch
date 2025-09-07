# Agent Specifications

## Research Coordinator
- Role: Orchestrates the research workflow
- Inputs: ResearchTask
- Outputs: Research Plan JSON
- Dependencies: All agents

## Web Scraper
- Role: Collects online sources
- Inputs: Keywords, topic
- Outputs: List of sources
- Tools: requests, BeautifulSoup, newspaper3k

## Content Analyzer
- Role: Extracts insights
- Inputs: Sources
- Outputs: Key themes, synthesis
- Methods: LLM summarization, keyword extraction

## Fact Checker
- Role: Verifies claims
- Inputs: Extracted statements
- Outputs: Verified/disputed claims
- Tools: Cross-referencing APIs (simulated)

## Writer
- Role: Drafts structured articles
- Inputs: Analysis + Verification
- Outputs: Markdown/text draft
- Tools: LLM (GPT/Claude/Gemini)

## Editor
- Role: Improves readability & flow
- Inputs: Draft text
- Outputs: Polished text, quality metrics
- Tools: LLM refinement

## Citation Manager
- Role: Manages APA/MLA references
- Inputs: Source metadata
- Outputs: Formatted citations
- Tools: BibTeX, CSL (future)
