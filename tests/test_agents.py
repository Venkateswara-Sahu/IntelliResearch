# import pytest
# from intelliresearch.main import WebScraperAgent, ContentAnalyzer, FactChecker, WriterAgent, EditorAgent, CitationManager
#
# @pytest.mark.asyncio
# async def test_web_scraper():
#     agent = WebScraperAgent({})
#     results = await agent.process_task({"keywords": ["AI"]})
#     assert isinstance(results, list)
#     assert "url" in results[0]
#
# @pytest.mark.asyncio
# async def test_content_analyzer():
#     agent = ContentAnalyzer({})
#     analysis = await agent.process_task([{"title": "AI Research", "url": "http://example.com"}])
#     assert "key_themes" in analysis
#
# @pytest.mark.asyncio
# async def test_fact_checker():
#     agent = FactChecker({})
#     verification = await agent.process_task({"key_themes": [{"theme": "AI is powerful", "source_refs": ["http://example.com"]}]})
#     assert "verified_claims" in verification
#
# @pytest.mark.asyncio
# async def test_writer():
#     agent = WriterAgent({})
#     content = await agent.process_task({"topic": "AI Research"})
#     assert isinstance(content, str)
#     assert "AI Research" in content
#
# @pytest.mark.asyncio
# async def test_editor():
#     agent = EditorAgent({})
#     result = await agent.process_task("This is a test content.")
#     assert "edited_content" in result
#
# @pytest.mark.asyncio
# async def test_citation_manager():
#     agent = CitationManager({})
#     citations = await agent.process_task([{"title": "AI Paper", "url": "http://example.com"}])
#     assert isinstance(citations, list)
#     assert "formatted_apa" in citations[0]


"""
Unit tests for IntelliResearch Multi-Agent System
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intelliresearch.main import (
    AgentRole,
    Message,
    ResearchTask,
    BaseAgent,
    ResearchCoordinator,
    WebScraperAgent,
    ContentAnalyzer,
    FactChecker,
    WriterAgent,
    EditorAgent,
    CitationManager,
    MultiAgentOrchestrator
)


# ================== Test Fixtures ==================

@pytest.fixture
def llm_config():
    """Standard LLM configuration for testing"""
    return {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 2000,
        "api_key": "test-key"
    }


@pytest.fixture
def research_task():
    """Standard research task for testing"""
    return ResearchTask(
        topic="Test Research Topic",
        depth="comprehensive",
        output_format="article",
        word_count=1000,
        sources_required=3,
        keywords=["test", "research", "AI"]
    )


@pytest.fixture
def orchestrator(llm_config):
    """Initialize orchestrator for testing"""
    return MultiAgentOrchestrator(llm_config)


# ================== Test Data Models ==================

class TestDataModels:
    """Test data models and enums"""

    def test_agent_roles(self):
        """Test AgentRole enum values"""
        assert AgentRole.RESEARCH_COORDINATOR.value == "research_coordinator"
        assert AgentRole.WEB_SCRAPER.value == "web_scraper"
        assert AgentRole.CONTENT_ANALYZER.value == "content_analyzer"
        assert AgentRole.FACT_CHECKER.value == "fact_checker"
        assert AgentRole.WRITER.value == "writer"
        assert AgentRole.EDITOR.value == "editor"
        assert AgentRole.CITATION_MANAGER.value == "citation_manager"
        assert len(AgentRole) == 7

    def test_message_creation(self):
        """Test Message dataclass"""
        msg = Message(
            sender="Agent1",
            receiver="Agent2",
            content={"data": "test"},
            message_type="task"
        )
        assert msg.sender == "Agent1"
        assert msg.receiver == "Agent2"
        assert msg.content["data"] == "test"
        assert msg.message_type == "task"
        assert isinstance(msg.timestamp, datetime)
        assert msg.priority == 1

    def test_research_task_creation(self):
        """Test ResearchTask dataclass"""
        task = ResearchTask(
            topic="AI in Healthcare",
            depth="basic",
            output_format="summary",
            word_count=500,
            sources_required=2
        )
        assert task.topic == "AI in Healthcare"
        assert task.depth == "basic"
        assert task.output_format == "summary"
        assert task.word_count == 500
        assert task.sources_required == 2
        assert task.keywords == []
        assert task.constraints == {}


# ================== Test Base Agent ==================

class TestBaseAgent:
    """Test BaseAgent functionality"""

    class DummyAgent(BaseAgent):
        """Concrete implementation for testing"""

        async def process_task(self, task):
            return {"processed": True}

    def test_agent_initialization(self, llm_config):
        """Test agent initialization"""
        agent = self.DummyAgent("TestAgent", AgentRole.WRITER, llm_config)
        assert agent.name == "TestAgent"
        assert agent.role == AgentRole.WRITER
        assert agent.llm_config == llm_config
        assert agent.message_queue == []
        assert agent.knowledge_base == {}
        assert agent.performance_metrics["tasks_completed"] == 0

    @pytest.mark.asyncio
    async def test_message_handling(self, llm_config):
        """Test message receiving and sending"""
        agent = self.DummyAgent("TestAgent", AgentRole.WRITER, llm_config)

        # Test receiving message
        msg = Message(
            sender="Sender",
            receiver="TestAgent",
            content="Test content",
            message_type="info"
        )
        await agent.receive_message(msg)
        assert len(agent.message_queue) == 1
        assert agent.message_queue[0].content == "Test content"

        # Test sending message
        sent_msg = await agent.send_message(
            receiver="Receiver",
            content="Reply",
            message_type="response"
        )
        assert sent_msg.sender == "TestAgent"
        assert sent_msg.receiver == "Receiver"
        assert sent_msg.content == "Reply"

    def test_metrics_update(self, llm_config):
        """Test performance metrics update"""
        agent = self.DummyAgent("TestAgent", AgentRole.WRITER, llm_config)

        # Update with successful task
        agent.update_metrics(task_time=5.0, success=True)
        assert agent.performance_metrics["tasks_completed"] == 1
        assert agent.performance_metrics["success_rate"] == 100

        # Update with failed task
        agent.update_metrics(task_time=3.0, success=False)
        assert agent.performance_metrics["tasks_completed"] == 2
        assert agent.performance_metrics["success_rate"] == 50


# ================== Test Individual Agents ==================

class TestSpecializedAgents:
    """Test specialized agent implementations"""

    @pytest.mark.asyncio
    async def test_research_coordinator(self, llm_config, research_task):
        """Test ResearchCoordinator agent"""
        coordinator = ResearchCoordinator(llm_config)
        result = await coordinator.process_task(research_task)

        assert result["topic"] == research_task.topic
        assert "phases" in result
        assert len(result["phases"]) == 5
        assert result["phases"][0]["phase"] == "information_gathering"
        assert "requirements" in result

    @pytest.mark.asyncio
    async def test_web_scraper(self, llm_config):
        """Test WebScraperAgent"""
        scraper = WebScraperAgent(llm_config)
        task = {"keywords": ["AI", "testing"]}

        sources = await scraper.process_task(task)

        assert len(sources) == 2
        assert all("url" in source for source in sources)
        assert all("title" in source for source in sources)
        assert all("credibility_score" in source for source in sources)
        assert scraper.sources_database == sources

    @pytest.mark.asyncio
    async def test_content_analyzer(self, llm_config):
        """Test ContentAnalyzer agent"""
        analyzer = ContentAnalyzer(llm_config)
        sources = [
            {"title": "Source 1", "url": "http://test1.com"},
            {"title": "Source 2", "url": "http://test2.com"}
        ]

        analysis = await analyzer.process_task(sources)

        assert "key_themes" in analysis
        assert "main_arguments" in analysis
        assert "synthesis" in analysis
        assert len(analysis["key_themes"]) == 2
        assert analyzer.analysis_results == analysis

    @pytest.mark.asyncio
    async def test_fact_checker(self, llm_config):
        """Test FactChecker agent"""
        checker = FactChecker(llm_config)
        content = {
            "key_themes": [
                {"theme": "Test theme", "source_refs": ["http://test.com"]}
            ]
        }

        result = await checker.process_task(content)

        assert "verified_claims" in result
        assert "disputed_claims" in result
        assert "confidence_score" in result
        assert len(result["verified_claims"]) == 1
        assert result["confidence_score"] == 0.92

    @pytest.mark.asyncio
    async def test_writer_agent(self, llm_config):
        """Test WriterAgent"""
        writer = WriterAgent(llm_config)
        research_data = {
            "topic": "Test Topic",
            "analysis": {"key_themes": []},
            "verification": {"confidence_score": 0.9}
        }

        content = await writer.process_task(research_data)

        assert isinstance(content, str)
        assert "Test Topic" in content
        assert "Executive Summary" in content
        assert len(writer.drafts) == 1
        assert writer.drafts[0]["version"] == 1

    @pytest.mark.asyncio
    async def test_editor_agent(self, llm_config):
        """Test EditorAgent"""
        editor = EditorAgent(llm_config)
        content = "This is test content for editing."

        result = await editor.process_task(content)

        assert "original_content" in result
        assert "edited_content" in result
        assert "changes_made" in result
        assert "quality_score" in result
        assert result["quality_score"] == 0.88
        assert len(editor.editing_history) == 1

    @pytest.mark.asyncio
    async def test_citation_manager(self, llm_config):
        """Test CitationManager agent"""
        citation_mgr = CitationManager(llm_config)
        sources = [
            {
                "author": "Test Author",
                "title": "Test Title",
                "url": "http://test.com",
                "publication_date": "2024"
            }
        ]

        citations = await citation_mgr.process_task(sources)

        assert len(citations) == 1
        assert citations[0]["id"] == 1
        assert "formatted_apa" in citations[0]
        assert "formatted_mla" in citations[0]
        assert "Test Author" in citations[0]["formatted_apa"]


# ================== Test Orchestrator ==================

class TestMultiAgentOrchestrator:
    """Test the main orchestrator"""

    def test_orchestrator_initialization(self, llm_config):
        """Test orchestrator initialization"""
        orchestrator = MultiAgentOrchestrator(llm_config)

        assert orchestrator.llm_config == llm_config
        assert len(orchestrator.agents) == 7
        assert "coordinator" in orchestrator.agents
        assert "web_scraper" in orchestrator.agents
        assert isinstance(orchestrator.communication_bus, list)
        assert orchestrator.results_cache == {}

    @pytest.mark.asyncio
    async def test_process_research_request(self, orchestrator, research_task):
        """Test complete research request processing"""
        results = await orchestrator.process_research_request(research_task)

        assert results["status"] == "completed"
        assert "stages" in results
        assert "final_output" in results
        assert "content" in results["final_output"]
        assert "citations" in results["final_output"]
        assert "quality_metrics" in results["final_output"]

    def test_get_agent_metrics(self, orchestrator):
        """Test getting agent metrics"""
        metrics = orchestrator.get_agent_metrics()

        assert len(metrics) == 7
        assert all(agent in metrics for agent in orchestrator.agents.keys())
        assert all("tasks_completed" in m for m in metrics.values())
        assert all("success_rate" in m for m in metrics.values())

    @pytest.mark.asyncio
    async def test_broadcast_message(self, orchestrator):
        """Test message broadcasting"""
        msg = Message(
            sender="test",
            receiver="coordinator",
            content="Test broadcast",
            message_type="info"
        )

        await orchestrator.broadcast_message(msg)

        assert len(orchestrator.communication_bus) == 1
        assert orchestrator.communication_bus[0] == msg

        # Check if coordinator received the message
        coordinator = orchestrator.agents["coordinator"]
        assert len(coordinator.message_queue) == 1


# ================== Integration Tests ==================

class TestIntegration:
    """Integration tests for the complete system"""

    @pytest.mark.asyncio
    async def test_end_to_end_research(self, llm_config):
        """Test complete end-to-end research workflow"""
        orchestrator = MultiAgentOrchestrator(llm_config)

        task = ResearchTask(
            topic="Integration Test Topic",
            depth="basic",
            output_format="summary",
            word_count=500,
            sources_required=2,
            keywords=["test"]
        )

        results = await orchestrator.process_research_request(task)

        # Verify all stages completed
        assert results["status"] == "completed"
        assert all(stage in results["stages"] for stage in [
            "planning", "sources", "analysis", "verification",
            "content", "editing", "citations"
        ])

        # Verify final output structure
        final_output = results["final_output"]
        assert "content" in final_output
        assert "citations" in final_output
        assert "quality_metrics" in final_output
        assert "metadata" in final_output

        # Verify content generation
        assert len(final_output["content"]) > 100
        assert final_output["metadata"]["word_count"] > 0
        assert final_output["metadata"]["sources_used"] == 2

    @pytest.mark.asyncio
    async def test_error_handling(self, llm_config):
        """Test error handling in the system"""
        orchestrator = MultiAgentOrchestrator(llm_config)

        # Create task with invalid parameters
        task = ResearchTask(
            topic="",  # Empty topic should cause issues
            depth="invalid_depth",
            output_format="invalid_format",
            word_count=-100,
            sources_required=0
        )

        # The system should handle this gracefully
        results = await orchestrator.process_research_request(task)

        # Even with invalid input, system should complete or fail gracefully
        assert results["status"] in ["completed", "error"]


# ================== Performance Tests ==================

class TestPerformance:
    """Performance and scalability tests"""

    @pytest.mark.asyncio
    async def test_concurrent_tasks(self, llm_config):
        """Test handling multiple concurrent research tasks"""
        orchestrator = MultiAgentOrchestrator(llm_config)

        tasks = [
            ResearchTask(
                topic=f"Topic {i}",
                depth="basic",
                output_format="summary",
                word_count=500,
                sources_required=2
            )
            for i in range(3)
        ]

        # Process tasks concurrently
        results = await asyncio.gather(*[
            orchestrator.process_research_request(task)
            for task in tasks
        ])

        # All tasks should complete
        assert len(results) == 3
        assert all(r["status"] == "completed" for r in results)

    def test_agent_scalability(self, llm_config):
        """Test adding/removing agents dynamically"""
        orchestrator = MultiAgentOrchestrator(llm_config)
        initial_count = len(orchestrator.agents)

        # Should be able to access and modify agents
        assert initial_count == 7

        # Test agent removal (simulation)
        if "editor" in orchestrator.agents:
            del orchestrator.agents["editor"]
            assert len(orchestrator.agents) == initial_count - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])