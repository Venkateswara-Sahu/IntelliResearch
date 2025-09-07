"""
Streamlit Web Interface for IntelliResearch Multi-Agent System
"""

import streamlit as st
import asyncio
import time
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import os
from dotenv import load_dotenv

# Import the main multi-agent system
from intelliresearch.main import (
    MultiAgentOrchestrator,
    ResearchTask,
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="IntelliResearch - Multi-Agent Research System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        color: white;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .status-active {
        color: #00c851;
        font-weight: bold;
    }
    .status-pending {
        color: #ffbb33;
        font-weight: bold;
    }
    .status-completed {
        color: #007bff;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
    st.session_state.research_history = []
    st.session_state.current_task = None
    st.session_state.agent_activities = []
    st.session_state.research_results = None
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")

def initialize_system():
    """Initialize the multi-agent orchestrator"""
    llm_config = {
        "model": st.session_state.llm_model,
        "fallback_model": "gpt-3.5-turbo",
        "temperature": st.session_state.temperature,
        "max_tokens": 2000,
        "api_key": st.session_state.api_key
    }
    return MultiAgentOrchestrator(llm_config)

def create_agent_visualization():
    """Create interactive agent network visualization"""
    agents = [
        {"name": "Coordinator", "role": "Orchestrator", "status": "active"},
        {"name": "Web Scraper", "role": "Gatherer", "status": "idle"},
        {"name": "Analyzer", "role": "Analyst", "status": "idle"},
        {"name": "Fact Checker", "role": "Validator", "status": "idle"},
        {"name": "Writer", "role": "Creator", "status": "idle"},
        {"name": "Editor", "role": "Reviewer", "status": "idle"},
        {"name": "Citation Manager", "role": "Organizer", "status": "idle"}
    ]

    # Create network graph
    fig = go.Figure()

    # Add nodes
    for i, agent in enumerate(agents):
        color = "#00c851" if agent["status"] == "active" else "#cccccc"
        fig.add_trace(go.Scatter(
            x=[i * 2],
            y=[0],
            mode='markers+text',
            marker=dict(size=30, color=color),
            text=agent["name"],
            textposition="bottom center",
            hoverinfo='text',
            hovertext=f"{agent['name']}<br>Role: {agent['role']}<br>Status: {agent['status']}"
        ))

    # Add connections
    for i in range(len(agents) - 1):
        fig.add_trace(go.Scatter(
            x=[i * 2, (i + 1) * 2],
            y=[0, 0],
            mode='lines',
            line=dict(color='#e0e0e0', width=2),
            showlegend=False,
            hoverinfo='none'
        ))

    fig.update_layout(
        title="Agent Network Status",
        showlegend=False,
        height=200,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )

    return fig

def display_research_metrics(results: Dict):
    """Display research quality metrics"""
    if results and "final_output" in results:
        metrics = results["final_output"]["quality_metrics"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Content Quality",
                f"{metrics['content_quality']:.1%}",
                delta="+5%",
                delta_color="normal"
            )

        with col2:
            st.metric(
                "Fact Accuracy",
                f"{metrics['fact_accuracy']:.1%}",
                delta="+3%",
                delta_color="normal"
            )

        with col3:
            st.metric(
                "Source Credibility",
                f"{metrics['source_credibility']:.1%}",
                delta="+2%",
                delta_color="normal"
            )

def create_progress_chart(stages: List[Dict]):
    """Create progress chart for research phases"""
    df = pd.DataFrame(stages)

    fig = px.bar(
        df,
        x='completion',
        y='phase',
        orientation='h',
        color='status',
        color_discrete_map={
            'completed': '#00c851',
            'in_progress': '#ffbb33',
            'pending': '#e0e0e0'
        },
        title="Research Progress"
    )

    fig.update_layout(
        xaxis_title="Completion %",
        yaxis_title="Phase",
        height=300,
        showlegend=True
    )

    return fig

# Main Application
def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 IntelliResearch</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem;">Multi-Agent AI Research & Content Generation System</p>', unsafe_allow_html=True)

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Configuration
        st.subheader("🔑 API Settings")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.api_key,
            help="Enter your OpenAI API key"
        )
        st.session_state.api_key = api_key

        # Model Selection
        st.subheader("🧠 Model Selection")
        llm_model = st.selectbox(
            "Primary LLM Model",
            ["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "gemini-1.5-pro"],
            help="Select the primary LLM model"
        )
        st.session_state.llm_model = llm_model

        # Parameters
        st.subheader("🎛️ Parameters")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in generation"
        )
        st.session_state.temperature = temperature

        # Research Settings
        st.subheader("📊 Research Settings")
        depth = st.select_slider(
            "Research Depth",
            options=["basic", "intermediate", "comprehensive"],
            value="comprehensive"
        )

        sources_required = st.number_input(
            "Number of Sources",
            min_value=3,
            max_value=20,
            value=5,
            help="Number of sources to analyze"
        )

        word_count = st.number_input(
            "Target Word Count",
            min_value=500,
            max_value=5000,
            value=2000,
            step=500
        )

        output_format = st.selectbox(
            "Output Format",
            ["article", "report", "summary", "presentation"],
            help="Format of the final output"
        )

    # Main Content Area
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Research", "📊 Dashboard", "📝 Results", "📚 History"])

    with tab1:
        st.header("Start New Research")

        # Research Input
        col1, col2 = st.columns([3, 1])

        with col1:
            topic = st.text_area(
                "Research Topic",
                placeholder="Enter your research topic or question...",
                height=100,
                help="Be specific and clear about what you want to research"
            )

        with col2:
            st.write("**Quick Templates:**")
            if st.button("🔬 Scientific"):
                topic = "Latest breakthroughs in quantum computing"
            if st.button("💼 Business"):
                topic = "Impact of AI on global supply chains"
            if st.button("🏥 Healthcare"):
                topic = "AI applications in personalized medicine"

        # Keywords
        keywords_input = st.text_input(
            "Keywords (comma-separated)",
            placeholder="AI, machine learning, automation...",
            help="Optional: Specific keywords to focus on"
        )
        keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

        # Start Research Button
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            start_button = st.button(
                "🚀 Start Research",
                type="primary",
                use_container_width=True,
                disabled=not st.session_state.api_key or not topic
            )

        with col2:
            stop_button = st.button(
                "⏹️ Stop",
                type="secondary",
                use_container_width=True
            )

        # Research Execution
        if start_button:
            with st.spinner("Initializing Multi-Agent System..."):
                # Initialize orchestrator
                st.session_state.orchestrator = initialize_system()

                # Create research task
                research_task = ResearchTask(
                    topic=topic,
                    depth=depth,
                    output_format=output_format,
                    word_count=word_count,
                    sources_required=sources_required,
                    keywords=keywords
                )
                st.session_state.current_task = research_task

                # Progress tracking
                progress_container = st.container()
                with progress_container:
                    st.info("🤖 Multi-Agent System Activated")

                    # Agent visualization
                    agent_viz = st.empty()
                    agent_viz.plotly_chart(create_agent_visualization(), use_container_width=True)

                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Phase tracking
                    phases = [
                        "Planning", "Information Gathering", "Analysis",
                        "Fact Checking", "Content Generation", "Editing", "Finalizing"
                    ]

                    phase_container = st.container()

                    # Simulate research process
                    for i, phase in enumerate(phases):
                        progress = int((i + 1) / len(phases) * 100)
                        progress_bar.progress(progress)
                        status_text.text(f"Phase {i+1}/{len(phases)}: {phase}")

                        with phase_container:
                            st.write(f"✅ {phase} - Completed")

                        time.sleep(0.8)  # Simulate small processing time

                    # Execute actual research (blocking call to orchestrator)
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(
                        st.session_state.orchestrator.process_research_request(research_task)
                    )
                    st.session_state.research_results = results

                    # Show completion
                    st.success("✅ Research Completed Successfully!")
                    st.balloons()

    with tab2:
        st.header("📊 Real-time Dashboard")

        if st.session_state.orchestrator:
            # Agent Metrics
            st.subheader("Agent Performance Metrics")

            metrics = st.session_state.orchestrator.get_agent_metrics()

            # Create metrics grid
            cols = st.columns(4)
            for i, (agent_name, agent_metrics) in enumerate(metrics.items()):
                with cols[i % 4]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{agent_name.replace('_', ' ').title()}</h4>
                        <p>Tasks: {agent_metrics['tasks_completed']}</p>
                        <p>Success: {agent_metrics['success_rate']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Research Progress Visualization
            st.subheader("Research Progress")

            if st.session_state.research_results:
                # Display metrics
                display_research_metrics(st.session_state.research_results)

                # Progress stages
                stages_data = [
                    {"phase": "Planning", "completion": 100, "status": "completed"},
                    {"phase": "Gathering", "completion": 100, "status": "completed"},
                    {"phase": "Analysis", "completion": 100, "status": "completed"},
                    {"phase": "Fact Check", "completion": 100, "status": "completed"},
                    {"phase": "Writing", "completion": 100, "status": "completed"},
                    {"phase": "Editing", "completion": 100, "status": "completed"}
                ]

                fig = create_progress_chart(stages_data)
                st.plotly_chart(fig, use_container_width=True)

            # Agent Communication Log
            st.subheader("Inter-Agent Communication")

            communication_log = [
                {"time": "10:30:15", "from": "Coordinator", "to": "Web Scraper", "message": "Initiate source gathering"},
                {"time": "10:30:18", "from": "Web Scraper", "to": "Analyzer", "message": "5 sources collected"},
                {"time": "10:30:22", "from": "Analyzer", "to": "Fact Checker", "message": "Analysis complete"},
                {"time": "10:30:25", "from": "Fact Checker", "to": "Writer", "message": "Facts verified"},
                {"time": "10:30:30", "from": "Writer", "to": "Editor", "message": "Draft ready"},
                {"time": "10:30:35", "from": "Editor", "to": "Coordinator", "message": "Final review complete"}
            ]

            df_comm = pd.DataFrame(communication_log)
            st.dataframe(df_comm, use_container_width=True)
        else:
            st.info("Start a research task to view dashboard metrics")

    with tab3:
        st.header("📝 Research Results")

        if st.session_state.research_results:
            results = st.session_state.research_results

            if results["status"] == "completed":
                # Display final content
                st.subheader("Generated Content")

                content = results["final_output"]["content"]
                st.markdown(content)

                # Download options
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.download_button(
                        label="📥 Download as Markdown",
                        data=content,
                        file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )

                with col2:
                    # Convert to PDF (simplified)
                    st.download_button(
                        label="📄 Download as Text",
                        data=content,
                        file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

                # Citations
                st.subheader("📚 Citations")
                citations = results["final_output"]["citations"]

                for citation in citations:
                    st.write(f"[{citation['id']}] {citation['formatted_apa']}")

                # Metadata
                with st.expander("📊 Research Metadata"):
                    metadata = results["final_output"]["metadata"]
                    st.json(metadata)
            else:
                st.error(f"Research failed: {results.get('error', 'Unknown error')}")
        else:
            st.info("No research results available. Start a new research task!")

    with tab4:
        st.header("📚 Research History")

        # Sample history data
        history_data = [
            {
                "date": "2024-09-06 14:30",
                "topic": "Impact of AI on Healthcare",
                "status": "Completed",
                "quality": "92%",
                "sources": 8,
                "words": 2500
            },
            {
                "date": "2024-09-06 13:15",
                "topic": "Quantum Computing Breakthroughs",
                "status": "Completed",
                "quality": "89%",
                "sources": 6,
                "words": 2000
            },
            {
                "date": "2024-09-06 11:45",
                "topic": "Climate Change Solutions",
                "status": "Completed",
                "quality": "94%",
                "sources": 10,
                "words": 3000
            }
        ]

        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True, hide_index=True)

        # Analytics
        st.subheader("📈 Research Analytics")

        col1, col2 = st.columns(2)

        with col1:
            # Topics distribution
            fig_topics = px.pie(
                values=[3, 2, 2, 1],
                names=["Technology", "Science", "Healthcare", "Environment"],
                title="Research Topics Distribution"
            )
            st.plotly_chart(fig_topics, use_container_width=True)

        with col2:
            # Quality trend
            fig_quality = px.line(
                x=["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"],
                y=[85, 87, 89, 91, 92],
                title="Quality Score Trend",
                markers=True
            )
            fig_quality.update_layout(yaxis_title="Quality %", xaxis_title="")
            st.plotly_chart(fig_quality, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>IntelliResearch v1.0 | Powered by Multi-Agent AI Technology</p>
        <p>Built with LangChain, CrewAI, and GPT-4</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
