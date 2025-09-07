#!/usr/bin/env python
"""
IntelliResearch - Easy Startup Script
Run this file to start the application with your preferred interface
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import webbrowser
import signal
import multiprocessing


def check_requirements():
    """Check if all requirements are installed"""
    print("🔍 Checking requirements...")
    try:
        import streamlit
        import fastapi
        import langchain
        print("✅ All core requirements found!")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("\n📦 Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return False


def check_env():
    """Check if environment variables are set"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  No .env file found. Creating from template...")
        # Create .env.example content
        env_example = """# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Alternative LLM APIs (Optional)
GOOGLE_API_KEY=your-google-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
HUGGINGFACE_API_KEY=your-huggingface-api-key-here

# Application Settings
APP_ENV=development
DEBUG=True
LOG_LEVEL=INFO

# Demo Mode (set to true to run without API keys)
DEMO_MODE=false
"""
        with open(".env", "w") as f:
            f.write(env_example)
        print("✅ Created .env file. Please add your API keys!")
        print("\n🔑 Get your API keys from:")
        print("   - OpenAI: https://platform.openai.com/api-keys")
        print("   - Google AI: https://makersuite.google.com/app/apikey")
        print("   - Anthropic: https://console.anthropic.com/")
        return False

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-openai-api-key-here":
        print("\n⚠️  OpenAI API key not configured!")
        print("Please edit .env file and add your OpenAI API key")
        print("You can get one from: https://platform.openai.com/api-keys")
        use_demo = input("\nWould you like to run in DEMO mode? (y/n): ").lower()
        if use_demo == 'y':
            os.environ["OPENAI_API_KEY"] = "demo-key"
            os.environ["DEMO_MODE"] = "true"
            print("✅ Running in DEMO mode (limited functionality)")
            return True
        return False

    print("✅ Environment configured!")
    return True


def display_menu():
    """Display startup menu"""
    print("\n" + "=" * 60)
    print("🤖 INTELLIRESEARCH - Multi-Agent Research System")
    print("=" * 60)
    print("\nSelect how you want to run the application:\n")
    print("1. 🌐 Web Interface (Streamlit) - Recommended")
    print("2. 🚀 API Server (FastAPI)")
    print("3. 💻 Command Line Interface")
    print("4. 🎯 Run Everything (Web + API)")
    print("5. 📚 View Documentation")
    print("6. 🔧 Run Tests")
    print("7. 🐳 Docker Setup")
    print("0. ❌ Exit")
    print("\n" + "=" * 60)


def run_streamlit():
    """Run Streamlit web interface"""
    print("\n🌐 Starting Streamlit Web Interface...")
    print("📍 Opening browser at http://localhost:8501")
    print("\nPress Ctrl+C to stop the server\n")

    # Open browser after a short delay
    time.sleep(2)
    webbrowser.open("http://localhost:8501")

    try:
        subprocess.run(["streamlit", "run", "intelliresearch/app.py"])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit server stopped")


def run_api():
    """Run FastAPI server"""
    print("\n🚀 Starting FastAPI Server...")
    print("📍 API Documentation at http://localhost:8001/docs")
    print("📍 Alternative docs at http://localhost:8000/redoc")
    print("\nPress Ctrl+C to stop the server\n")

    # Open browser after a short delay
    time.sleep(2)
    webbrowser.open("http://localhost:8001/docs")

    try:
        subprocess.run(["uvicorn", "intelliresearch.api:app", "--reload", "--port", "8001"])
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")


def run_cli():
    """Run command line interface"""
    print("\n💻 Starting Command Line Interface...")
    import asyncio
    from intelliresearch.main import MultiAgentOrchestrator, ResearchTask

    print("\n" + "=" * 60)
    print("INTELLIRESEARCH CLI")
    print("=" * 60)

    # Get user input
    topic = input("\n📝 Enter research topic: ")
    if not topic:
        topic = "The Impact of AI on Healthcare"
        print(f"Using default topic: {topic}")

    depth = input("📊 Research depth (basic/intermediate/comprehensive) [comprehensive]: ") or "comprehensive"
    sources = input("📚 Number of sources (3-20) [5]: ") or "5"
    word_count = input("📄 Word count (500-5000) [2000]: ") or "2000"

    # Create and run research task
    async def run_research():
        llm_config = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "api_key": os.getenv("OPENAI_API_KEY", "demo-key")
        }

        orchestrator = MultiAgentOrchestrator(llm_config)

        task = ResearchTask(
            topic=topic,
            depth=depth,
            output_format="article",
            word_count=int(word_count),
            sources_required=int(sources)
        )

        print("\n🤖 Starting multi-agent research...")
        print("-" * 40)

        # Show progress
        phases = ["Planning", "Gathering", "Analyzing", "Verifying", "Writing", "Editing", "Finalizing"]
        for i, phase in enumerate(phases):
            print(f"Phase {i + 1}/{len(phases)}: {phase}...")
            time.sleep(0.5)  # Simulate processing

        results = await orchestrator.process_research_request(task)

        if results["status"] == "completed":
            print("\n✅ Research Completed!")
            print("\n📝 Generated Content:")
            print("-" * 40)
            content = results["final_output"]["content"]
            print(content[:1000] + "..." if len(content) > 1000 else content)

            # Quality metrics
            metrics = results["final_output"]["quality_metrics"]
            print("\n📊 Quality Metrics:")
            print(f"  • Content Quality: {metrics['content_quality']:.1%}")
            print(f"  • Fact Accuracy: {metrics['fact_accuracy']:.1%}")
            print(f"  • Source Credibility: {metrics['source_credibility']:.1%}")

            # Save to file
            save = input("\n💾 Save to file? (y/n): ").lower()
            if save == 'y':
                filename = f"research_{topic[:30].replace(' ', '_')}.md"
                with open(filename, 'w') as f:
                    f.write(results["final_output"]["content"])
                print(f"✅ Saved to {filename}")
        else:
            print(f"\n❌ Research failed: {results.get('error', 'Unknown error')}")

    # Run the async function
    asyncio.run(run_research())


def start_api():
    """Helper function to start the API server"""
    subprocess.run(["uvicorn", "intelliresearch.api:app", "--port", "8001"])

def run_both():
    """Run both Streamlit and FastAPI"""
    print("\n🎯 Starting both Web Interface and API Server...")
    print("\n📍 Web Interface: http://localhost:8501")
    print("📍 API Documentation: http://localhost:8001/docs")
    print("\nPress Ctrl+C to stop both servers\n")

    # Start API server in a separate process
    api_process = multiprocessing.Process(target=start_api)
    api_process.start()

    # Give API server time to start
    time.sleep(3)

    # Open browsers
    webbrowser.open("http://localhost:8001/docs")
    webbrowser.open("http://localhost:8501")

    try:
        # Run Streamlit in main process
        subprocess.run(["streamlit", "run", "intelliresearch/app.py"])
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        api_process.terminate()
        api_process.join()
        print("✅ All servers stopped")


def view_docs():
    """View documentation"""
    print("\n📚 Documentation Options:")
    print("1. View README")
    print("2. Open API Docs in Browser")
    print("3. View Agent Architecture")
    print("4. Back to Main Menu")

    choice = input("\nSelect option: ")

    if choice == "1":
        if Path("README.md").exists():
            with open("README.md", "r") as f:
                content = f.read()
                # Show first 100 lines
                lines = content.split("\n")[:100]
                print("\n".join(lines))
                print("\n... [README continues] ...")
        else:
            print("README.md not found!")
    elif choice == "2":
        webbrowser.open("http://localhost:8001/docs")
        print("Opening API documentation in browser...")
    elif choice == "3":
        print("\n🏗️ Agent Architecture:")
        print("""
        1. Research Coordinator - Orchestrates the process
        2. Web Scraper - Gathers information
        3. Content Analyzer - Extracts insights
        4. Fact Checker - Verifies information
        5. Writer - Generates content
        6. Editor - Refines content
        7. Citation Manager - Handles references
        """)


def run_tests():
    """Run test suite"""
    print("\n🔧 Running Tests...")
    print("-" * 40)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tests_dir = os.path.join(base_dir, "tests")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", tests_dir, "-v", "--tb=short"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed. Check output above.")
    except FileNotFoundError:
        print("❌ pytest not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest"])
        print("Please run tests again.")


def setup_docker():
    """Setup Docker environment"""
    print("\n🐳 Docker Setup")
    print("-" * 40)

    project_root = Path(__file__).resolve().parent.parent  # IntelliResearch root

    # Dockerfile content
    dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8501 8001

# Create entrypoint script
RUN echo '#!/bin/bash\\n\\
if [ "$1" = "api" ]; then\\n\\
    uvicorn intelliresearch.api:app --host 0.0.0.0 --port 8001\\n\\
elif [ "$1" = "web" ]; then\\n\\
    streamlit run intelliresearch/app.py --server.port 8501 --server.address 0.0.0.0\\n\\
else\\n\\
    python run.py\\n\\
fi' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["web"]
"""

    # Write Dockerfile to root
    dockerfile_path = project_root / "Dockerfile"
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)
    print("✅ Created/Updated Dockerfile")

    # docker-compose.yml content
    compose_content = """services:
  web:
    build:
      context: .
      dockerfile: Dockerfile
    command: web
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEMO_MODE=${DEMO_MODE:-false}
    volumes:
      - ./outputs:/app/outputs
    restart: unless-stopped

  api:
    build:
      context: .
      dockerfile: Dockerfile
    command: api
    ports:
      - "8001:8001"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEMO_MODE=${DEMO_MODE:-false}
    volumes:
      - ./outputs:/app/outputs
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
"""

    # Write docker-compose.yml to root
    compose_path = project_root / "docker-compose.yml"
    with open(compose_path, "w") as f:
        f.write(compose_content)
    print("✅ Created/Updated docker-compose.yml")

    print("\n📋 Docker commands:")
    print("  Build: docker-compose build")
    print("  Run: docker-compose up")
    print("  Run in background: docker-compose up -d")
    print("  Stop: docker-compose down")
    print("  View logs: docker-compose logs -f")

    build = input("\n🔨 Build Docker images now? (y/n): ").lower()
    if build == 'y':
        print("\n🔨 Building Docker images...")
        subprocess.run(["docker-compose", "build"], cwd=project_root)

        run = input("\n🚀 Start containers? (y/n): ").lower()
        if run == 'y':
            subprocess.run(["docker-compose", "up"], cwd=project_root)



def main():
    """Main entry point"""
    print("\n" + "=" * 60)
    print("🚀 IntelliResearch Startup Script")
    print("=" * 60)

    # Check requirements
    if not check_requirements():
        print("\n⚠️  Requirements were installed. Please run the script again.")
        return

    # Check environment
    if not check_env():
        print("\n⚠️  Please configure your environment and run again.")
        return

    while True:
        display_menu()
        choice = input("\nEnter your choice (0-7): ")

        if choice == "0":
            print("\n👋 Thank you for using IntelliResearch!")
            break
        elif choice == "1":
            run_streamlit()
        elif choice == "2":
            run_api()
        elif choice == "3":
            run_cli()
        elif choice == "4":
            run_both()
        elif choice == "5":
            view_docs()
        elif choice == "6":
            run_tests()
        elif choice == "7":
            setup_docker()
        else:
            print("❌ Invalid choice. Please try again.")

        if choice in ["1", "2", "3", "4"]:
            # After running an app, ask if user wants to continue
            cont = input("\n\nReturn to main menu? (y/n): ").lower()
            if cont != 'y':
                print("\n👋 Thank you for using IntelliResearch!")
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
        print("👋 Thank you for using IntelliResearch!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check the logs or contact support.")