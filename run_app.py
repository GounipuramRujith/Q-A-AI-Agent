#!/usr/bin/env python3
"""
Launch script for RAG Streamlit Application
Simple wrapper to run the Streamlit app with proper configuration
"""

import os
import sys
import subprocess
import argparse

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit', 'sentence_transformers', 'faiss_cpu', 
        'transformers', 'pandas', 'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('_', '-').replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def run_streamlit_app(port=8501, host="localhost"):
    """Run the Streamlit application"""
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"🚀 Starting RAG Q&A System on http://{host}:{port}")
        print("📝 To stop the application, press Ctrl+C")
        print("-" * 50)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Launch RAG Streamlit Application")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the application (default: 8501)")
    parser.add_argument("--host", type=str, default="localhost", help="Host address (default: localhost)")
    parser.add_argument("--skip-check", action="store_true", help="Skip dependency check")
    
    args = parser.parse_args()
    
    print("🔍 RAG Q&A System Launcher")
    print("=" * 40)
    
    # Check dependencies unless skipped
    if not args.skip_check:
        if not check_dependencies():
            sys.exit(1)
    
    # Run the application
    run_streamlit_app(args.port, args.host)

if __name__ == "__main__":
    main() 