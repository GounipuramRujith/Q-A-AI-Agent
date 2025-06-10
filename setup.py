"""
Setup script for MusicRAG - Enhanced Temple Information RAG System
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Need Python 3.8+")
        return False

def create_virtual_environment():
    """Create virtual environment"""
    venv_name = "temple_rag_env"
    if not os.path.exists(venv_name):
        return run_command(f"python -m venv {venv_name}", "Creating virtual environment")
    else:
        print(f"✅ Virtual environment '{venv_name}' already exists")
        return True

def install_requirements():
    """Install required packages"""
    if platform.system() == "Windows":
        pip_command = "temple_rag_env\\Scripts\\pip"
    else:
        pip_command = "temple_rag_env/bin/pip"
    
    return run_command(f"{pip_command} install -r requirements.txt", "Installing requirements")

def validate_data():
    """Run data validation"""
    if platform.system() == "Windows":
        python_command = "temple_rag_env\\Scripts\\python"
    else:
        python_command = "temple_rag_env/bin/python"
    
    print("🔍 Running data validation...")
    try:
        result = subprocess.run(f"{python_command} data_validator.py", 
                              shell=True, capture_output=True, text=True)
        if "PASSED" in result.stdout:
            print("✅ Data validation passed")
            return True
        else:
            print("⚠️ Data validation completed with warnings")
            print(result.stdout)
            return True
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        return False

def check_files():
    """Check if required files exist"""
    required_files = [
        "temples_optimized.csv",
        "requirements.txt",
        "app.py",
        "rag_pipeline.py",
        "data_handler.py",
        "config.py",
        "data_validator.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    else:
        print("✅ All required files found")
        return True

def main():
    """Main setup function"""
    print("🏛️ MusicRAG - Enhanced Temple Information RAG System Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check required files
    if not check_files():
        print("Please ensure all required files are present before running setup.")
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Run data validation
    validate_data()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    if platform.system() == "Windows":
        print("1. Activate virtual environment: temple_rag_env\\Scripts\\activate")
    else:
        print("1. Activate virtual environment: source temple_rag_env/bin/activate")
    print("2. Run the application: streamlit run app.py")
    print("3. Open browser to: http://localhost:8501")
    print("\n🔗 For detailed instructions, see README_Enhanced.md")

if __name__ == "__main__":
    main() 