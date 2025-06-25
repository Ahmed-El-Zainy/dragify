#!/usr/bin/env python3
"""
Dragify AI Agent Template - Demo Launcher
Simple script to launch the Gradio demo with proper configuration
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['gradio', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def setup_environment():
    """Setup environment variables for demo"""
    env_vars = {
        'GRADIO_ANALYTICS_ENABLED': 'False',
        'GRADIO_SERVER_NAME': '0.0.0.0',
        'GRADIO_SERVER_PORT': '7860',
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value

def launch_demo(share=False, debug=False, port=7860):
    """Launch the Gradio demo"""
    print("🚀 Starting Dragify AI Agent Demo...")
    print(f"📡 Demo will be available at: http://localhost:{port}")
    
    if share:
        print("🌐 Public sharing enabled - you'll get a public URL")
    
    # Import and run the demo
    try:
        # Set up the demo configuration
        demo_config = {
            'server_port': port,
            'share': share,
            'debug': debug,
            'show_api': False,
            'favicon_path': None,
            'ssl_verify': False,
            'quiet': not debug
        }
        
        # Import the demo
        from backend.app.launch_gradio import interface
        
        # Launch the interface
        interface.launch(**demo_config)
        
    except ImportError as e:
        print(f"❌ Error importing demo: {e}")
        print("Make sure dragify_gradio_demo.py is in the same directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching demo: {e}")
        sys.exit(1)

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Launch Dragify AI Agent Template Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py                    # Launch locally
  python launcher.py --share           # Launch with public sharing
  python launcher.py --port 8080       # Launch on custom port
  python launcher.py --debug --share   # Launch in debug mode with sharing
        """
    )
    
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public shareable link'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to run the demo on (default: 7860)'
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check requirements and exit'
    )
    
    args = parser.parse_args()
    
    # Check requirements
    if not check_requirements():
        if args.check:
            sys.exit(1)
        
        print("\n🔧 Attempting to install missing packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gradio', 'pandas', 'numpy'])
            print("✅ Packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages automatically")
            print("Please install manually: pip install gradio pandas numpy")
            sys.exit(1)
    
    if args.check:
        print("✅ All requirements satisfied!")
        sys.exit(0)
    
    # Setup environment
    setup_environment()
    
    # Print banner
    print("=" * 60)
    print("🤖 DRAGIFY AI AGENT TEMPLATE - GRADIO DEMO")
    print("=" * 60)
    print("Interactive demo showcasing modular AI agent automation")