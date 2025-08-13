#!/usr/bin/env python3
"""
Simple test script to verify the setup and imports work correctly.
"""
import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test basic Python imports
        import langchain
        import langchain_openai
        import langsmith
        import langgraph
        print("✅ Core LangChain packages imported successfully")
        
        # Test project imports
        from config.settings import settings
        print("✅ Settings configuration imported successfully")
        
        from src.chains.basic_chain import BasicChain
        print("✅ Basic chain imported successfully")
        
        from src.tools.calculator import calculator_tool
        print("✅ Calculator tool imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")
    
    try:
        from config.settings import settings
        
        print(f"✅ Debug mode: {settings.DEBUG}")
        print(f"✅ Log level: {settings.LOG_LEVEL}")
        print(f"✅ LangSmith tracing: {settings.LANGSMITH_TRACING}")
        print(f"✅ LangSmith project: {settings.LANGSMITH_PROJECT}")
        
        # Test API key validation (should show missing keys with test values)
        validation = settings.validate_api_keys()
        if validation["valid"]:
            print("✅ API key validation passed")
        else:
            print(f"⚠️  Missing API keys (expected with test setup): {validation['missing_keys']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without requiring real API keys."""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test calculator tool (doesn't require API keys)
        from src.tools.calculator import calculator_tool
        
        result = calculator_tool.invoke("2 + 2")
        print(f"✅ Calculator test: 2 + 2 = {result}")
        
        # Test basic chain initialization (without calling LLM)
        from src.chains.basic_chain import BasicChain
        chain = BasicChain()
        print("✅ Basic chain initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        return False

def test_project_structure():
    """Test that all expected files and directories exist."""
    print("\n📁 Testing project structure...")
    
    expected_files = [
        "README.md",
        "requirements.txt",
        "environment.yml",
        ".env",
        ".env.example",
        ".gitignore",
        "config/settings.py",
        "src/chains/basic_chain.py",
        "src/tools/calculator.py",
        "examples/basic_usage.py",
        "tests/test_config.py"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All expected files found")
    return True

def main():
    """Run all tests."""
    print("🚀 LangChain Platform Demo - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your setup is ready.")
        print("\n💡 Next steps:")
        print("1. Add your real API keys to the .env file")
        print("2. Run: python examples/basic_usage.py")
        print("3. Initialize git repo and push to GitHub")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)