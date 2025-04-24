#!/usr/bin/env python
# run_test.py - Simple script to run the RAG core tests

import unittest
import os
import sys
from unittest.mock import patch

# Ensure we're working in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Mock environment variables
os.environ["OPENAI_API_KEY"] = "fake_key_for_testing"
os.environ["PINECONE_API_KEY"] = "fake_key_for_testing"

print("=== Testing RAG Core Functionality ===")
print("Note: This uses mocks and doesn't require API access")

# Run the tests with the unittest module
from test_rag_core import TestRAGCore

if __name__ == "__main__":
    # Create a test suite with our tests
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestRAGCore)
    
    # Run the tests
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    # Return non-zero exit code if tests failed
    sys.exit(not test_result.wasSuccessful()) 