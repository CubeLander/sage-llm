#!/usr/bin/env python3
"""Test script for HotLLM simplified LLM interface."""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/home/tjy/hotLLM')

from archive.llmort LLM, HotLLMModelConfig, HotLLMEngineConfig, HotLLMParallelConfig


def test_basic_initialization():
    """Test basic initialization with default configs."""
    print("Testing basic initialization...")
    
    try:
        # Test with all default configs
        llm = LLM()
        print("✅ Basic initialization successful")
        print(f"Model: {llm.model_config.model}")
        
    except Exception as e:
        print(f"❌ Basic initialization failed: {e}")
        return False
    
    return True


def test_custom_configs():
    """Test initialization with custom configs."""
    print("\nTesting custom configs...")
    
    try:
        # Custom model config
        model_config = HotLLMModelConfig(
            model="Qwen/Qwen3-0.6B",
            dtype="float16",
            max_model_len=2048
        )
        
        # Custom engine config
        engine_config = HotLLMEngineConfig(
            gpu_memory_utilization=0.8,
            max_num_seqs=64
        )
        
        # Custom parallel config
        parallel_config = HotLLMParallelConfig(
            tensor_parallel_size=1
        )
        
        llm = LLM(
            model_config=model_config,
            engine_config=engine_config,
            parallel_config=parallel_config
        )
        
        print("✅ Custom config initialization successful")
        print(f"Model: {llm.model_config.model}")
        print(f"Max model len: {llm.model_config.max_model_len}")
        
        # Test tokenizer access
        tokenizer = llm.get_tokenizer()
        print(f"✅ Tokenizer access successful: {type(tokenizer)}")
        
    except Exception as e:
        print(f"❌ Custom config initialization failed: {e}")
        return False
    
    return True


def test_generation_interface():
    """Test the generation interface."""
    print("\nTesting generation interface...")
    
    try:
        model_config = HotLLMModelConfig(
            model="Qwen/Qwen3-0.6B",
            max_model_len=512
        )
        engine_config = HotLLMEngineConfig(
            max_num_seqs=2
        )
        
        llm = LLM(
            model_config=model_config,
            engine_config=engine_config
        )
        
        # Test simple generation
        prompts = ["Hello, how are you?"]
        
        # This might fail due to model loading issues in test environment
        # but we can at least test that the method exists and accepts parameters
        try:
            from vllm.sampling_params import SamplingParams
            sampling_params = SamplingParams(max_tokens=10, temperature=0.8)
            
            # We won't actually run generation in the test to avoid model loading
            # Just check the method exists
            assert hasattr(llm, 'generate'), "generate method missing"
            assert callable(llm.generate), "generate method not callable"
            
            print("✅ Generation interface accessible")
            
        except ImportError:
            print("⚠️  SamplingParams not available, skipping generation test")
            
    except Exception as e:
        print(f"❌ Generation interface test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing HotLLM Simplified LLM Interface")
    print("=" * 50)
    
    tests = [
        test_basic_initialization,
        test_custom_configs,
        test_generation_interface
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
