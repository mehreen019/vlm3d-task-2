#!/usr/bin/env python3
"""
Quick Performance Test for VLM3D Task 2
Tests different configurations to find the best setup for your dataset
"""

import os
import subprocess
import time

def run_test(name, command, description):
    """Run a test configuration"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {name}")
    print(f"📝 {description}")
    print(f"💻 Command: {command}")
    print('='*60)

    start_time = time.time()

    try:
        # Run the command
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)

        # Calculate runtime
        runtime = time.time() - start_time
        print(".1f"        print("✅ Test completed successfully!")

        return True

    except subprocess.CalledProcessError as e:
        runtime = time.time() - start_time
        print(".1f"        print(f"❌ Test failed: {e}")
        print(f"🔍 Check the output above for error details")

        return False

def main():
    print("🚀 VLM3D Task 2 Performance Optimization Test")
    print("This script tests different configurations to improve your model's performance")
    print("=" * 80)

    # Test configurations (short epochs for testing)
    tests = [
        {
            "name": "Baseline (Current Config)",
            "command": "python run_task2.py --model efficientnet_b0 --epochs 5 --batch-size 64",
            "description": "Your current setup for comparison"
        },
        {
            "name": "Asymmetric Loss (Quick Fix)",
            "command": "python run_task2.py --model efficientnet_b0 --loss-type asl --freeze-backbone --epochs 5 --batch-size 64",
            "description": "Best quick fix for imbalanced data - should show immediate improvement"
        },
        {
            "name": "Focal Loss + SE Attention",
            "command": "python run_task2.py --model efficientnet_b0 --loss-type focal --use-attention se --freeze-backbone --epochs 5 --batch-size 64",
            "description": "Focal loss with Squeeze-Excitation attention"
        },
        {
            "name": "Progressive Unfreezing",
            "command": "python run_task2.py --model efficientnet_b0 --loss-type asl --progressive-unfreeze --unfreeze-epoch 2 --epochs 5 --batch-size 32",
            "description": "Gradual backbone unfreezing for better feature learning"
        }
    ]

    print(f"📊 Will run {len(tests)} test configurations")
    print("💡 Each test uses short epochs (5) for quick evaluation")
    print("🎯 Compare AUROC, Hamming Accuracy, and Positive Prediction Rate")
    print()

    # Ask user if they want to proceed
    response = input("🔥 Ready to start testing? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Test cancelled. Run again when ready!")
        return

    # Run tests
    results = []
    for i, test in enumerate(tests, 1):
        print(f"\n🎯 Test {i}/{len(tests)}")
        success = run_test(test["name"], test["command"], test["description"])
        results.append((test["name"], success))

        # Wait a bit between tests
        if i < len(tests):
            print("⏳ Waiting 10 seconds before next test...")
            time.sleep(10)

    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print('='*60)

    successful_tests = sum(1 for _, success in results if success)
    print(f"✅ Successful tests: {successful_tests}/{len(results)}")

    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status}: {name}")

    print(f"\n{'='*60}")
    print("🎯 RECOMMENDATIONS")
    print('='*60)

    if successful_tests > 0:
        print("📈 Based on successful tests, try these full training runs:")
        print()
        print("1️⃣ BEST QUICK FIX:")
        print("   python run_task2.py --model efficientnet_b0 --loss-type asl --freeze-backbone --epochs 100")
        print()
        print("2️⃣ BEST COMPREHENSIVE:")
        print("   python run_task2.py --model efficientnet_b0 --loss-type asl --progressive-unfreeze --unfreeze-epoch 15 --use-attention cbam --epochs 150")
        print()
        print("3️⃣ ENSEMBLE APPROACH:")
        print("   Train multiple models with different configs for best results")
    else:
        print("❌ No tests passed. Check your environment and try again.")
        print("🔧 Common issues:")
        print("   - GPU memory issues (reduce batch size)")
        print("   - Missing dependencies (run setup)")
        print("   - Data path issues (check ct_rate_data/)")

    print(f"\n{'='*60}")
    print("📚 NEXT STEPS")
    print('='*60)
    print("1. 📖 Read PERFORMANCE_OPTIMIZATION_GUIDE.md for detailed analysis")
    print("2. 🎯 Choose the best configuration from test results")
    print("3. 🚀 Run full training with selected configuration")
    print("4. 📊 Evaluate results and iterate")
    print()
    print("🎉 Happy training!")

if __name__ == "__main__":
    main()
