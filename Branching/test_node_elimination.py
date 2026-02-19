"""
Test script to compare performance with and without node elimination strategy.
Runs the main algorithm twice and logs timing and solution quality metrics.
"""

from main import main
import time
from datetime import datetime

def run_comparison():
    """Run the algorithm with and without node elimination and compare results."""
    
    dataset = "C:\\Users\\CiSTUP\\Downloads\\VRTPW_Dataset\\solomon_100_customer_instances\\c102.txt"
    
    # Create results log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"node_elimination_comparison_{timestamp}.txt"
    
    with open(log_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("NODE ELIMINATION STRATEGY COMPARISON\n")
        f.write("="*80 + "\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
    
    print("="*80)
    print("NODE ELIMINATION STRATEGY COMPARISON")
    print("="*80)
    print(f"Dataset: {dataset}\n")
    
    # Test 1: WITHOUT Node Elimination (but with other strategies)
    print("\n" + "="*80)
    print("TEST 1: ALL STRATEGIES EXCEPT NODE ELIMINATION")
    print("="*80 + "\n")
    
    with open(log_file, 'a') as f:
        f.write("TEST 1: ALL STRATEGIES EXCEPT NODE ELIMINATION\n")
        f.write("(Column Deletion + Early Stop Pricing)\n")
        f.write("-"*80 + "\n")
    
    test1_start = time.time()
    main(datasetPath=dataset, 
         SHOWFIG=False,
         enable_column_deletion=True,
         enable_early_stop_pricing=True,  # ENABLED
         enable_node_elimination=False)  # DISABLED
    test1_end = time.time()
    test1_time = test1_end - test1_start
    
    print(f"\n\nTest 1 Total Runtime: {test1_time:.2f} seconds\n")
    
    with open(log_file, 'a') as f:
        f.write(f"Total Runtime: {test1_time:.2f} seconds\n")
        f.write("\n")
    
    # Test 2: WITH Node Elimination (and all other strategies)
    print("\n" + "="*80)
    print("TEST 2: ALL STRATEGIES INCLUDING NODE ELIMINATION")
    print("="*80 + "\n")
    
    with open(log_file, 'a') as f:
        f.write("TEST 2: ALL STRATEGIES INCLUDING NODE ELIMINATION\n")
        f.write("(Column Deletion + Early Stop Pricing + Node Elimination threshold=1.5)\n")
        f.write("-"*80 + "\n")
    
    test2_start = time.time()
    main(datasetPath=dataset, 
         SHOWFIG=False,
         enable_column_deletion=True,
         enable_early_stop_pricing=True,  # ENABLED
         enable_node_elimination=True,   # ENABLED
         node_elimination_threshold=1.5)
    test2_end = time.time()
    test2_time = test2_end - test2_start
    
    print(f"\n\nTest 2 Total Runtime: {test2_time:.2f} seconds\n")
    
    with open(log_file, 'a') as f:
        f.write(f"Total Runtime: {test2_time:.2f} seconds\n")
        f.write("\n")
    
    # Summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"Time WITHOUT Node Elimination:  {test1_time:.2f} seconds")
    print(f"Time WITH Node Elimination:     {test2_time:.2f} seconds")
    speedup = test1_time / test2_time
    time_saved = test1_time - test2_time
    print(f"Speedup Factor:                 {speedup:.2f}x")
    print(f"Time Saved:                     {time_saved:.2f} seconds ({(time_saved/test1_time*100):.1f}%)")
    print("="*80)
    
    with open(log_file, 'a') as f:
        f.write("COMPARISON SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Time WITHOUT Node Elimination:  {test1_time:.2f} seconds\n")
        f.write(f"Time WITH Node Elimination:     {test2_time:.2f} seconds\n")
        f.write(f"Speedup Factor:                 {speedup:.2f}x\n")
        f.write(f"Time Saved:                     {time_saved:.2f} seconds ({(time_saved/test1_time*100):.1f}%)\n")
        f.write("="*80 + "\n")
    
    print(f"\nResults logged to: {log_file}")

if __name__ == "__main__":
    run_comparison()
