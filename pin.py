#!/usr/bin/env python3
"""
Enhanced 2D Bin Packing Optimizer with Python-C++ Integration
Author: abdelbar472
Date: 2025-09-02 06:26:39 UTC
System: Intel i5-6440HQ @ 2.60GHz, 16GB RAM
Location: D:\codes\2d_pin
"""

import os
import sys
import json
import time
import subprocess
import threading
import tempfile
from pathlib import Path
import argparse
from typing import List, Tuple, Optional, Dict, Any, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from tqdm import tqdm
import random
import math
from copy import deepcopy
import psutil
import platform

# Import original classes from pin.py
from pin import Item, Plank, greedy_packing, genetic_algorithm, simulated_annealing, visualize_planks, save_layout

class SystemInfo:
    """System information and optimization detection."""
    
    @staticmethod
    def get_system_info():
        """Get detailed system information."""
        info = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "python_version": platform.python_version()
        }
        return info
    
    @staticmethod
    def detect_optimizations():
        """Detect available optimizations."""
        optimizations = {
            "multiprocessing": True,
            "numpy_available": True,
            "cpp_compiler": False,
            "openmp": False
        }
        
        # Check for C++ compiler
        try:
            result = subprocess.run(["g++", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                optimizations["cpp_compiler"] = True
                # Check for OpenMP support
                if "openmp" in result.stdout.lower() or "-fopenmp" in result.stdout:
                    optimizations["openmp"] = True
        except FileNotFoundError:
            try:
                result = subprocess.run(["cl"], capture_output=True, text=True)
                if result.returncode == 0:
                    optimizations["cpp_compiler"] = True
            except FileNotFoundError:
                pass
        
        return optimizations

class PerformanceMonitor:
    """Monitor performance metrics during optimization."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.start_time = None
        self.end_time = None
        self.memory_usage = []
        self.cpu_usage = []
        self.iteration_times = []
        self.best_scores = []
        self.current_iteration = 0
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        self.end_time = time.time()
    
    def _monitor_resources(self):
        """Background thread to monitor system resources."""
        while getattr(self, 'monitoring', False):
            try:
                self.memory_usage.append(psutil.virtual_memory().percent)
                self.cpu_usage.append(psutil.cpu_percent(interval=0.1))
                time.sleep(0.5)
            except:
                break
    
    def log_iteration(self, score: float, iteration_time: float = None):
        """Log iteration results."""
        self.current_iteration += 1
        self.best_scores.append(score)
        if iteration_time:
            self.iteration_times.append(iteration_time)
    
    def get_duration(self) -> float:
        """Get total execution duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "duration": self.get_duration(),
            "iterations": self.current_iteration,
            "avg_memory_usage": np.mean(self.memory_usage) if self.memory_usage else 0,
            "max_memory_usage": np.max(self.memory_usage) if self.memory_usage else 0,
            "avg_cpu_usage": np.mean(self.cpu_usage) if self.cpu_usage else 0,
            "max_cpu_usage": np.max(self.cpu_usage) if self.cpu_usage else 0,
            "avg_iteration_time": np.mean(self.iteration_times) if self.iteration_times else 0,
            "best_final_score": self.best_scores[-1] if self.best_scores else 0,
            "convergence_rate": self._calculate_convergence_rate()
        }
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate."""
        if len(self.best_scores) < 2:
            return 0.0
        
        improvements = 0
        for i in range(1, len(self.best_scores)):
            if self.best_scores[i] > self.best_scores[i-1]:
                improvements += 1
        
        return improvements / (len(self.best_scores) - 1) if len(self.best_scores) > 1 else 0.0

class CppExecutor:
    """Execute C++ implementation with optimized compilation."""
    
    def __init__(self, cpp_file: str = "main.cpp", exe_name: str = "main.exe"):
        self.cpp_file = Path(cpp_file)
        self.exe_name = exe_name
        self.exe_path = Path(exe_name)
        self.compiled = False
        self.compilation_flags = self._get_optimal_flags()
    
    def _get_optimal_flags(self) -> List[str]:
        """Get optimal compilation flags for the system."""
        flags = [
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "-march=native",
            "-mtune=native"
        ]
        
        # Intel-specific optimizations for i5-6440HQ (Skylake)
        intel_flags = [
            "-msse4.2",
            "-mavx",
            "-mavx2", 
            "-mfma"
        ]
        
        # Check if we're on Windows with MSVC or MinGW
        system_info = SystemInfo.get_system_info()
        if system_info["platform"] == "Windows":
            # Try to detect compiler
            try:
                result = subprocess.run(["g++", "--version"], capture_output=True)
                if result.returncode == 0:
                    # MinGW/GCC
                    flags.extend(intel_flags)
                    flags.extend(["-static", "-s"])
            except:
                # Try MSVC
                try:
                    result = subprocess.run(["cl"], capture_output=True)
                    if result.returncode == 0:
                        # MSVC flags
                        flags = ["/O2", "/GL", "/arch:AVX2", "/favor:INTEL64"]
                except:
                    pass
        else:
            flags.extend(intel_flags)
        
        return flags
    
    def compile_if_needed(self, force: bool = False) -> bool:
        """Compile C++ code if needed."""
        if self.compiled and not force and self.exe_path.exists():
            return True
        
        if not self.cpp_file.exists():
            print(f"Error: C++ file {self.cpp_file} not found")
            return False
        
        print(f"Compiling {self.cpp_file} with optimizations for Intel i5-6440HQ...")
        
        try:
            # Try GCC first
            cmd = ["g++"] + self.compilation_flags + [str(self.cpp_file), "-o", self.exe_name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✅ C++ compilation successful with GCC")
                self.compiled = True
                return True
            else:
                print(f"GCC compilation failed: {result.stderr}")
                
                # Try simpler flags
                simple_cmd = ["g++", "-O2", str(self.cpp_file), "-o", self.exe_name]
                result = subprocess.run(simple_cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    print("✅ C++ compilation successful with basic optimization")
                    self.compiled = True
                    return True
                
        except subprocess.TimeoutExpired:
            print("❌ Compilation timed out")
        except FileNotFoundError:
            print("❌ GCC compiler not found")
        
        return False
    
    def run_with_input(self, input_data: Dict[str, Any], timeout: int = 60) -> Optional[Dict[str, Any]]:
        """Run C++ executable with JSON input."""
        if not self.compile_if_needed():
            return None
        
        if not self.exe_path.exists():
            print(f"❌ Executable {self.exe_path} not found")
            return None
        
        try:
            # Create temporary input file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(input_data, f, indent=2)
                temp_input = f.name
            
            # Run executable
            cmd = [str(self.exe_path)]
            result = subprocess.run(
                cmd,
                input=json.dumps(input_data),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Clean up
            try:
                os.unlink(temp_input)
            except:
                pass
            
            if result.returncode == 0:
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse C++ output: {e}")
                    print(f"Raw output: {result.stdout[:500]}...")
                    return None
            else:
                print(f"❌ C++ execution failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"❌ C++ execution timed out after {timeout} seconds")
            return None
        except Exception as e:
            print(f"❌ Error running C++ executable: {e}")
            return None

class HybridOptimizer:
    """Hybrid optimizer combining Python and C++ implementations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.cpp_executor = CppExecutor()
        self.performance_monitor = PerformanceMonitor()
        self.system_info = SystemInfo.get_system_info()
        self.optimizations = SystemInfo.detect_optimizations()
        
        print(f"🖥️  System: {self.system_info['processor']}")
        print(f"💾 Memory: {self.system_info['memory_gb']} GB")
        print(f"🔧 CPU Cores: {self.system_info['cpu_count']} physical, {self.system_info['cpu_count_logical']} logical")
        print(f"🚀 C++ Compiler: {'✅' if self.optimizations['cpp_compiler'] else '❌'}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get optimized default configuration."""
        return {
            "plank": {"length": 240, "width": 120},
            "prefer_cpp": True,
            "parallel": True,
            "algorithms": {
                "greedy": {"step": 1},
                "genetic": {
                    "population": 50,  # Increased for better results
                    "generations": 100,  # Increased for convergence
                    "mutation_rate": 0.15,
                    "tournament_size": 3
                },
                "simulated_annealing": {
                    "initial_temp": 1000,
                    "cooling_rate": 0.95,
                    "iterations": 500  # Increased for thorough search
                }
            },
            "visualization": {
                "show_progress": True,
                "save_plots": True,
                "plot_convergence": True,
                "animation": False
            },
            "output": {
                "save_layouts": True,
                "formats": ["json", "txt", "csv"]
            }
        }
    
    def optimize_single_algorithm(self, items: List[Item], algorithm: str, 
                                 use_cpp: bool = None) -> Tuple[List[Plank], Dict[str, Any]]:
        """Run single algorithm with performance monitoring."""
        if use_cpp is None:
            use_cpp = self.config["prefer_cpp"] and self.optimizations["cpp_compiler"]
        
        self.performance_monitor.reset()
        self.performance_monitor.start_monitoring()
        
        start_time = time.time()
        planks = []
        implementation = "Unknown"
        
        try:
            if use_cpp and self.cpp_executor.compile_if_needed():
                # Try C++ implementation
                print(f"🚀 Running {algorithm} with C++ implementation...")
                planks, implementation = self._run_cpp_algorithm(items, algorithm)
                
                if not planks:
                    print("⚠️  C++ failed, falling back to Python...")
                    planks, implementation = self._run_python_algorithm(items, algorithm)
            else:
                # Use Python implementation
                print(f"🐍 Running {algorithm} with Python implementation...")
                planks, implementation = self._run_python_algorithm(items, algorithm)
        
        except Exception as e:
            print(f"❌ Algorithm failed: {e}")
            planks = []
            implementation = "Failed"
        
        end_time = time.time()
        self.performance_monitor.stop_monitoring()
        
        # Calculate metrics
        execution_time = end_time - start_time
        total_area = sum(item.length * item.width * item.quantity for item in items)
        plank_area = self.config["plank"]["length"] * self.config["plank"]["width"]
        theoretical_min = math.ceil(total_area / plank_area)
        efficiency = (theoretical_min / len(planks)) * 100 if planks else 0
        packing_efficiency = sum(plank.get_used_area() for plank in planks) / (len(planks) * plank_area) * 100 if planks else 0
        
        metrics = {
            "algorithm": algorithm,
            "implementation": implementation,
            "execution_time": execution_time,
            "planks_used": len(planks),
            "theoretical_minimum": theoretical_min,
            "algorithm_efficiency": efficiency,
            "packing_efficiency": packing_efficiency,
            "items_total": sum(item.quantity for item in items),
            "performance": self.performance_monitor.get_stats()
        }
        
        return planks, metrics
    
    def _run_cpp_algorithm(self, items: List[Item], algorithm: str) -> Tuple[List[Plank], str]:
        """Run C++ algorithm implementation."""
        # Prepare input data
        input_data = {
            "algorithm": algorithm,
            "plank": self.config["plank"],
            "items": [
                {
                    "name": item.name,
                    "length": item.length,
                    "width": item.width,
                    "quantity": item.quantity
                }
                for item in items
            ],
            "parameters": self.config["algorithms"][algorithm]
        }
        
        # Run C++ executable
        result = self.cpp_executor.run_with_input(input_data, timeout=120)
        
        if result and "planks" in result:
            # Convert C++ output to Python objects
            planks = self._parse_cpp_result(result)
            return planks, "C++"
        
        return [], "C++ (Failed)"
    
    def _run_python_algorithm(self, items: List[Item], algorithm: str) -> Tuple[List[Plank], str]:
        """Run Python algorithm implementation."""
        plank_length = self.config["plank"]["length"]
        plank_width = self.config["plank"]["width"]
        
        if algorithm == "greedy":
            planks = greedy_packing(items, plank_length, plank_width, 
                                  step=self.config["algorithms"]["greedy"]["step"])
        
        elif algorithm == "genetic":
            params = self.config["algorithms"]["genetic"]
            planks = genetic_algorithm(items, plank_length, plank_width, 
                                     pop_size=params["population"],
                                     generations=params["generations"],
                                     mutation_rate=params["mutation_rate"])
        
        elif algorithm == "simulated_annealing":
            params = self.config["algorithms"]["simulated_annealing"]
            planks = simulated_annealing(items, plank_length, plank_width,
                                       initial_temp=params["initial_temp"],
                                       cooling_rate=params["cooling_rate"],
                                       iterations=params["iterations"])
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return planks, "Python"
    
    def _parse_cpp_result(self, result: Dict[str, Any]) -> List[Plank]:
        """Parse C++ JSON result into Python Plank objects."""
        planks = []
        
        if "planks" not in result:
            return planks
        
        for plank_data in result["planks"]:
            plank = Plank(
                length=plank_data["dimensions"]["length"],
                width=plank_data["dimensions"]["width"]
            )
            
            for item_data in plank_data["items"]:
                # Reconstruct the item placement tuple
                placement = (
                    item_data["position"]["x"],
                    item_data["position"]["y"],
                    item_data["size"]["length"], 
                    item_data["size"]["width"],
                    item_data["name"],
                    item_data["rotated"]
                )
                plank.items.append(placement)
            
            planks.append(plank)
        
        return planks
    
    def compare_all_algorithms(self, items: List[Item]) -> Dict[str, Tuple[List[Plank], Dict[str, Any]]]:
        """Compare all algorithms with comprehensive analysis."""
        algorithms = ["greedy", "genetic", "simulated_annealing"]
        results = {}
        
        print(f"\n{'='*70}")
        print("🔥 ENHANCED 2D BIN PACKING OPTIMIZATION COMPARISON")
        print(f"{'='*70}")
        print(f"📦 Items to pack: {len(items)} types, {sum(item.quantity for item in items)} total pieces")
        print(f"📏 Plank size: {self.config['plank']['length']} × {self.config['plank']['width']} cm")
        print(f"💻 System: {self.system_info['processor']}")
        
        # Calculate theoretical minimum
        total_area = sum(item.length * item.width * item.quantity for item in items)
        plank_area = self.config["plank"]["length"] * self.config["plank"]["width"]
        theoretical_min = math.ceil(total_area / plank_area)
        print(f"🎯 Theoretical minimum: {theoretical_min} planks")
        print()
        
        # Run each algorithm
        for algorithm in algorithms:
            print(f"⚡ Running {algorithm.upper()}...")
            planks, metrics = self.optimize_single_algorithm(items, algorithm)
            results[algorithm] = (planks, metrics)
            
            # Print immediate results
            print(f"   📊 Result: {len(planks)} planks ({metrics['implementation']} - {metrics['execution_time']:.3f}s)")
            print(f"   🎯 Efficiency: {metrics['algorithm_efficiency']:.1f}% vs theoretical minimum")
            print(f"   📦 Packing: {metrics['packing_efficiency']:.1f}% of plank area used")
            print()
        
        return results
    
    def print_detailed_comparison(self, results: Dict[str, Tuple[List[Plank], Dict[str, Any]]]):
        """Print detailed comparison table."""
        print(f"{'='*90}")
        print("📈 DETAILED PERFORMANCE COMPARISON")
        print(f"{'='*90}")
        
        # Table header
        header = f"{'Algorithm':<15} {'Impl.':<6} {'Planks':<7} {'Time(s)':<8} {'Alg.Eff':<8} {'Pack.Eff':<9} {'Memory':<8} {'CPU':<6}"
        print(header)
        print("-" * len(header))
        
        # Sort by number of planks (best first)
        sorted_results = sorted(results.items(), key=lambda x: len(x[1][0]))
        
        for algorithm, (planks, metrics) in sorted_results:
            perf = metrics['performance']
            row = (f"{algorithm:<15} "
                   f"{metrics['implementation']:<6} "
                   f"{len(planks):<7} "
                   f"{metrics['execution_time']:<8.3f} "
                   f"{metrics['algorithm_efficiency']:<8.1f}% "
                   f"{metrics['packing_efficiency']:<9.1f}% "
                   f"{perf['avg_memory_usage']:<8.1f}% "
                   f"{perf['avg_cpu_usage']:<6.1f}%")
            print(row)
        
        print()
        
        # Find best algorithm
        best_alg = min(results.keys(), key=lambda k: len(results[k][0]))
        best_count = len(results[best_alg][0])
        best_metrics = results[best_alg][1]
        
        print(f"🏆 WINNER: {best_alg.upper()}")
        print(f"   📦 Planks: {best_count}")
        print(f"   ⏱️  Time: {best_metrics['execution_time']:.3f} seconds")
        print(f"   🔧 Implementation: {best_metrics['implementation']}")
        print(f"   🎯 Efficiency: {best_metrics['algorithm_efficiency']:.1f}% of theoretical optimum")
    
    def save_comprehensive_results(self, results: Dict[str, Tuple[List[Plank], Dict[str, Any]]], 
                                  output_dir: str = "results"):
        """Save comprehensive results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        for algorithm, (planks, metrics) in results.items():
            base_name = f"{algorithm}_{timestamp}"
            
            # Save layout in multiple formats
            if "json" in self.config["output"]["formats"]:
                self._save_json_layout(planks, metrics, output_path / f"{base_name}.json")
            
            if "txt" in self.config["output"]["formats"]:
                save_layout(planks, str(output_path / f"{base_name}.txt"))
            
            if "csv" in self.config["output"]["formats"]:
                self._save_csv_layout(planks, output_path / f"{base_name}.csv")
        
        # Save comparison summary
        self._save_comparison_summary(results, output_path / f"comparison_{timestamp}.json")
        print(f"📁 Results saved to: {output_path.absolute()}")
    
    def _save_json_layout(self, planks: List[Plank], metrics: Dict[str, Any], filename: Path):
        """Save layout in JSON format with metadata."""
        data = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "system": self.system_info,
                "config": self.config,
                "metrics": metrics
            },
            "summary": {
                "planks_count": len(planks),
                "total_efficiency": sum(plank.get_efficiency() for plank in planks) / len(planks) if planks else 0,
                "total_used_area": sum(plank.get_used_area() for plank in planks),
                "total_available_area": len(planks) * self.config["plank"]["length"] * self.config["plank"]["width"]
            },
            "planks": [plank.to_dict() for plank in planks]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_csv_layout(self, planks: List[Plank], filename: Path):
        """Save layout in CSV format."""
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(["Plank_ID", "Item_Name", "X", "Y", "Length", "Width", "Rotated", "Area"])
            
            # Data
            for plank_id, plank in enumerate(planks, 1):
                for x, y, length, width, name, rotated in plank.items:
                    writer.writerow([plank_id, name, x, y, length, width, rotated, length * width])
    
    def _save_comparison_summary(self, results: Dict[str, Tuple[List[Plank], Dict[str, Any]]], filename: Path):
        """Save comparison summary."""
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "system_info": self.system_info,
            "configuration": self.config,
            "algorithms": {}
        }
        
        for algorithm, (planks, metrics) in results.items():
            summary["algorithms"][algorithm] = {
                "planks_used": len(planks),
                "metrics": metrics,
                "layouts": [plank.to_dict() for plank in planks[:3]]  # Save first 3 planks as examples
            }
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def create_enhanced_visualization(self, results: Dict[str, Tuple[List[Plank], Dict[str, Any]]]):
        """Create enhanced visualization with multiple views."""
        if not results:
            return
        
        # Create a comprehensive figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Algorithm comparison bar chart
        ax1 = fig.add_subplot(gs[0, :2])
        algorithms = list(results.keys())
        planks_counts = [len(results[alg][0]) for alg in algorithms]
        execution_times = [results[alg][1]['execution_time'] for alg in algorithms]
        
        x_pos = np.arange(len(algorithms))
        bars = ax1.bar(x_pos, planks_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Planks Used')
        ax1.set_title('Algorithm Comparison - Planks Used')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([alg.title() for alg in algorithms])
        
        # Add value labels on bars
        for bar, value in zip(bars, planks_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # 2. Execution time comparison
        ax2 = fig.add_subplot(gs[0, 2:])
        bars2 = ax2.bar(x_pos, execution_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Execution Time (s)')
        ax2.set_title('Algorithm Comparison - Execution Time')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([alg.title() for alg in algorithms])
        
        for bar, value in zip(bars2, execution_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}s', ha='center', va='bottom', fontweight='bold')
        
        # 3. Efficiency comparison radar chart
        ax3 = fig.add_subplot(gs[1, :2], projection='polar')
        metrics_names = ['Algorithm\nEfficiency', 'Packing\nEfficiency', 'Speed\n(inverse)', 'Memory\nEfficiency']
        
        for i, (algorithm, (planks, metrics)) in enumerate(results.items()):
            # Normalize metrics to 0-100 scale
            alg_eff = metrics['algorithm_efficiency']
            pack_eff = metrics['packing_efficiency'] 
            speed_eff = 100 / (metrics['execution_time'] + 0.001)  # Inverse of time
            mem_eff = 100 - metrics['performance']['avg_memory_usage']
            
            values = [alg_eff, pack_eff, min(speed_eff, 100), mem_eff]
            
            angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False)
            values += values[:1]  # Complete the circle
            angles = np.concatenate((angles, [angles[0]]))
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            ax3.plot(angles, values, 'o-', linewidth=2, label=algorithm.title(), color=colors[i])
            ax3.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(metrics_names)
        ax3.set_ylim(0, 100)
        ax3.set_title('Performance Radar Chart')
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 4. Best result layout visualization
        best_alg = min(results.keys(), key=lambda k: len(results[k][0]))
        best_planks = results[best_alg][0]
        
        if best_planks:
            # Show first 2 planks of best result
            for i, plank in enumerate(best_planks[:2]):
                ax = fig.add_subplot(gs[1, 2 + i])
                
                ax.set_xlim(0, plank.length)
                ax.set_ylim(0, plank.width)
                ax.set_title(f'Best Result - Plank {i+1}\n{best_alg.title()}')
                ax.set_aspect('equal')
                
                # Draw plank boundary
                boundary = patches.Rectangle((0, 0), plank.length, plank.width, 
                                           linewidth=2, edgecolor='black', 
                                           facecolor='lightgray', alpha=0.3)
                ax.add_patch(boundary)
                
                # Draw items
                colors = plt.cm.Set3(np.linspace(0, 1, 12))
                for j, (x, y, l, w, name, rotated) in enumerate(plank.items):
                    color = colors[j % len(colors)]
                    rect = patches.Rectangle((x, y), l, w, linewidth=1,
                                           edgecolor='black', facecolor=color, alpha=0.7)
                    ax.add_patch(rect)
                    
                    # Add label
                    ax.text(x + l/2, y + w/2, f"{name}\n{l}×{w}" + (" (R)" if rotated else ""),
                           ha='center', va='center', fontsize=6, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                
                ax.grid(True, alpha=0.3)
        
        # 5. Performance metrics table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('tight')
        ax5.axis('off')
        
        # Create table data
        table_data = []
        headers = ['Algorithm', 'Implementation', 'Planks', 'Time (s)', 'Alg. Eff. (%)', 'Pack. Eff. (%)', 'Memory (%)', 'CPU (%)']
        table_data.append(headers)
        
        for algorithm, (planks, metrics) in results.items():
            perf = metrics['performance']
            row = [
                algorithm.title(),
                metrics['implementation'],
                len(planks),
                f"{metrics['execution_time']:.3f}",
                f"{metrics['algorithm_efficiency']:.1f}",
                f"{metrics['packing_efficiency']:.1f}",
                f"{perf['avg_memory_usage']:.1f}",
                f"{perf['avg_cpu_usage']:.1f}"
            ]
            table_data.append(row)
        
        table = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color code the best values
        best_plank_count = min(len(results[alg][0]) for alg in results.keys())
        for i, (algorithm, (planks, metrics)) in enumerate(results.items(), 1):
            if len(planks) == best_plank_count:
                table[(i, 2)].set_facecolor('#90EE90')  # Light green for best plank count
        
        plt.suptitle('Enhanced 2D Bin Packing Optimization Analysis\n' + 
                    f'Intel i5-6440HQ @ 2.60GHz - {time.strftime("%Y-%m-%d %H:%M:%S")}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if self.config["visualization"]["save_plots"]:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'comprehensive_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        
        plt.show()

def get_interactive_items() -> List[Item]:
    """Get items interactively from user."""
    print("\n📦 ITEM INPUT")
    print("=" * 50)
    
    items = []
    
    # Offer quick start options
    print("Quick start options:")
    print("1. Use default furniture example (Top, Legs, Back, Support)")
    print("2. Enter custom items")
    print("3. Load from file")
    
    try:
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            # Default furniture items
            items = [
                Item(180, 80, 1, "Top"),
                Item(75, 80, 2, "Leg"),
                Item(75, 180, 1, "Back"),
                Item(75, 10, 1, "Front_Support")
            ]
            print("✅ Using default furniture items")
        
        elif choice == "3":
            filename = input("Enter JSON file path: ").strip()
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    items = [Item.from_dict(item_data) for item_data in data["items"]]
                print(f"✅ Loaded {len(items)} item types from {filename}")
            except Exception as e:
                print(f"❌ Error loading file: {e}")
                print("Using default items instead")
                return get_interactive_items()
        
        else:
            # Custom input
            try:
                num_types = int(input("Number of item types: "))
                
                for i in range(num_types):
                    print(f"\n📦 Item type {i+1}:")
                    name = input("  Name: ").strip() or f"Item_{i+1}"
                    length = int(input("  Length (cm): "))
                    width = int(input("  Width (cm): "))
                    quantity = int(input("  Quantity: "))
                    
                    items.append(Item(length, width, quantity, name))
                    print(f"  ✅ Added: {items[-1]}")
            
            except (ValueError, KeyboardInterrupt):
                print("\n⚠️  Invalid input, using default items")
                return get_interactive_items()
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    
    print(f"\n📋 Final item list:")
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item}")
    
    return items

def main():
    """Main function with enhanced CLI."""
    parser = argparse.ArgumentParser(
        description="Enhanced 2D Bin Packing Optimizer with C++ Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_pin_optimizer.py                    # Interactive mode
  python enhanced_pin_optimizer.py --algorithm greedy # Run specific algorithm  
  python enhanced_pin_optimizer.py --cpp-only        # Force C++ implementation
  python enhanced_pin_optimizer.py --config config.json # Use custom config
        """)
    
    parser.add_argument("--algorithm", choices=["greedy", "genetic", "simulated_annealing", "all"],
                       default="all", help="Algorithm to run")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--items", help="Items JSON file path")
    parser.add_argument("--cpp-only", action="store_true", help="Force C++ implementation only")
    parser.add_argument("--python-only", action="store_true", help="Force Python implementation only")
    parser.add_argument("--no-visualization", action="store_true", help="Disable visualization")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🚀" * 25)
        print("🔥 ENHANCED 2D BIN PACKING OPTIMIZER 🔥")
        print("🚀" * 25)
        print(f"📅 Date: 2025-09-02 06:26:39 UTC")
        print(f"👤 Author: abdelbar472") 
        print(f"💻 System: Intel i5-6440HQ @ 2.60GHz")
        print(f"📂 Location: D:\\codes\\2d_pin")
        print("🚀" * 25)
    
    # Load configuration
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
                if not args.quiet:
                    print(f"✅ Loaded configuration from {args.config}")
        except Exception as e:
            print(f"❌ Error loading config: {e}")
    
    # Initialize optimizer
    optimizer = HybridOptimizer(config)
    
    # Override implementation preference
    if args.cpp_only:
        optimizer.config["prefer_cpp"] = True
    elif args.python_only:
        optimizer.config["prefer_cpp"] = False
    
    if args.no_visualization:
        optimizer.config["visualization"] = {k: False for k in optimizer.config["visualization"]}
    
    # Get items
    if args.items:
        try:
            with open(args.items, 'r') as f:
                data = json.load(f)
                items = [Item.from_dict(item_data) for item_data in data["items"]]
                if not args.quiet:
                    print(f"✅ Loaded {len(items)} item types from {args.items}")
        except Exception as e:
            print(f"❌ Error loading items: {e}")
            return
    else:
        items = get_interactive_items()
    
    if not items:
        print("❌ No items to pack")
        return
    
    # Validate items
    plank_length = optimizer.config["plank"]["length"]
    plank_width = optimizer.config["plank"]["width"]
    
    for item in items:
        if ((item.length > plank_length and item.width > plank_width) and 
            (item.width > plank_length and item.length > plank_width)):
            print(f"❌ Item {item.name} ({item.length}x{item.width}) too large for plank ({plank_length}x{plank_width})")
            return
    
    # Run optimization
    try:
        if args.algorithm == "all":
            results = optimizer.compare_all_algorithms(items)
            
            if results:
                optimizer.print_detailed_comparison(results)
                
                if optimizer.config["visualization"]["save_plots"]:
                    optimizer.create_enhanced_visualization(results)
                
                if optimizer.config["output"]["save_layouts"]:
                    optimizer.save_comprehensive_results(results, args.output_dir)
                
                # Show individual visualizations
                for algorithm, (planks, metrics) in results.items():
                    if planks and optimizer.config["visualization"]["save_plots"]:
                        visualize_planks(planks, f"{algorithm.title()} Algorithm ({metrics['implementation']})")
        
        else:
            planks, metrics = optimizer.optimize_single_algorithm(items, args.algorithm)
            
            if planks:
                print(f"\n✅ {args.algorithm.upper()} Results:")
                print(f"   📦 Planks used: {len(planks)}")
                print(f"   ⏱️  Execution time: {metrics['execution_time']:.3f} seconds")
                print(f"   🔧 Implementation: {metrics['implementation']}")
                print(f"   🎯 Algorithm efficiency: {metrics['algorithm_efficiency']:.1f}%")
                print(f"   📊 Packing efficiency: {metrics['packing_efficiency']:.1f}%")
                
                if optimizer.config["visualization"]["save_plots"]:
                    visualize_planks(planks, f"{args.algorithm.title()} Algorithm ({metrics['implementation']})")
                
                if optimizer.config["output"]["save_layouts"]:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    save_layout(planks, f"{args.algorithm}_layout_{timestamp}.txt")
            else:
                print(f"❌ {args.algorithm} failed to find solution")
    
    except KeyboardInterrupt:
        print("\n👋 Optimization interrupted by user")
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()