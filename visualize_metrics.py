import matplotlib.pyplot as plt
import numpy as np
import os
import re
from collections import defaultdict
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_tensorboard_data(log_dir):
    """从tensorboard事件文件中加载训练指标"""
    metrics = defaultdict(list)
    
    # 查找最新的事件文件
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        print(f"No event files found in {log_dir}")
        return metrics
    
    # 使用最新的事件文件
    latest_event_file = max(event_files, key=os.path.getctime)
    print(f"Loading data from: {latest_event_file}")
    
    # 加载事件文件
    event_acc = EventAccumulator(latest_event_file)
    event_acc.Reload()
    
    # 获取所有可用的指标
    tags = event_acc.Tags()
    print("\nAvailable metrics:", tags['scalars'])
    
    # 读取训练损失
    if 'Loss' in event_acc.Tags()['scalars']:
        events = event_acc.Scalars('Loss')
        metrics['train_loss'] = [(e.step, e.value) for e in events]
    
    # 读取验证指标
    if 'Val/MR' in event_acc.Tags()['scalars']:
        events = event_acc.Scalars('Val/MR')
        metrics['val_mr'] = [(e.step, e.value) for e in events]
    
    if 'Val/minADE' in event_acc.Tags()['scalars']:
        events = event_acc.Scalars('Val/minADE')
        metrics['val_minade'] = [(e.step, e.value) for e in events]
    
    if 'Val/minFDE' in event_acc.Tags()['scalars']:
        events = event_acc.Scalars('Val/minFDE')
        metrics['val_minfde'] = [(e.step, e.value) for e in events]
    
    return metrics

def plot_metrics(metrics):
    """绘制训练指标变化图"""
    if not metrics:
        print("No metrics found!")
        return
    
    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Over Time', fontsize=16)
    
    # 绘制训练损失
    if metrics['train_loss']:
        ax = axes[0, 0]
        steps, values = zip(*metrics['train_loss'])
        ax.plot(steps, values, 'b-', label='Training Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True)
        ax.legend()
    
    # 绘制验证Miss Rate
    if metrics['val_mr']:
        ax = axes[0, 1]
        steps, values = zip(*metrics['val_mr'])
        ax.plot(steps, values, 'r-', label='Validation MR')
        ax.set_xlabel('Step')
        ax.set_ylabel('Miss Rate')
        ax.set_title('Validation Miss Rate')
        ax.grid(True)
        ax.legend()
    
    # 绘制验证minADE
    if metrics['val_minade']:
        ax = axes[1, 0]
        steps, values = zip(*metrics['val_minade'])
        ax.plot(steps, values, 'g-', label='Validation minADE')
        ax.set_xlabel('Step')
        ax.set_ylabel('minADE')
        ax.set_title('Validation minADE')
        ax.grid(True)
        ax.legend()
    
    # 绘制验证minFDE
    if metrics['val_minfde']:
        ax = axes[1, 1]
        steps, values = zip(*metrics['val_minfde'])
        ax.plot(steps, values, 'y-', label='Validation minFDE')
        ax.set_xlabel('Step')
        ax.set_ylabel('minFDE')
        ax.set_title('Validation minFDE')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图表
    save_dir = 'visualization_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印统计信息
    print("\nTraining Statistics:")
    
    if metrics['train_loss']:
        values = [v for _, v in metrics['train_loss']]
        print(f"\nTraining Loss:")
        print(f"Initial: {values[0]:.3f}")
        print(f"Final: {values[-1]:.3f}")
        print(f"Best: {min(values):.3f}")
    
    if metrics['val_mr']:
        values = [v for _, v in metrics['val_mr']]
        print(f"\nValidation Miss Rate:")
        print(f"Initial: {values[0]:.3f}")
        print(f"Final: {values[-1]:.3f}")
        print(f"Best: {min(values):.3f}")
    
    if metrics['val_minade']:
        values = [v for _, v in metrics['val_minade']]
        print(f"\nValidation minADE:")
        print(f"Initial: {values[0]:.3f}")
        print(f"Final: {values[-1]:.3f}")
        print(f"Best: {min(values):.3f}")
    
    if metrics['val_minfde']:
        values = [v for _, v in metrics['val_minfde']]
        print(f"\nValidation minFDE:")
        print(f"Initial: {values[0]:.3f}")
        print(f"Final: {values[-1]:.3f}")
        print(f"Best: {min(values):.3f}")

def main():
    # 查找最新的日志目录
    log_dirs = glob.glob('./log/*')
    if not log_dirs:
        print("Error: No log directories found in ./log/")
        return
    
    # 使用最新的日志目录
    latest_log_dir = max(log_dirs, key=os.path.getctime)
    print(f"Using log directory: {latest_log_dir}")
    
    # 加载并绘制指标
    metrics = load_tensorboard_data(latest_log_dir)
    plot_metrics(metrics)

if __name__ == '__main__':
    main()
