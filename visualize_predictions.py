import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def visualize_test_trajectory(file_path, save_path=None):
    """
    可视化测试轨迹数据
    Args:
        file_path: CSV文件路径
        save_path: 保存图像的路径
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 提取AGENT的轨迹
    agent_df = df[df['OBJECT_TYPE'] == 'AGENT']
    
    if len(agent_df) > 0:
        # 按时间戳排序
        agent_df = agent_df.sort_values('TIMESTAMP')
        
        # 提取轨迹点
        trajectory = agent_df[['X', 'Y']].values
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 绘制轨迹
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Trajectory')
        
        # 标记起点和终点
        plt.scatter(trajectory[0, 0], trajectory[0, 1], c='g', s=100, label='Start')
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='r', s=100, label='End')
        
        plt.grid(True)
        plt.legend()
        plt.title('Agent Trajectory')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def get_actual_trajectories(predictions, last_obs_point, origin_point=None, origin_angle=None):
    """
    Convert relative displacement predictions to actual trajectories.
    
    Args:
        predictions: Array of shape (k, future_len, 2) containing relative displacements
        last_obs_point: Last observed point (x, y)
        origin_point: Origin point for coordinate transformation
        origin_angle: Angle for coordinate transformation
    
    Returns:
        actual_trajectories: List of k trajectories in global coordinates
    """
    k, future_len, _ = predictions.shape
    actual_trajectories = []
    
    for k_idx in range(k):
        # Start from the last observed point
        current_traj = np.zeros((future_len + 1, 2))
        current_traj[0] = last_obs_point
        
        # Accumulate relative displacements
        for t in range(future_len):
            current_traj[t + 1] = current_traj[t] + predictions[k_idx, t]
        
        # Transform to global coordinates if needed
        if origin_point is not None and origin_angle is not None:
            for t in range(len(current_traj)):
                x, y = current_traj[t]
                current_traj[t] = rotate(x - origin_point[0], 
                                       y - origin_point[1], 
                                       -origin_angle)
        
        actual_trajectories.append(current_traj)
    
    return actual_trajectories

def visualize_predictions(predictions, last_obs_point, origin_point=None, origin_angle=None, 
                         save_path=None):
    """
    Visualize predicted trajectories.
    
    Args:
        predictions: Array of shape (k, future_len, 2) containing relative displacements
        last_obs_point: Last observed point (x, y)
        origin_point: Origin point for coordinate transformation
        origin_angle: Angle for coordinate transformation
        save_path: Path to save the visualization
    """
    actual_trajectories = get_actual_trajectories(predictions, last_obs_point, 
                                                origin_point, origin_angle)
    
    plt.figure(figsize=(10, 8))
    
    # Plot each predicted trajectory
    for i, traj in enumerate(actual_trajectories):
        alpha = 0.7 if i == 0 else 0.3  # Make the first prediction more prominent
        plt.plot(traj[:, 0], traj[:, 1], '-', alpha=alpha, label=f'Prediction {i+1}')
        
        # Mark start and end points
        plt.scatter(traj[0, 0], traj[0, 1], c='g', marker='o', label='Start' if i == 0 else '')
        plt.scatter(traj[-1, 0], traj[-1, 1], c='r', marker='x', label='End' if i == 0 else '')
    
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.title('Predicted Trajectories')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def rotate(x, y, angle):
    new_x = x * np.cos(angle) - y * np.sin(angle)
    new_y = x * np.sin(angle) + y * np.cos(angle)
    return np.array([new_x, new_y])

def analyze_trajectory_errors(pred_trajectories, gt_trajectory, time_steps=30):
    """
    分析每个时间步的预测轨迹和真实轨迹的坐标点及其误差
    
    Args:
        pred_trajectories: 预测轨迹列表，每个元素shape为(time_steps, 2)
        gt_trajectory: 真实轨迹，shape为(time_steps, 2)
        time_steps: 时间步数，默认30
    
    Returns:
        errors_by_step: 每个时间步的误差统计
    """
    errors_by_step = []
    
    # 对每个时间步进行分析
    for t in range(time_steps):
        step_errors = {
            'time_step': t,
            'gt_coord': gt_trajectory[t],
            'predictions': [],
            'errors': []
        }
        
        # 计算每个预测轨迹在当前时间步的误差
        for k, pred_traj in enumerate(pred_trajectories):
            pred_coord = pred_traj[t]
            error = np.sqrt((pred_coord[0] - gt_trajectory[t][0])**2 + 
                          (pred_coord[1] - gt_trajectory[t][1])**2)
            
            step_errors['predictions'].append(pred_coord)
            step_errors['errors'].append(error)
        
        errors_by_step.append(step_errors)
    
    return errors_by_step

def visualize_trajectory_analysis(errors_by_step, save_path=None):
    """
    可视化轨迹分析结果
    
    Args:
        errors_by_step: analyze_trajectory_errors的输出
        save_path: 保存图像的路径
    """
    # 创建两个子图：左边显示轨迹，右边显示误差
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 提取数据
    time_steps = len(errors_by_step)
    gt_coords = np.array([step['gt_coord'] for step in errors_by_step])
    pred_coords_by_k = []
    errors_by_k = []
    
    # 重组预测数据
    for k in range(len(errors_by_step[0]['predictions'])):
        pred_coords_k = np.array([step['predictions'][k] for step in errors_by_step])
        errors_k = np.array([step['errors'][k] for step in errors_by_step])
        pred_coords_by_k.append(pred_coords_k)
        errors_by_k.append(errors_k)
    
    # 绘制轨迹图
    ax1.plot(gt_coords[:, 0], gt_coords[:, 1], 'g-', label='Ground Truth', linewidth=2)
    for k, pred_coords in enumerate(pred_coords_by_k):
        alpha = 0.7 if k == 0 else 0.3
        ax1.plot(pred_coords[:, 0], pred_coords[:, 1], '--', 
                alpha=alpha, label=f'Prediction {k+1}')
    
    ax1.grid(True)
    ax1.legend()
    ax1.set_title('Trajectories Comparison')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.axis('equal')
    
    # 绘制误差图
    time_steps = range(len(errors_by_step))
    for k, errors in enumerate(errors_by_k):
        alpha = 0.7 if k == 0 else 0.3
        ax2.plot(time_steps, errors, '--', alpha=alpha, label=f'Prediction {k+1}')
    
    ax2.grid(True)
    ax2.legend()
    ax2.set_title('Error by Time Step')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Error (meters)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    # 打印统计信息
    print("\nTrajectory Analysis Summary:")
    for k in range(len(errors_by_k)):
        errors = errors_by_k[k]
        print(f"\nPrediction {k+1}:")
        print(f"  Average Error: {np.mean(errors):.2f} meters")
        print(f"  Max Error: {np.max(errors):.2f} meters at step {np.argmax(errors)}")
        print(f"  Min Error: {np.min(errors):.2f} meters at step {np.argmin(errors)}")
        print(f"  Final Displacement Error: {errors[-1]:.2f} meters")

def main():
    # 测试数据目录
    test_dir = '/Users/chuanhanyuan/Desktop/code/testdata/test_obs/'
    
    # 检查测试数据目录是否存在
    if not os.path.exists(test_dir):
        print(f"错误：测试数据目录不存在: {test_dir}")
        return
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"错误：在目录 {test_dir} 中没有找到CSV文件")
        return
        
    print(f"找到 {len(csv_files)} 个CSV文件")

    # 先检查所有CSV文件的轨迹点数
    print("\n=== CSV文件轨迹点统计 ===")
    for file_name in csv_files:
        file_path = os.path.join(test_dir, file_name)
        df = pd.read_csv(file_path)
        agent_df = df[df['OBJECT_TYPE'] == 'AGENT']
        agent_df = agent_df.sort_values('TIMESTAMP')
        
        print(f"\n文件 {file_name}:")
        print(f"  总行数: {len(df)}")
        print(f"  AGENT轨迹点数: {len(agent_df)}")
        print(f"  时间戳范围: {agent_df['TIMESTAMP'].min()} 到 {agent_df['TIMESTAMP'].max()}")
        
        # 显示前几个轨迹点的时间戳和坐标
        print("\n  前5个轨迹点:")
        for idx, row in agent_df.head().iterrows():
            print(f"    t={row['TIMESTAMP']}: ({row['X']:.2f}, {row['Y']:.2f})")
            
        # 显示后几个轨迹点的时间戳和坐标
        print("\n  后5个轨迹点:")
        for idx, row in agent_df.tail().iterrows():
            print(f"    t={row['TIMESTAMP']}: ({row['X']:.2f}, {row['Y']:.2f})")
            
    user_input = input("\n是否继续处理轨迹分析？(y/n): ")
    if user_input.lower() != 'y':
        return
        
    # 创建结果目录
    results_dir = 'visualization_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # 创建轨迹分析结果目录
    analysis_dir = os.path.join(results_dir, 'trajectory_analysis')
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    # 检查测试数据目录中的所有CSV文件
    csv_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"错误：在目录 {test_dir} 中没有找到CSV文件")
        return
        
    print(f"找到 {len(csv_files)} 个CSV文件")

    # 遍历测试数据目录中的所有CSV文件
    for file_name in csv_files:
        try:
            file_path = os.path.join(test_dir, file_name)
            print(f"\n处理文件: {file_name}")
            
            # 读取并处理轨迹数据
            df = pd.read_csv(file_path)
            agent_df = df[df['OBJECT_TYPE'] == 'AGENT']
            
            if len(agent_df) >= 50:  # 确保有足够的轨迹点（20个历史点+30个未来点）
                # 提取轨迹坐标
                agent_df = agent_df.sort_values('TIMESTAMP')
                trajectory = agent_df[['X', 'Y']].values
                
                # 分割历史轨迹和未来轨迹
                hist_trajectory = trajectory[:20]
                future_trajectory = trajectory[20:50]  # 取30个未来点
                
                # 生成模拟的预测轨迹
                pred_trajectories = []
                for k in range(3):
                    noise = np.random.normal(0, 0.5, future_trajectory.shape)
                    pred_traj = future_trajectory + noise
                    pred_trajectories.append(pred_traj)
                
                # 分析轨迹误差
                errors_by_step = analyze_trajectory_errors(pred_trajectories, future_trajectory)
                
                # 保存分析结果
                base_name = os.path.splitext(file_name)[0]
                analysis_path = os.path.join(analysis_dir, f'{base_name}_analysis.png')
                visualize_trajectory_analysis(errors_by_step, save_path=analysis_path)
                
                # 保存基本轨迹图
                basic_vis_path = os.path.join(results_dir, f'{base_name}_basic.png')
                visualize_test_trajectory(file_path, save_path=basic_vis_path)
                
                print(f"✓ 分析结果已保存到: {analysis_path}")
                print(f"✓ 基本可视化已保存到: {basic_vis_path}")
            else:
                print(f"跳过文件 {file_name}: 轨迹点数量不足 ({len(agent_df)} < 50)")
                
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")
            continue
            
    print("\n处理完成！")

if __name__ == "__main__":
    main()
