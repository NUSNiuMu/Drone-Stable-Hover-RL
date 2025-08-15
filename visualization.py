import numpy as np
import matplotlib.pyplot as plt
import os
from drone_PPO import train_ppo, evaluate_model

def create_plots_directory():
    """创建保存图表的目录"""
    plots_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def plot_training_performance(training_data, plots_dir):
    """绘制训练性能图表"""
    # 训练奖励曲线
    plt.figure(figsize=(12, 6))
    rewards = training_data['rewards']
    plt.plot(rewards, label='每回合奖励')
    
    # 计算移动平均
    window_size = 10
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(window_size-1, len(rewards)), moving_avg, 'r-', label=f'{window_size}回合移动平均')
    
    plt.title('训练奖励曲线')
    plt.xlabel('Episode')
    plt.ylabel('奖励')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'training_reward.png'))
    plt.close()

    # 位置误差曲线
    plt.figure(figsize=(12, 6))
    position_errors = training_data['position_errors']
    plt.plot(position_errors, label='位置误差')
    
    # 计算移动平均
    moving_avg = np.convolve(position_errors, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(window_size-1, len(position_errors)), moving_avg, 'r-', label=f'{window_size}步移动平均')
    
    plt.title('训练过程中的位置误差')
    plt.xlabel('步数')
    plt.ylabel('位置误差 (m)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'position_error.png'))
    plt.close()

def plot_evaluation_results(eval_data, plots_dir):
    """绘制评估结果图表"""
    # 轨迹图
    plt.figure(figsize=(10, 6))
    trajectory = eval_data['trajectory']
    plt.plot(trajectory[:, 0], trajectory[:, 1], label='XY平面轨迹')
    plt.scatter([0], [0], color='red', label='目标位置')
    plt.title('无人机轨迹')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'trajectory.png'))
    plt.close()

    # 高度变化
    plt.figure(figsize=(10, 6))
    plt.plot(trajectory[:, 2], label='高度')
    plt.axhline(y=1, color='r', linestyle='--', label='目标高度')
    plt.title('高度变化')
    plt.xlabel('步数')
    plt.ylabel('高度 (m)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'height.png'))
    plt.close()

    # 控制输入
    plt.figure(figsize=(10, 6))
    control_inputs = eval_data['control_inputs']
    for i in range(4):
        plt.plot(control_inputs[:, i], label=f'旋翼 {i+1}')
    plt.title('控制输入 (RPM)')
    plt.xlabel('步数')
    plt.ylabel('转速 (RPM)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'control_inputs.png'))
    plt.close()

def main():
    # 创建图表目录
    plots_dir = create_plots_directory()
    
    # 训练模型并获取数据
    print("开始训练...")
    training_data = train_ppo(max_episodes=800, enable_gui=False)
    
    # 打印训练数据统计信息
    print("\n训练数据统计:")
    print(f"总回合数: {len(training_data['rewards'])}")
    print(f"平均奖励: {np.mean(training_data['rewards']):.2f}")
    print(f"最大奖励: {np.max(training_data['rewards']):.2f}")
    print(f"最小奖励: {np.min(training_data['rewards']):.2f}")
    print(f"平均位置误差: {np.mean(training_data['position_errors']):.2f} m")
    
    # 绘制训练性能图表
    print("\n生成训练性能图表...")
    plot_training_performance(training_data, plots_dir)
    
    # 评估模型
    print("\n开始评估...")
    eval_data = evaluate_model(num_episodes=5, enable_gui=False, wind_speed=0.3)
    
    # 绘制评估结果图表
    print("生成评估结果图表...")
    plot_evaluation_results(eval_data, plots_dir)
    
    print(f"\n所有图表已保存到: {plots_dir}")

if __name__ == "__main__":
    main() 