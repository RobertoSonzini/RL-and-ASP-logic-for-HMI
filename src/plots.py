import matplotlib.pyplot as plt
import numpy as np
import math

def plot_reward_history(reward_history):
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Reward history")
    plt.show()

def plot_q_table_analysis(agent):
    analysis_q_table = agent.analyze_actual_q_table()

    total_states = analysis_q_table.get("total_states", 0)
    visited_states = analysis_q_table.get("visited_states", 0)
    coverage_ratio = analysis_q_table.get("state_coverage_ratio", 0.0)

    plt.figure(figsize=(8, 5))
    plt.bar(["Visited", "Unvisited"], [visited_states, total_states - visited_states],
            color=["skyblue", "lightcoral"])
    
    plt.title(f"Q-table State Coverage: {coverage_ratio*100:.2f}%")
    plt.ylabel("Number of states")
    plt.grid(axis="y", alpha=0.3)
    plt.show()

def plot_epsilon_histories(epsilon_histories):
    plt.figure(figsize=(10, 5))
    
    all_eps = np.array(epsilon_histories)
    
    for episode_eps in all_eps:
        plt.plot(episode_eps, color='skyblue', alpha=0.3)
    
    mean_eps = np.mean(all_eps, axis=0)
    
    plt.plot(mean_eps, color='blue', linewidth=2.5, label='Mean Epsilon')
    
    plt.title("Epsilon per step in tutti gli episodi")
    plt.xlabel("Step")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_convergences(convergences):
    plt.figure(figsize=(10, 5))

    all_conv = np.array(convergences)

    for conv_curve in all_conv:
        plt.plot(conv_curve, alpha=0.25)

    mean_conv = np.mean(all_conv, axis=0)
    plt.plot(mean_conv, linewidth=3, label="Mean ΔQ")

    plt.title("Q-table Convergence (ΔQ tra episodi)")
    plt.xlabel("Episode")
    plt.ylabel("Mean |ΔQ|")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()

def plot_entropy_distribution(entropies):
    plt.figure(figsize=(8,5))
    plt.hist(entropies, bins=30)
    plt.xlabel("Normalized Entropy")
    plt.ylabel("Number of States")
    plt.title("Distribution of Policy Entropy Across States")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_total_reactions(human, feedbacks, resolution=100, ncols = 2):
    actions = list(human.reactions.keys())
    emotions = list(human.emotions.keys())
    emotion_range = np.linspace(human.min_value, human.max_value, resolution)
    colors = ["#2b7dbf", "#E77F35", "#D6B12B", "#7836B6"]

    n_actions = len(actions)
    nrows = math.ceil(n_actions / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3*nrows), sharex=True)
    axes = axes.flatten()

    for i, action in enumerate(actions):
        ax = axes[i]

        happiness_base = human.reactions[action]["happiness"]
        happiness_base_curve = happiness_base * np.exp(-emotion_range / 2)
        happiness_total_curve = happiness_base_curve + feedbacks[action]["happiness"]

        negative_indices = np.where(happiness_total_curve < 0)[0]
        critical_x = emotion_range[negative_indices[0]] if len(negative_indices) > 0 else None

        for j, emotion in enumerate(emotions):
            base_delta = human.reactions[action][emotion]

            base_curve = base_delta * np.exp(-emotion_range / 2)
            total_curve = base_curve + feedbacks[action][emotion]

            ax.plot(emotion_range, base_curve,
                    linestyle='--',
                    color=colors[j],
                    alpha=0.6)

            ax.plot(emotion_range, total_curve,
                    label=emotion,
                    color=colors[j],
                    linewidth=2)

            ax.fill_between(
                emotion_range,
                base_curve,
                total_curve,
                color='red',
                alpha=0.15
            )

        if critical_x is not None:
            ax.axvline(critical_x, color='red', linestyle='-', alpha=0.7)

        ax.set_title(f"{action}")
        ax.grid(True, linestyle='--', alpha=0.6)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.supxlabel("Emotion value")
    fig.supylabel("Scaled delta")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_results(results, plot_history=True, plot_actions=True, happiness_window=3, intervals=None, envs=None):
    history = results['history']
    actions = results['actions']
    explored_states = results['explored_states']
    
    n_plots = int(plot_history) + int(plot_actions)
    if n_plots == 0:
        print("Nothing to plot.")
        return
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    ax_index = 0
    
    if plot_history:
        ax = axes[ax_index]
        for emotion, values in history.items():
            ax.plot(values, label=emotion)
        
        happiness_values = np.array(history.get('happiness', []))
        if len(happiness_values) > happiness_window:
            other_emotions = [e for e in history if e != 'happiness']
            for i in range(len(happiness_values) - happiness_window):
                window_delta_h = happiness_values[i + happiness_window] - happiness_values[i]
                
                others_not_increasing = True
                for e in other_emotions:
                    e_values = np.array(history[e])
                    if i + happiness_window < len(e_values):
                        delta = e_values[i + happiness_window] - e_values[i]
                        if delta > 0:
                            others_not_increasing = False
                            break
                
                if window_delta_h > 0 and others_not_increasing:
                    ax.axvspan(i, i + happiness_window, color=(1.0, 0.6, 0.2), alpha=0.3)

        ax.set_ylabel("Emotion Level")
        ax.set_title("Emotional State Over Time (highlight happiness rising and decrease of other emotions)")
        ax.legend()
        ax.grid(True)
        
        ax_index += 1

    if plot_actions:
        ax = axes[ax_index]
        steps = list(range(len(actions)))
        unique_actions = sorted(set(actions))
        action_to_index = {action: i for i, action in enumerate(unique_actions)}
        indexed_actions = [action_to_index[a] for a in actions]
        
        explored_states = results['explored_states']
        
        ax.plot(steps, indexed_actions, linestyle='-', alpha=0.4)
        
        explored_x = [i for i in steps if explored_states[i]]
        explored_y = [indexed_actions[i] for i in steps if explored_states[i]]
        not_explored_x = [i for i in steps if not explored_states[i]]
        not_explored_y = [indexed_actions[i] for i in steps if not explored_states[i]]
        
        ax.scatter(explored_x, explored_y, marker='o', label="Explored", color='blue')
        ax.scatter(not_explored_x, not_explored_y, marker='x', label="Not explored", color='red')
        
        ax.set_xlabel("Step")
        ax.set_ylabel("Action")
        ax.set_title("Selected Actions Over Time")
        
        ax.set_yticks(range(len(unique_actions)))
        ax.set_yticklabels(unique_actions)
        ax.legend()
        ax.grid(True)

    if intervals is not None:
        prev = 0
        xticks = []
        xtick_labels = []

        for i, x in enumerate(intervals):
            for ax in axes:
                ax.axvline(x=x, linestyle="--", alpha=0.6)

            mid = (prev + x) / 2
            xticks.append(mid)

            if envs is not None and i < len(envs):
                xtick_labels.append(str(envs[i]))
            else:
                xtick_labels.append(str(i))

            prev = x

        for ax in axes:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels, rotation = 90)
    
    plt.tight_layout()
    plt.show()

def plot_comparison_results(dynamic_results, baseline_results, intervals=None, envs=None):
    emotions = list(dynamic_results['history'].keys())
    colors = ['blue', 'green', 'red', 'purple']
    
    fig, ax = plt.subplots(figsize=(14, 6))

    # Dynamic
    for i, emotion in enumerate(emotions):
        ax.plot(dynamic_results['history'][emotion],
                label=f"Dynamic - {emotion}",
                color=colors[i % len(colors)],
                linewidth=2)

    # Baseline
    for i, emotion in enumerate(emotions):
        ax.plot(baseline_results['history'][emotion],
                label=f"Baseline - {emotion}",
                color=colors[i % len(colors)],
                linestyle='--',
                linewidth=2)

    # Intervalli + env labels
    if intervals is not None and envs is not None:
        # linee verticali per tutti i boundary
        for x in intervals:
            ax.axvline(x=x, color='black', linestyle=':', alpha=0.5)

        # label all'inizio di ogni intervallo
        y_top = ax.get_ylim()[1]
        for i, env in enumerate(envs):
            x = intervals[i]  # inizio intervallo

            # 👉 per centrare nel blocco usa:
            x = (intervals[i] + intervals[i+1]) / 2

            ax.text(x, y_top * 0.95, str(env),
                    rotation=90,
                    verticalalignment='top',
                    horizontalalignment='right',
                    fontsize=9,
                    bbox=dict(facecolor='wheat', alpha=0.5, boxstyle='round'))

    ax.set_xlabel("Step")
    ax.set_ylabel("Emotion Value")
    ax.set_title("Dynamic vs Baseline - Emotions Over Time")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()