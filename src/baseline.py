from training import *
from plots import *

def simulate_comparison_same_human(num_envs=5):
    human_seed = Human(settings.get_min_value(), settings.get_max_value())
    while human_seed.get_state()['happiness'] >= 3.0:
        human_seed = Human(settings.get_min_value(), settings.get_max_value())
    
    human_dynamic = human_seed.clone()
    human_baseline = human_seed.clone()

    widths = [random.randint(settings.get_min_window_width(), settings.get_max_window_width()) for _ in range(num_envs)]
    windows = [0]
    total = 0
    for w in widths:
        total += w
        windows.append(total)

    envs = [sample_environment(ENV_OBS) for _ in range(num_envs)]
    obs = [observations_to_facts(generate_observations(env, ENV_OBS)) for env in envs]

    print("Step intervals:", [(windows[i], windows[i+1]) for i in range(len(windows)-1)])

    def run_agent(human, agent_type):
        agent = Agent(settings.get_min_value(), settings.get_max_value(), settings.get_step_size())
        results = {'history': {'happiness': [], 'sadness': [], 'anger': [], 'anxiety': []}, 
                   'actions': [],
                   'explored_states': []}
        step = 0

        for i in range(len(windows)-1):
            start, end = windows[i], windows[i+1]
            env_name = envs[i]
            facts = obs[i]
            environment = SocialEnvironment(context_type=env_name)

            inferred_env = infer_environment("env_rules.lp", facts)[0]

            if agent_type == "dynamic":
                agent.load_q_table(inferred_env)
            elif agent_type == "baseline":
                agent.load_baseline_q_table()

            while step < end:
                state = human.get_state()
                action, explored_state = agent.choose_best_action(state)

                reaction = human.get_scaled_reaction(action)
                feedback = environment.get_feedback(action)
                
                human.step(action, feedback)

                for emotion in results['history']:
                    results['history'][emotion].append(human.emotions[emotion])

                results['actions'].append(action)
                results['explored_states'].append(explored_state)
                step += 1

        return results

    print("=== Simulating Dynamic Agent ===")
    dynamic_results = run_agent(human_dynamic, "dynamic")
    print("\n=== Simulating Baseline Agent ===")
    baseline_results = run_agent(human_baseline, "baseline")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for emotion, values in dynamic_results['history'].items():
        axes[0].plot(values, label=emotion)
    axes[0].set_title("Dynamic Agent Emotions Over Time (Same Human)")
    axes[0].legend()
    axes[0].grid(True)

    for emotion, values in baseline_results['history'].items():
        axes[1].plot(values, label=emotion)
    axes[1].set_title("Baseline Agent Emotions Over Time (Same Human)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    return dynamic_results, baseline_results, windows, envs

if __name__ == "__main__":
    ENV_OBS = load_env_obs()
    train(num_episodes=1000,
            max_steps_per_episode=800, 
            epsilon=0.3,
            decay_rate=0.995,
            learning_rate=0.1,
            discount_factor=0.3
            )
    
    train_baseline(num_episodes=1000,
            max_steps_per_episode=800, 
            epsilon=0.3,
            decay_rate=0.995,
            learning_rate=0.1,
            discount_factor=0.3,
            env_type='neutral')

    dynamic_results, baseline_results, windows, envs = simulate_comparison_same_human()
    plot_comparison_results(dynamic_results, baseline_results, windows, envs)
