import clingo
import json
import random

def load_env_obs(path="env_obs.json"):
    with open(path) as f:
        return json.load(f)
    
def sample_environment(env_rules):
    return random.choice(list(env_rules.keys()))

def generate_observations(env, env_obs):
    return random.choice(env_obs[env])

def get_all_observation(env, env_obs):
    return env_obs[env]

def observations_to_facts(observations):
    return "\n".join(f"observed({obs})." for obs in observations)

def infer_environment(asp_file, facts):
    ctl = clingo.Control()
    ctl.add("base", [], open(asp_file).read() + "\n" + facts)
    ctl.ground([("base", [])])

    scores = {}
    possible_envs = set()

    def on_model(model):
        atoms = model.symbols(shown=True)

        for atom in atoms:
            if atom.name == "possible":
                env = atom.arguments[0].name
                possible_envs.add(env)

            elif atom.name == "total_score":
                env = atom.arguments[0].name
                score = atom.arguments[1].number
                scores[env] = score

    ctl.solve(on_model=on_model)

    ordered = sorted(
        possible_envs,
        key=lambda e: scores.get(e, 0),
        reverse=True
    )

    return ordered

def test_on_all_obs():
    ENV_OBS = load_env_obs()

    envs = ['neutral', 'hostile', 'friendly', 'tense', 'formal']

    accuracy = 0
    count = 0

    for env in envs:
        observations = get_all_observation(env, ENV_OBS)

        for obs in observations:
            facts = observations_to_facts(obs)
            inferred = infer_environment("env_rules.lp", facts)

            inferred_env = inferred[0]

            count += 1

            if inferred_env == env:
                accuracy += 1
            else:
                print(f"---> ERRORE: \n\t real:{env}\n\tinferred:{inferred}")

    print(f"TEST: {accuracy} / {count}")

if __name__ == '__main__':
    ENV_OBS = load_env_obs()

    real_env = sample_environment(ENV_OBS)

    observations = generate_observations(real_env, ENV_OBS)
    
    print("REAL ENV:", real_env)

    facts = observations_to_facts(observations)
    print("\nASP FACTS:\n" + facts)

    inferred = infer_environment("env_rules.lp", facts)
    print("\nINFERRED ENV:", inferred) 

    test_on_all_obs()
