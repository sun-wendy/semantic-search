import argparse
import sglang as sgl
from sglang import OpenAI, Anthropic, Runtime, assistant, gen, set_default_backend, system, user
import random

from sys_prompt import SYS_PROMPT


# Function adapted from: https://github.com/ChuyueSun/Clover/blob/main/clover/clover.py
@sgl.function
def generate_corpus_from_llm(s, model, num_pairs):
    s += system(SYS_PROMPT)
    model_output = ""

    for _ in range(num_pairs):
        cur_animal = get_animal_name()
        s += user(f"Generate one utterance-scene pair similar to the given examples. Use the animal {cur_animal}.")
        with s.copy() as tmp:
            tmp += assistant(gen("output", max_tokens=128, temperature=1))
            model_output = tmp["output"] + "\n"
        save_to_corpus(model_output)


def get_animal_name():
    with open("data/animals.txt", 'r') as f:
        animals = f.readlines()
    return random.choice(animals).strip().lower()


def save_to_corpus(utterance_scene_pair):
    corpus_file = "results/training_corpus/llm_corpus.txt"
    try:
        with open(corpus_file, 'r') as f:
            lines = f.readlines()
            last_num = 0
            for line in reversed(lines):
                if '-----' in line:  # Look specifically for lines with dashes
                    last_num = int(line.split('-----')[0])  # Split by ----- instead of just -
                    break
    except FileNotFoundError:
        last_num = 0
        lines = []

    # Add number to the output while keeping SENTENCE/SEM_REP format
    new_entry = f"\n{last_num + 1}-----\n{utterance_scene_pair}\n"
    with open(corpus_file, 'a') as f:
        if last_num == 0:
            f.write(new_entry.rstrip('\n'))
        else:
            f.write(new_entry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic corpus data from LLM")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_pairs", type=int, default=5000)
    args = parser.parse_args()
    
    # Model name examples: gpt-4o, claude-3-5-sonnet-20241022
    if args.model.startswith("gpt"):
        set_default_backend(OpenAI(args.model))
    elif args.model.startswith("claude"):
        set_default_backend(Anthropic(args.model))
    else:
        raise ValueError("Invalid model name")

    generate_corpus_from_llm(args.model, args.num_pairs)
