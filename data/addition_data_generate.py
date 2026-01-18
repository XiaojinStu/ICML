import random
import json
import itertools

def generate_addition_problems(digit_length, num_problems):
    problems = []

    if digit_length <= 4:
        all_combinations = list(itertools.product(range(10**(digit_length-1), 10**digit_length), repeat=2))
        sampled_combinations = random.sample(all_combinations, num_problems)
        for a, b in sampled_combinations:
            problems.append({
                'question': f'What is the sum of {a} and {b}?',
                'answer': a + b
            })
    else:
        for _ in range(num_problems):
            a = random.randint(10**(digit_length-1), 10**digit_length - 1)
            b = random.randint(10**(digit_length-1), 10**digit_length - 1)
            problems.append({
                'question': f'What is the sum of {a} and {b}?',
                'answer': a + b
            })

    return problems

def create_dataset(l, r, num_problems):
    dataset = []
    # num_problems = 20

    for digits in range(l, r+1):
        dataset.extend(generate_addition_problems(digits, num_problems))

    return dataset

def save_to_json(dataset, filename):
    with open(filename, 'w') as json_file:
        json.dump(dataset, json_file, indent=4)

if __name__ == '__main__':
    print (1)
    # Generate the dataset
    l = 1
    r = 50
    num_problems=1
    dataset = create_dataset(l, r, num_problems)

    # Save dataset to a JSON file
    save_to_json(dataset, f'addition_problems_dataset({l}-{r})({num_problems}).json')

    print(f"Dataset has been saved to 'addition_problems_dataset({l}-{r})({num_problems}).json'")
