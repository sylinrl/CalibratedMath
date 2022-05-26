from configs import *
from prompts import *
import numpy as np
from fractions import Fraction
import math
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.calibration import calibration_curve


def is_prime(n):
    return (n > 1) and all(n % i for i in range(2, int(n ** 0.5) + 1))


def is_square(n):
    return n == (math.isqrt(n) ** 2)


def generate_samples(operation, n1, n2, separator, n_samples):

    """
    returns a size-n list of dictionaries representing individual arithmetic questions for the given operation.
    where relevant, n1 and n2 are the number of digits in each operand.
    """

    results = []
    for sample_idx in range(n_samples):

        x3 = None

        if operation == '%':  # fixed 'easy' percentages with integer answers
            pcts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 25, 75]
            pct = np.random.choice(pcts)

            lower = 10 ** (n1 - 1)
            upper = 10 ** n1

            if (pct % 10) == 0:
                x1 = np.random.randint(lower / 10, upper / 10)
                x1 = x1 * 10
            else:
                x1 = np.random.randint(lower / 4, upper / 4)
                x1 = x1 * 4

            x2 = pct

        elif operation == 'round':  # easy rounding, 5 / 10 / 100
            rnds = [5, 10, 100]
            x2 = np.random.choice(rnds)

            lower = 10 ** (n1 - 1)
            upper = 10 ** n1

            x1 = np.random.randint(lower, upper)

        elif operation == 'frac':  # express fraction in reduced form
            x1 = np.random.randint(10 ** (n1 - 1), 10 ** n1)  # digit count refers to reduced form
            x2 = np.random.randint(10 ** (n2 - 1), 10 ** n2)

            if x1 > x2:
                (x1, x2) = (x2, x1)  # keep the numerator smaller than the denominator

            x3 = np.random.randint(1, 11)

            x1 = x1 * x3
            x2 = x2 * x3

        elif operation in ['3*', '3+']:
            x1 = np.random.randint(1, 10)
            x2 = np.random.randint(1, 10)
            x3 = np.random.randint(1, 10)

        elif operation in ['seq']:
            x1 = np.random.randint(10 ** (n1 - 1), 10 ** n1)
            x2 = np.random.randint(10 ** (n2 - 1), 10 ** n2)

        elif operation in ['multiple']:
            x3 = np.random.randint(10 ** (n1 - 1), 10 ** n1)
            x1 = np.random.randint(10 ** (n2 - 1), 10 ** n2)
            x2 = np.random.randint(10 ** (n2 - 1), 10 ** n2)

            order = min(x1, x2), max(x1, x2)
            x1, x2 = order
            if (x2 - x1) < x3:
                x2 += x3

        else:
            # generate random ints -- for some questions, x2 isn't used
            x1 = np.random.randint(10 ** (n1 - 1), 10 ** n1)
            x2 = np.random.randint(10 ** (n2 - 1), 10 ** n2)

            # randomly shuffle order of the two integers for operations where order doesn't matter
            if np.random.randint(2) == 1:
                (x1, x2) = (x2, x1)

            if operation in ['/', '//', 'mod', 'remain']:
                x1 = x1 * x2  # these operations should use the reverse of the multiplication setup
                if operation in ['//', 'mod', 'remain']:  # doesn't need to be evenly divisible
                    x1 += np.random.randint(0, x2)

        # compute answer -- for multi-answer tasks, no answer is given
        answer = None
        if operation in ['+', 'v+']:
            answer = x1 + x2
        elif operation in ['-', 'v-']:
            answer = x1 - x2
        elif operation in ['*']:
            answer = x1 * x2
        elif operation == '%':
            answer = int(np.rint(x1 * (x2 / 100)))
        elif operation == '/':
            answer = int(np.rint(x1 / x2))
        elif operation == '//':
            answer = int(np.floor_divide(x1, x2))
        elif operation in ['mod', 'remain']:
            answer = x1 % x2
        elif operation == '3*':
            answer = x1 * x2 * x3
        elif operation == '3+':
            answer = x1 + x2 + x3
        elif operation == 'round':
            answer = x2 * round(x1 / x2)
        elif operation == 'seq':
            answer = x1 + 4*x2
        elif operation == 'frac':
            answer = Fraction(x1, x2).numerator, Fraction(x1, x2).denominator

        input_prompt = generate_math_prompt(operation, x1, x2, separator, x3=x3)
        results.append({
            'operation': operation,
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'n1': n1,
            'n2': n2,
            'separator': separator,
            'prompt': input_prompt,
            'answer': answer
        })

    return results


def check_correct(results):

    """
    'results' is the list of dictionaries from generate_samples()
    users should add an 'answer' field to each dictionary, containing their model's answer

    for 'frac' tasks, the model's answer should be a tuple of (numerator, denominator)
    for '2sum' tasks, the model's answer should be a tuple of two values that sum to the target
    all other tasks require a numeric (not string) value as the answer

    this function adds a new 'correct' field, indicating if the model answer is correct
    """

    for sample in results:
        operation = sample['operation']
        model = sample['model']
        answer = sample['answer']

        # separate evaluation for multi-answer
        if operation == '<':
            correct = model < sample['x1']
        elif operation == '>':
            correct = model > sample['x1']
        elif operation == 'prime':
            correct = is_prime(model) and (model < sample['x1'])
        elif operation == 'square':
            correct = is_square(model) and (model < sample['x1'])
        elif operation == 'multiple':
            correct = (model >= sample['x1']) and (model <= sample['x2']) and (model % sample['x3'] == 0)
        elif operation == '2sum':
            correct = sum(model) == sample['x1']
        else:  # single-answer
            correct = model == answer

        sample['correct'] = correct

    return results


def make_test_data(dataset):

    # creates test data: random model answers + confidence scores
    # used to show plotting + metrics

    group_p = {
        'Add-subtract': 0.5,
        'Multiply-divide': 0.2,
        'Multi-answer': 0
    }

    for group in dataset:
        p = group_p[group]
        for task in dataset[group]:
            for sample in dataset[group][task]:
                if np.random.uniform(0, 1) < (p / task[1]):
                    sample['model'] = sample['answer']
                else:
                    if task[0] in ['2sum', 'frac']:
                        sample['model'] = (np.random.randint(10), np.random.randint(100))
                    else:
                        sample['model'] = np.random.randint(1000)
                sample['confidence'] = np.clip((2 / (task[1] + task[2])) * 100 + np.random.randint(-5, 6), 0, 100)

    return dataset


def main():

    # generate dataset
    n_samples = 100
    dataset = {
        'Add-subtract': {},
        'Multiply-divide': {},
        'Multi-answer': {}
    }

    for task in TASKS:
        samples = generate_samples(*task, n_samples)
        if task[0] in ADDSUB_OPS:
            dataset['Add-subtract'][task] = samples
        elif task[0] in MULTDIV_OPS:
            dataset['Multiply-divide'][task] = samples
        else:
            dataset['Multi-answer'][task] = samples

    # add toy model answers + confidence scores -- replace with values from real model
    make_test_data(dataset)

    # check answers
    for group in dataset:
        for task in dataset[group]:
            check_correct(dataset[group][task])

    # set task-based confidence targets for fine-tuning
    for group in dataset:
        for task in dataset[group]:
            target = np.mean([example['correct'] for example in dataset[group][task]]) * 100
            for example in dataset[group][task]:
                example['target'] = int(target)

    # compute metrics and plot
    figure, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    axs = axs.ravel()
    ax_idx = 0
    axs[0].set_ylabel("Model accuracy")
    for group in dataset:
        ax = axs[ax_idx]
        corrects = np.array([example['correct'] for task in dataset[group] for example in dataset[group][task]]).astype(
            int)
        confidences = np.array(
            [example['confidence'] for task in dataset[group] for example in dataset[group][task]]) / 100.
        prob_true, prob_pred = calibration_curve(corrects, confidences, n_bins=20, strategy='quantile')

        # metrics
        MSE = np.power(corrects - confidences, 2).mean()
        MAD = abs(prob_true - prob_pred).mean() * 100.

        # calibration curve
        ax.scatter(prob_pred, prob_true)
        ax.plot([[0, 0], [1, 1]], linestyle='--', color='k', alpha=0.3)
        ax.set_xlabel("Model probability")
        ax.set_title(group)
        ax_idx += 1

        print("Dataset: {0}\nMSE: {1}\nMAD: {2}".format(group, MSE, MAD))


if __name__ == '__main__':

    main()
