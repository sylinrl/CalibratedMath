def generate_math_prompt(operation, x1, x2, separator, x3=None):

    if len(separator):
        x1_string = '{:,}'.format(x1)
        x2_string = '{:,}'.format(x2)
    else:
        x1_string = str(x1)
        x2_string = str(x2)
    x3_string = str(x3)

    prompt = ''

    if operation == '%':
        prompt += 'Q: What is {0}% of {1}?\nA:'.format(x2_string, x1_string)
    elif operation == '<':
        prompt += 'Q: Name any number smaller than {0}?\nA:'.format(x1_string)
    elif operation == '>':
        prompt += 'Q: Name any number larger than {0}?\nA:'.format(x1_string)
    elif operation == 'prime':
        prompt += 'Q: Name any prime number smaller than {0}?\nA:'.format(x1_string)
    elif operation == 'square':
        prompt += 'Q: Name any perfect square smaller than {0}?\nA:'.format(x1_string)
    elif operation == '2sum':
        prompt += 'Q: Name two numbers that sum to {0}?\nA:'.format(x1_string)
    elif operation == 'multiple':
        prompt += 'Q: Name a single multiple of {0} between {1} and {2}?\nA:'.format(x3_string, x1_string, x2_string)
    elif operation == 'round':
        prompt += 'Q: What is {0} rounded to the nearest {1}?\nA:'.format(x1_string, x2_string)
    elif operation == 'remain':
        prompt += 'Q: What is the remainder when {0} is divided by {1}?\nA:'.format(x1_string, x2_string)
    elif operation == 'seq':
        seq_vals = [x1, x1 + x2, x1 + 2*x2, x1 + 3*x2]
        if len(separator):
            seq_strings = ['{:,}'.format(s) for s in seq_vals]
        else:
            seq_strings = [str(s) for s in seq_vals]
        prompt += 'Q: What comes next: {0}, {1}, {2}, {3}...?\nA:'.format(*seq_strings)
    elif operation in ['v+']:
        prompt += 'Q: What is {1} more than {0}?\nA:'.format(x1_string, x2_string)
    elif operation in ['v-']:
        prompt += 'Q: What is {1} less than {0}?\nA:'.format(x1_string, x2_string)
    elif operation in ['3*', '3+', '3-']:
        prompt += 'Q: What is {0} {3} {1} {3} {2}?\nA:'.format(x1_string, x2_string, x3_string, operation[-1])
    elif operation in ['frac']:
        prompt += 'Q: What is {0}/{1} in reduced form?\nA:'.format(x1_string, x2_string)
    else:
        prompt += 'Q: What is {0} {1} {2}?\nA:'.format(x1_string, operation, x2_string)

    return prompt

if __name__ == '__main__':
    print(generate_math_prompt('%', 1024, 25, ','))