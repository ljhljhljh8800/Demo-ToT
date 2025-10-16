# 5-shot
standard_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24
Input: {input}
'''

# 5-shot
cot_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24

Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24

Input: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24

Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24

Input: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24

Instruction: Mimick the format of the demonstration, and generate Steps and/or Answer for the following input. Please do not generate any other text contents. 

Input: {input}
'''

# 1-shot
propose_prompt = '''<demonstrations>

<placeholder>

Instruction: Mimick the format of the above <demonstrations>, and generate possible next steps for the following input. Please do not generate any other text contents. 

Input: {input}
Possible next steps:
'''


demo_format = """Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 10 8 14)
2 * 8 = 16 (left: 16 8 14)
8 - 2 = 6 (left: 6 8 14)
8 / 2 = 4 (left: 4 8 14)
2 + 14 = 16 (left: 8 8 16)
2 * 14 = 12 (left: 8 8 28)
14 / 2 = 7 (left: 8 8 7)
14 - 2 = 12 (left: 8 8 12)
8 + 8 = 16 (left: 2 16 14)
8 - 8 = 16 (left: 2 0 14)
8 * 8 = 16 (left: 2 64 14)
8 / 8 = 1 (left: 2 1 14)
"""



value_prompt = '''<demonstrations>

<placeholder>

Task: Evaluate if given numbers can reach 24 (sure/likely/impossible)

Instruction: Mimick the format and reasoning steps of the <demonstrations>, and generate possible future steps and the final evaluation for the following input. Please do not generate any other text contents. 

input: {input}
'''

demo_format = """input: 11 12
possible future steps: 
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
final evaluation: impossible

input: 4 4 10
possible future steps: 
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
final evaluation: sure

input: 4 9 11
possible future steps: 
9 + 11 + 4 = 20 + 4 = 24
final evaluation: sure

input: 5 7 8
possible future steps: 
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
final evaluation: likely

input: 5 6 6
possible future steps: 
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
final evaluation: likely

input: 10 10 11
possible future steps: 
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
final evaluation: impossible

input: 1 3 3
possible future steps: 
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
final evaluation: impossible"""


value_last_step_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24. 
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judge: 
sure

Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Judge: 
sure

Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Judge: 
sure

Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge: 
impossible

Input: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge: 
impossible

Input: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge: 
impossible

Instruction: Mimick the format of the demonstrations, and generate a judgement (sure/impossible) for the following input. Please do not generate any other text contents. 

Input: {input}
Answer: {answer}
Judge:'''