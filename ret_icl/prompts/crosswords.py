# 5 shot


propose_prompt = f'''<demonstrations>

<placeholder>

Let's play a 5 x 5 mini crossword game, where each word should have exactly 5 letters. 

<current query>

{input}

Instruct: 
(a) Given the current status of the above board for the current query, list all possible answers for unfilled or changed words, and your confidence levels (certain/high/medium/low), using the format "h1. apple (medium)". Use "certain" cautiously and only when you are 100% sure this is the correct word. You can list more then one possible answer for each word. 
**Note that your response should only contain the contents described above, and nothing else (as in the <demonstrations>)**
(b) Please follow the format of the above <demonstrations> strictly.
(c) You will be engaged in a two-phase task. Phase 1: Absorb the information from the input-output pairs in the <demonstrations>. Phase 2: Use that context, combined with current query and your own database of knowledge, to accurately complete the response.

Response: '''