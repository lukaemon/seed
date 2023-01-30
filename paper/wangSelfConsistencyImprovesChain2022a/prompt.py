def math_word_problem_template(example):
    """
    example: {'question': ... , 'answer': ...}
    """
    few_shot = """\
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6. 

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. 

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39. 

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? 
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8. 

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? 
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9. 

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? 
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29. 

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? 
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33. 

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? 
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.
    """

    prompt = f"{few_shot}\n\nQ: {example['question']}\nA:"

    return prompt


def csqa_example2text(d):
    """by ChatGPT with minor modifications, productive!"""
    question = d["question"]
    choices_text = d["choices"]["text"]
    choices_string = ""
    for i, choice in enumerate(choices_text):
        choices_string += f'({chr(ord("a")+i)}) {choice}\n'
    output = f"Q: {question}\nAnswer Choices: \n{choices_string}A:"
    return output


def csqa_template(example):
    """
    {'id': '1afa02df02c908a558b4036e80242fac',
    'question': 'A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?',
    'question_concept': 'revolving door',
    'choices': {'label': ['A', 'B', 'C', 'D', 'E'],
    'text': ['bank', 'library', 'department store', 'mall', 'new york']},
    'answerKey': 'A'}
    """

    few_shot = """\
Q: What do people use to absorb extra ink from a fountain pen? 
Answer Choices: 
(a) shirt pocket 
(b) calligrapher’s hand 
(c) inkwell 
(d) desk drawer 
(e) blotter 
A: The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. So the answer is (e). 

Q: What home entertainment equipment requires cable? 
Answer Choices: 
(a) radio shack 
(b) substation 
(c) television 
(d) cabinet 
A: The answer must require cable. Of the above choices, only television requires cable. So the answer is (c). 

Q: The fox walked from the city into the forest, what was it looking for? 
Answer Choices: 
(a) pretty flowers 
(b) hen house 
(c) natural habitat 
(d) storybook 
A: The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. So the answer is (b). 

Q: Sammy wanted to go to where the people were. Where might he go? 
Answer Choices: 
(a) populated areas 
(b) race track 
(c) desert 
(d) apartment 
(e) roadblock 
A: The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. So the answer is (a). 

Q: Where do you put your grapes just before checking out? 
Answer Choices: 
(a) mouth 
(b) grocery cart 
(c)super market 
(d) fruit basket 
(e) fruit market 
A: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. So the answer is (b). 

Q: Google Maps and other highway and street GPS services have replaced what? 
Answer Choices: 
(a) united states 
(b) mexico 
(c) countryside 
(d) atlas 
A: The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. So the answer is (d). 

Q: Before getting a divorce, what did the wife feel who was doing all the work? 
Answer Choices: 
(a) harder 
(b) anguish 
(c) bitterness 
(d) tears 
(e) sadness 
A: The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness. So the answer is (c)."""

    prompt = f"{few_shot}\n\n{csqa_example2text(example)}"

    return prompt
