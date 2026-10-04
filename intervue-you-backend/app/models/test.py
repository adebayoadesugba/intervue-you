from session import InterviewSession, AnswerEvaluation

s = InterviewSession(category="frontend", difficulty="junior")
s.add_question("What is semantic HTML?")
s.add_answer("It's HTML that describes meaning, like <header>.")
s.add_evaluation(AnswerEvaluation(
    score=8, technical_accuracy="good", communication_clarity="good",
    feedback="Solid, could mention accessibility.", should_follow_up=True,
    follow_up_question="Can you give an example of a semantic HTML element?", next_question_difficulty="harder",
))

print(s.difficulty)       # should print "mid"
print(s.pending_follow_up())  # should print the follow-up question
print(s.should_end())     # should print False