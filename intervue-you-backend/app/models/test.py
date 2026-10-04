from session import InterviewSession, AnswerEvaluation

# Fix 2: case/validation
try:
    bad = InterviewSession(category="Frontend", difficulty="Junior")
    print("Normalized to:", bad.category, bad.difficulty)   # should print: frontend junior
except ValueError as e:
    print("Unexpected error:", e)

try:
    InterviewSession(category="frontend", difficulty="expert")  # not in ladder
except ValueError as e:
    print("Correctly rejected:", e)

# Fix 1: historical difficulty tracking
s = InterviewSession(category="nysc-trainee", difficulty="general")
s.add_question("What is semantic HTML?")
s.add_evaluation(AnswerEvaluation(
    score=9, technical_accuracy="excellent", communication_clarity="good",
    feedback="Great answer.", should_follow_up=False,
    follow_up_question=None, next_question_difficulty="harder",
))
s.add_question("Explain closures in JavaScript.")  # now asked at "mid"

print(s.scores_by_difficulty())
# should print something like: {'junior': [9]}
# (second question has no score yet, so it won't appear until evaluated)