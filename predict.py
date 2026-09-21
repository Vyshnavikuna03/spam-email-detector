import joblib
from train import clean

model = joblib.load("models/spam_model.pkl")

def check(text):
    return "SPAM" if model.predict([clean(text)])[0] == 1 else "NOT SPAM"

tests = [
    "Congratulations! You won a free iPhone. Click here to claim your prize now",
    "Hi, can we move tomorrow's meeting to 3 PM?",
    "URGENT: your account is suspended. Verify your password at http://secure-login.xyz",
    "Please find the attached report for last week's project status.",
]

for t in tests:
    print(f"{check(t):9} <- {t}")