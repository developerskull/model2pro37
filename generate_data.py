"""Generate a synthetic SMS dataset (data/sms.csv) with columns: label, text.

Why synthetic? It runs anywhere with no download.
To use the REAL "SMS Spam Collection" dataset from Kaggle/UCI instead, save it as
data/sms.csv with columns `label` (spam/ham) and `text`, then run train.py.
"""
import random
from pathlib import Path

import pandas as pd

random.seed(42)

# ---------- SPAM building blocks ----------
spam_openers = [
    "Congratulations! You have won", "URGENT! You've been selected for", "WINNER! Claim your",
    "Dear customer, you are eligible for", "Limited offer: get", "FREE entry to win",
    "You have been chosen to receive", "Final notice: collect your",
]
spam_prizes = [
    "a Rs 50000 cash prize", "a free iPhone 15", "a free holiday to Goa", "a 5000 Amazon voucher",
    "a brand new bike", "2 lakh loan approval", "a free recharge of Rs 499", "a lucky draw gift",
]
spam_actions = [
    "Call 09876543210 now", "Click http://bit.ly/claim-now", "Reply YES to claim",
    "Text WIN to 56789", "Visit www.free-prize-claim.com", "Call now before it expires",
    "Send your bank details to claim", "Claim within 24 hours",
]
spam_extra = [
    "T&C apply.", "Offer valid today only!", "Don't miss out!", "100% free, no charges.",
    "Your account will be blocked, verify now.", "Act fast!", "",
]
spam_plain = [  # spam that avoids the obvious words
    "Your KYC is expired. Update immediately at http://sbi-verify.co.in or your account will be suspended",
    "Hot singles in your area want to chat. Reply now",
    "Earn Rs 5000 daily working from home. Message us on WhatsApp to join",
    "Your parcel is held at customs. Pay Rs 49 fee at http://track-parcel.in to release",
    "Get cheap medicines without prescription. Order today",
    "Low interest personal loan up to 5 lakh, no documents needed. Call now",
]

# ---------- HAM building blocks ----------
ham_msgs = [
    "Hey, are you coming to college tomorrow?", "Can you send me the notes for DBMS?",
    "I'll call you after the lecture", "Lunch at 1? Canteen as usual",
    "Don't forget the project submission is on Friday", "Happy birthday! Have a great day",
    "Mom said dinner is ready, come home soon", "Where are you? We are waiting near the gate",
    "Did you finish the assignment? Please share", "Reached home safely, thanks for the ride",
    "Meeting rescheduled to 4 pm, please confirm", "Call me when you're free",
    "I won the cricket match today, we scored 120", "Let's study together tonight at my place",
    "Can you pick up some milk on your way back?", "The bus is late, I will be there in 15 minutes",
    "Thanks for the help yesterday, really appreciate it", "Check the group, sir uploaded the slides",
    "Your OTP for login is 482913. Do not share it with anyone", "Free for a quick call at 6?",
    "Congrats on the internship! Treat pending", "Are you free this weekend for the trip?",
    "Please bring your ID card tomorrow", "Haha that was so funny, send me the video",
    "Order delivered. Rate your experience in the app", "Reminder: doctor appointment at 5 pm",
    "Win or lose, we are playing the final tomorrow", "Click the link in the group to join the meeting",
]
ham_names = ["Rahul", "Priya", "Amit", "Sneha", "Kunal", "Rohan", "Neha", "Omkar"]


def make_spam():
    if random.random() < 0.25:
        return random.choice(spam_plain)
    return " ".join([random.choice(spam_openers), random.choice(spam_prizes) + ".",
                     random.choice(spam_actions) + ".", random.choice(spam_extra)]).strip()


def make_ham():
    msg = random.choice(ham_msgs)
    if random.random() < 0.3:
        msg = f"{random.choice(ham_names)}, " + msg[0].lower() + msg[1:]
    if random.random() < 0.2:
        msg += " " + random.choice(["ok?", "please", "thanks", "see you", ":)"])
    return msg


rows = [("spam", make_spam()) for _ in range(500)] + [("ham", make_ham()) for _ in range(1500)]
random.shuffle(rows)
df = pd.DataFrame(rows, columns=["label", "text"])

Path("data").mkdir(exist_ok=True)
df.to_csv("data/sms.csv", index=False)
print(f"Saved data/sms.csv with {len(df)} rows ({(df.label == 'spam').sum()} spam)")
print(df.head())
