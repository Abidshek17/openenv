"""
emails.py
---------
A curated dataset of realistic office emails with ground-truth labels.
Each email has:
  - id         : unique string
  - subject    : email subject line
  - sender     : sender name / address
  - body       : email body text
  - label      : "urgent" | "normal" | "spam"
  - priority   : integer rank (1 = highest priority, lower number = more important)
  - reply_keywords : words a good reply should contain (used by the hard grader)
"""

EMAILS = [
    {
        "id": "e001",
        "subject": "URGENT: Production server is down",
        "sender": "alice@company.com",
        "body": (
            "Hi team, our production server has been unreachable for the past 15 minutes. "
            "Customers are reporting errors. We need someone to look into this immediately. "
            "Please respond ASAP."
        ),
        "label": "urgent",
        "priority": 1,
        "reply_keywords": ["investigating", "team", "shortly", "apologize", "fix"],
    },
    {
        "id": "e002",
        "subject": "Congratulations! You won a FREE iPhone",
        "sender": "noreply@prize-winner.biz",
        "body": (
            "Dear valued customer, you have been selected to receive a FREE iPhone 15! "
            "Click the link below to claim your prize. Offer expires in 24 hours. "
            "Visit: http://totally-not-scam.biz/claim"
        ),
        "label": "spam",
        "priority": 10,
        "reply_keywords": [],
    },
    {
        "id": "e003",
        "subject": "Team lunch on Friday",
        "sender": "bob@company.com",
        "body": (
            "Hey everyone, I'm organising a team lunch this Friday at 12:30pm at The Garden Cafe. "
            "Please let me know by Thursday if you can make it so I can book the right table size. "
            "Hope to see you all there!"
        ),
        "label": "normal",
        "priority": 6,
        "reply_keywords": ["friday", "lunch", "confirm", "attend", "yes"],
    },
    {
        "id": "e004",
        "subject": "Security alert: Unusual login detected on your account",
        "sender": "security@company.com",
        "body": (
            "We detected a login to your account from an unrecognised device in a different country. "
            "If this was not you, please reset your password immediately and contact IT support. "
            "Time of login: 2024-01-15 03:42 UTC. Device: Unknown Linux. Location: Eastern Europe."
        ),
        "label": "urgent",
        "priority": 2,
        "reply_keywords": ["password", "reset", "IT", "security", "account"],
    },
    {
        "id": "e005",
        "subject": "Q3 report review — please comment by EOD",
        "sender": "carol@company.com",
        "body": (
            "Hi, I've shared the Q3 financial report in the shared drive. "
            "Could everyone please add their comments and approvals by end of day today? "
            "The board meeting is tomorrow morning and we need it finalised."
        ),
        "label": "urgent",
        "priority": 3,
        "reply_keywords": ["review", "comments", "today", "report", "board"],
    },
    {
        "id": "e006",
        "subject": "Newsletter: Top productivity tips for 2024",
        "sender": "newsletter@productivityhacks.io",
        "body": (
            "Welcome to this month's productivity newsletter! "
            "This week: 10 tips to supercharge your mornings, "
            "the best apps for deep work, and a special discount on our premium course. "
            "Unsubscribe at any time."
        ),
        "label": "spam",
        "priority": 9,
        "reply_keywords": [],
    },
    {
        "id": "e007",
        "subject": "Follow-up: Client proposal feedback",
        "sender": "david@company.com",
        "body": (
            "Hi, just following up on the proposal I sent to the Acme Corp client last week. "
            "Have they responded with any feedback yet? We are hoping to close the deal before "
            "the end of the month. Let me know what you hear."
        ),
        "label": "normal",
        "priority": 5,
        "reply_keywords": ["feedback", "client", "proposal", "update", "Acme"],
    },
    {
        "id": "e008",
        "subject": "Office supplies order — approval needed",
        "sender": "eve@company.com",
        "body": (
            "I've put together the monthly office supplies order. "
            "Total cost is $245. Could a manager please approve the purchase order "
            "in the procurement system when you get a chance? No rush, just before end of week."
        ),
        "label": "normal",
        "priority": 7,
        "reply_keywords": ["approved", "approve", "procurement", "order", "supplies"],
    },
    {
        "id": "e009",
        "subject": "Data breach — customer PII exposed",
        "sender": "frank@company.com",
        "body": (
            "CRITICAL: We have identified a potential data breach. "
            "Customer names, emails and phone numbers from our CRM may have been accessed "
            "by an unauthorised third party. Legal and IT need to be looped in immediately. "
            "Do not discuss over unencrypted channels."
        ),
        "label": "urgent",
        "priority": 1,  # tied for highest priority
        "reply_keywords": ["legal", "IT", "breach", "contain", "escalate", "immediate"],
    },
    {
        "id": "e010",
        "subject": "Happy Birthday wishes",
        "sender": "grace@company.com",
        "body": (
            "Hi! Just wanted to wish you a very happy birthday! "
            "Hope you have a wonderful day and get to celebrate with family and friends. "
            "See you at the office tomorrow 🎂"
        ),
        "label": "normal",
        "priority": 8,
        "reply_keywords": ["thank", "thanks", "birthday", "appreciate"],
    },
    {
        "id": "e011",
        "subject": "Make $5000/week working from home!",
        "sender": "money@easy-cash-now.net",
        "body": (
            "Are you tired of your 9-5 job? Our proven system lets anyone earn $5000 per week "
            "working just 2 hours a day from home. No experience needed! "
            "Click here to get started: http://easy-cash-now.net/join"
        ),
        "label": "spam",
        "priority": 10,
        "reply_keywords": [],
    },
    {
        "id": "e012",
        "subject": "Reminder: Performance reviews due next week",
        "sender": "hr@company.com",
        "body": (
            "This is a reminder that all manager performance review submissions are due "
            "by next Friday. Please log in to the HR portal and complete the forms for each "
            "of your direct reports. Contact HR if you have any questions."
        ),
        "label": "normal",
        "priority": 4,
        "reply_keywords": ["review", "portal", "complete", "HR", "Friday"],
    },
]
