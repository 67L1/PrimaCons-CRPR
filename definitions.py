LABELS = ["World", "Sports", "Business", "Sci/Tech"]
EPS=1e-7
SEED=20261013
FINE_CATEGORIES = [
    "Violence", "Physical Harm", "Sexual Content", "Human Trafficking",
    "Privacy Violation", "Cybercrime", "Economic Crime", "White-Collar Crime",
    "Insulting Behavior", "Discriminatory Behavior", "Mental Manipulation", "Psychological Harm",
]
MEDIUM_PAIRS = [
    ("Physical Violence and Injury", "Violence", "Physical Harm"),
    ("Sexual Exploitation and Sexual Harm", "Sexual Content", "Human Trafficking"),
    ("Privacy and Cyber Abuse", "Privacy Violation", "Cybercrime"),
    ("Economic and Professional Crime", "Economic Crime", "White-Collar Crime"),
    ("Social Degradation and Discrimination", "Insulting Behavior", "Discriminatory Behavior"),
    ("Psychological Abuse and Harm", "Mental Manipulation", "Psychological Harm"),
]
FINE=FINE_CATEGORIES
MEDIUM=MEDIUM_PAIRS
