import json

class WindowScorer:
    def __init__(self, window_size: int = 30, step: int = 10):
        """
        window_size: 每个窗口的字符数
        step: 滑动步长（字符）
        """
        self.detector = detector
        self.window_size = window_size
        self.step = step

    def score_text(self, text: str) -> List[WindowScore]:
        n = len(text)
        windows: List[WindowScore] = []

        if n == 0:
            return windows

        if n <= self.window_size:
            prob = self.detector.get_ai_prob(text)
            windows.append(WindowScore(0, n, prob))
            return windows

        i = 0
        while i < n:
            start = i
            end = min(i + self.window_size, n)
            snippet = text[start:end]
            prob = self.detector.get_ai_prob(snippet)
            windows.append(WindowScore(start, end, prob))

            if end == n:
                break
            i += self.step

        return windows
         
text = """{"text": ", LV, EE)Belgique / BelgiëDeutschlandFranceNederlandSchweizUSAUnited KingdomInternational Choose your language Switch Call us: +1 888 988 5661 Contact us Book Classes My Account Checkout Shop Buy your bellicon® Buy accessories Ring ropes bellicon® accessories DVDs Introducing the bellicon® Which bungee strength? Which accessories are right for me? Training Academy Education bellicon Basic bellicon Move bellicon Bounce bellicon Circle Book Courses Exercise Classes Overview bellicon Move bellicon Bounce bellicon Circle bellicon Studio Chicago Studio Class Booking FAQs and Policy Careers Studio Blog Quality Best Built Best Bounce Compare Brands Expert Testimonials Customer Testimonials Media coverage Benefits bellicon® Basics Fitness benefits Overview Power Cardio Flexibility Balance Relaxation Health benefits Overview Back Pain Osteoporosis Lymph & Edema Osteoarthritis Weight Control Pelvic Floor Dysfunction Menopause Incontinence Pregnancy Postpartum Fibromyalgia Chronic Fatigue Syndrome Multiple Sclerosis Neurological Diseases Cancer Sports Medicine Rehabilitation Dancing & Performing Arts Senior Fitness Children's Classes About Us History Mission Statement Our Story Team The bellicon® Difference Bungee Technology Quality Control Safety Environmentally Friendly Research and Education Contact us Locations Chicago Studio Retail Locations News Press Releases Media Coverage Testimonials FAQs Terms & Conditions Privacy Policy Cookie policy Legal information © Bellicon LLC. All rights reserved. All product names are trademarks or registered trademarks of their respective owners. The bellicon® and bellicon Move®", "labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "out_of_domain": true}
"""

text_json = json.loads(text)

for k, v in text_json.items():
    print(k, v)

print(len(text_json["text"].split()))
print(len(text_json["labels"]))

label_0 = []
label_1 = []
for i, label in enumerate(text_json["labels"]):
    if label == 1:
        label_1.append(text_json["text"].split()[i])
    else:
        label_0.append(text_json["text"].split()[i])

print(" ".join(label_0))
print(" ".join(label_1))