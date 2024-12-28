import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load skills dataset
with open("linkedin skill", "r") as file:
    skills_set = set(skill.strip().lower() for skill in file if len(skill.strip()) > 2)

# Define stop words
stop_words = set(stopwords.words("english"))

# Preprocess text
def preprocess_text(text):
    clean_text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    tokens = clean_text.lower().split()  # Tokenize and normalize case
    return [token for token in tokens if token not in stop_words]  # Remove stop words

# Generate n-grams
def generate_ngrams(tokens, n=2):
    vectorizer = CountVectorizer(ngram_range=(1, n), tokenizer=lambda x: x, preprocessor=lambda x: x)
    vectorizer.fit([tokens])
    return vectorizer.get_feature_names_out()

# Match skills
def match_skills(resume_text, skills_set):
    tokens = preprocess_text(resume_text)
    ngrams = generate_ngrams(tokens, n=3)  # Generate unigrams, bigrams, and trigrams
    matched_skills = [ngram for ngram in ngrams if ngram in skills_set]

    # Remove substrings if a longer match exists
    final_matches = set(matched_skills)
    for skill in matched_skills:
        if any(skill in match and skill != match for match in matched_skills):
            final_matches.discard(skill)

    return list(final_matches)

# Example resume text
resume_text = """
About the job
As a global leader in cybersecurity, CrowdStrike protects the people, processes and technologies that drive modern organizations. Since 2011, our mission hasn’t changed — we’re here to stop breaches, and we’ve redefined modern security with the world’s most advanced AI-native platform. Our customers span all industries, and they count on CrowdStrike to keep their businesses running, their communities safe and their lives moving forward. We’re also a mission-driven company. We cultivate an inclusive culture that gives every CrowdStriker both the flexibility and autonomy to own their careers. We’re always looking to add talented CrowdStrikers to the team who have limitless passion, a relentless focus on innovation and a fanatical commitment to our customers, our community and each other. Our University Recruiting program is dedicated to attracting and cultivating the future leaders of this industry. This program offers paid positions for students and recent grads, designed to provide exposure to work that makes an impact while being supported through a structured experience with seasoned professionals. Ready to join a mission that matters? The future of cybersecurity starts with you.

About The Role

The OverWatch Labs (OWL) team builds the platform and tools for our analysts on the OverWatch team to process and hunt (identify potentially harmful activity) through trillions of events per day, and growing. We are looking for an intern who wants to help move the OWL platform forward as we scale even further.

What You'll Do

Collaborate with OverWatch engineering members to build, develop, and maintain operational systems, projects, and tools.
Work with cloud technologies like Amazon Web Services
Build elegant solutions for complex technical problems to help build new components and extend the current system
Learn about our massively scalable distributed architecture
Work in a devops environment where you (and your team) are responsible for the systems you deploy.
Work as part of a distributed team of remote workers across timezones.

What You’ll Need

Currently enrolled at a four university, currently working towards a CS/Engineering degree, graduating between December 2025- August 2026
Development experienced with one or more of the following: Python, C/C++, Java, Rust, or Go
Able to communicate, collaborate, and work effectively in a distributed team.
Can think about and write high quality code and can demonstrate that capability, be it through job experience, schoolwork, or contributions to community projects

What You Can Expect

Remote-friendly and flexible work culture
Market leader in compensation and equity awards
Paid holidays (including birthday holidays) and 401k matching
Professional development opportunities including workshops, tech talks, and Executive Speaker Series
Assigned mentors from across the company for continuous support and feedback
Participation in companywide initiatives including ERGs, FalconFIT, Wellness Programs, and Employee Assistance Program
Employee Resource Groups, geographic neighbourhood groups and volunteer opportunities to build connections
Vibrant office culture with world class amenities
Ownership of impactful projects that move the company forward
Great Place to Work Certified™ across the globe

CrowdStrike is proud to be an equal opportunity and affirmative action employer. We are committed to fostering a culture of belonging where everyone is valued for who they are and empowered to succeed. Our approach to cultivating a diverse, equitable, and inclusive culture is rooted in listening, learning and collective action. By embracing the diversity of our people, we achieve our best work and fuel innovation - generating the best possible outcomes for our customers and the communities they serve.

All qualified applicants will receive consideration for employment without regard to race, color, religion, sex, sexual orientation, gender identity, national origin, disability, or status as a protected veteran. If you need assistance accessing or reviewing the information on this website or need help submitting an application for employment or requesting an accommodation, please contact us at recruiting@crowdstrike.com for further assistance.

Find out more about your rights as an applicant.

CrowdStrike participates in the E-Verify program.

Notice of E-Verify Participation

Right to Work


Benefits found in job post

401(k)
"""
matched_skills = match_skills(resume_text, skills_set)
print("Matched Skills:", matched_skills)
