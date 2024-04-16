<p align="center">
<img src="./assets/teachmylogo.png" width="250px"></p>
</p>

<br>

<h5 align="center">
  <a href="#about">About</a>  |
  <a href="#how-to-use">How to use</a>  |
  <a href="#how-it-works">How it works</a>  |
  <a href="#authors">Authors</a>
</h5>


# About

TeachMy is an NLP application designed for learners seeking to enhance their study sessions. By transforming PDFs into interactive experiences, it allows users to actively engage with their documents through questions and quizzes. It offers tailored feedback to its users, and it also recommends relevant educational videos from YouTube and courses from Udemy and Coursera, ensuring a comprehensive learning journey. 

&nbsp;


# How to use:

**1. Download the models**

The models can be found here:

```
https://github.com/TechLabs-Berlin/ws24-teach-my-ai-bot/releases/tag/v1
```

```
https://github.com/TechLabs-Berlin/ws24-teach-my-ai-bot/releases/tag/answerv1
```

**2. Place both .pth files into the empty models folder**

**3. Run this code**

```
python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

**4. Go to main.py and run it**


&nbsp;

# How it works:

**1. Upload PDF**

**2. Ask Questions:**
the model works best with factoid questions (short answers) not conceptual questions like: explain to me X

**3. After 3 question you can start quiz:**
but you can also continue as long as you want

**4. Quiz & Submit your answers!**

Share your feedback if you find good / bad examples! 

**Note:** if you run multiple times and notice weird things, open in incognito window and clean your browser cookies and cache if necessary


&nbsp;

# Authors:

**Deep Learning:** Oula Suliman, Sonia, Arpad Dusa

**Data Science:** Bibin Karthikeyan Krishna, Fernanda Portieri
