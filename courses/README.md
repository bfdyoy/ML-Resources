# Courses: how the best teach, and how this repo teaches

[← Back to the README](../README.md) · Syllabi: [Python for ML](00-python-for-ml.md) · [Core ML](01-core-ml.md) · [Deep Learning](02-deep-learning.md) · [LLMs & GenAI](03-llms-genai.md) · [ML in Production](04-ml-in-production.md) · [Computer Vision](05-computer-vision.md) · [Flashcards](flashcards/README.md)

The [paths](../paths/) say *what* to learn and in what order. This folder is about **how to learn it**. It turns each path into a
**week-by-week course** that uses the methods the best Python, ML, and DL courses rely on, and the learning-science results behind them.

This page has three parts:

1. **What we studied:** twenty well-known courses, and the signature technique of each.
2. **The learning science underneath:** why those techniques work.
3. **The design we adopted:** the weekly loop every syllabus follows, and a study routine you can use.

---

## 1. Twenty courses, and what each does best

| Course | Format & philosophy | Signature technique | What we borrow |
|---|---|---|---|
| [fast.ai: Practical Deep Learning](https://course.fast.ai/) + [fastbook](https://github.com/fastai/fastbook) | **Top-down, code-first.** Lesson 1 trains a state-of-the-art image classifier, lesson 2 deploys it, and only then does lesson 3 open up how a neural net works. Based on David Perkins' *Making Learning Whole*: "teach the whole game". | Every chapter ends with a **questionnaire** (the answers are in the text) and **"further research"** (they aren't). The course explicitly coaches learners through being stuck. | A **"whole game" first week** in every course. Questionnaires become our self-checks, and "further research" becomes our stretch tasks. |
| [Andrew Ng: ML Specialization](https://www.deeplearning.ai/specializations/machine-learning) & [DL Specialization](https://www.deeplearning.ai/specializations/deep-learning) | **Bottom-up, intuition-first.** Short videos, each concept shown visually, then in code. The math lives in optional videos. Three short courses of 3–4 weeks each. | **Optional labs**: a stress-free notebook for each concept, run line by line, before the graded lab. | Low-stakes **"run it and poke it"** cells in every study note before the graded lab. |
| [*Machine Learning Yearning*](https://home-wordpress.deeplearning.ai/wp-content/uploads/2022/03/andrew-ng-machine-learning-yearning.pdf) (Ng) | Short chapters on *strategy*: dev/test sets, error analysis, bias/variance decisions. | It teaches **what to do next** on a project, not algorithms. | The [playbook](../playbook/README.md)'s decision guides and debugging tables. |
| [Stanford CS229](https://cs229.stanford.edu/main_notes.pdf) | **Rigorous derivations**, with problem sets that are mostly proofs plus some code. | Lecture notes that derive everything from first principles. | Derivations in the [study notes](../notes/README.md), kept to what intuition needs. |
| [Stanford CS231n](https://cs231n.github.io/) | Course notes plus **from-scratch NumPy assignments**: kNN, SVM, softmax, then layer-by-layer nets with **numeric gradient checks**, before PyTorch. | Implement it yourself, and check it against a numeric gradient. | The [labs](../labs/README.md): from-scratch implementations with automatic checks. |
| [Stanford CS336](https://cs336.stanford.edu/) ([assignment 1](https://github.com/stanford-cs336/assignment1-basics)) | Build an LLM **from scratch**: tokenizer, transformer, optimizer, training loop, systems. | **Test-driven assignments**: every test starts out failing with `NotImplementedError`, and you implement until the suite passes. | Every [lab](../labs/README.md) ships as stubs plus a failing test suite. |
| [Karpathy: Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) | Long code-along videos that build **one artifact progressively**: micrograd → bigram → MLP → BatchNorm → manual backprop → WaveNet → GPT → tokenizer. | **"Backprop ninja"**: pause the video and derive it yourself, then compare. The same project grows lecture after lecture. | **Spiral projects**: the course projects grow week by week instead of restarting. |
| [Dive into Deep Learning](https://d2l.ai/) | A book in which every section is runnable. Concepts arrive **just in time**: "you will learn concepts at the very moment that they are needed". | **Two implementations of everything**: from scratch, then the concise framework version. | Labs build from scratch, then the notes show the library call that replaces it. |
| [Neuromatch Academy: Deep Learning](https://deeplearning.neuromatch.io/tutorials/intro.html) | An intensive 3-week cohort. Learners are matched into **pods of about 15 with a TA**, and each day splits into tutorial time and [**project time**](https://deeplearning.neuromatch.io/projects/README.html). | A project starts in **week 1**, not at the end. There are dedicated "discussion" days on ethics and the limits of DL. | **Project milestones from week 1**, and a weekly "discuss it" prompt for your study partner or journal. |
| [ARENA](https://github.com/callummcdougall/ARENA_3.0) | A residential bootcamp: day-long exercise sets with **tests and solutions**, done in **pair programming**. | Exercises chunked into small functions, each with a test. Bonus sections let you go further. | The lab format (small functions, tests, bonus questions) and a pair-programming suggestion. |
| [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) | Practical chapters built on the library you'll actually use. | An **end-of-chapter quiz** on every chapter. | A weekly retrieval quiz in every course. |
| [Kaggle Learn](https://www.kaggle.com/learn) ([Python](https://www.kaggle.com/learn/python), [pandas](https://www.kaggle.com/learn/pandas), [Data Visualization](https://www.kaggle.com/learn/data-visualization)) | Micro-courses of about 4–5 hours: short lessons, each paired with a **graded exercise notebook on real data**, entirely in the browser. | Immediate application: read for 5 minutes, then code for 15. | Short read → do cycles in the Python course. |
| [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) | Short modules with **interactive widgets** and inline exercises. | Learn by manipulating a live visualization. | The "Intuition" step of every lesson (interactive essays first). |
| [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) | A cohort course with homework every module. **Deployment comes in module 5**, before trees and deep learning. A midterm project plus two capstones. | Projects are **peer-reviewed** (you review 3 others). **Learning in public** is encouraged. | Deploy early, a **midterm** project in every course, and peer review / learning-in-public suggestions. |
| [Made With ML](https://madewithml.com/) | One product-oriented ML system, taken from design to CI/CD. | **First principles, then best practices**, on a single running project. | The production course follows one system end to end. |
| [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/) | Lectures and labs on **everything around the model**: data, deployment, monitoring, teams. | The "ML system", not the "ML model", is the unit of study. | The production syllabus structure. |
| [Microsoft ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners) | 12 weeks, 26 lessons, **project-based**, with a **common theme** (data from world cuisines and regions). | A **pre-lecture quiz** (it sets your intention) and a **post-lecture quiz** (it consolidates), three questions each. | A **warm-up quiz** before each week and **review questions** after it. One running dataset per course. |
| [Software Carpentry: Programming with Python](https://swcarpentry.github.io/python-novice-inflammation/) | Live-coded workshop lessons. **One dataset threads through every episode** (clinical-trial inflammation data). Each episode lists its objectives, questions, and key points, with challenges and inline solutions. | Teaching time and exercise time are budgeted separately. Learners type along. | Objectives and questions up front, key points at the end, and a single dataset for the Python course. |
| [CS50P: Programming with Python](https://cs50.harvard.edu/python/syllabus/) | 9 weeks. Lecture → notes → **problem sets with automatic checkers** (check50) and a style checker. A final project. | Instant, automatic feedback on correctness. | Automatic checks: `pytest` in the labs, and `check_notes.py` for the notes. |
| [MIT: The Missing Semester](https://missing.csail.mit.edu/) | One lecture per tool: shell, git, editors, debugging and profiling. | It teaches **tool fluency**, which degree courses assume you already have. | The engineering lesson of the Python course (PY-05). |

Also studied: [MIT 6.S191](https://introtodeeplearning.com/) (a one-week intensive with [Colab labs](https://github.com/MITDeepLearning/introtodeeplearning)), the [UvA DL notebooks](https://uvadlc-notebooks.readthedocs.io/), [CS224n](https://web.stanford.edu/class/cs224n/), and Maxime Labonne's [LLM course roadmap](https://github.com/mlabonne/llm-course), which splits the field into *fundamentals → scientist → engineer*.

### Patterns across all of them

- **Everyone builds.** No serious course is lecture-only. The ratio of *doing* to *watching* is at least 1:1, and often 3:1.
- **The two camps agree more than they argue.** Top-down (fast.ai) and bottom-up (Ng, CS231n) courses both **alternate**: fast.ai digs into internals by lesson 3, and CS231n uses PyTorch after the NumPy assignments. The real difference is only *which pass comes first*.
- **Feedback is automatic and immediate**: check50, test suites, end-of-chapter quizzes, Kaggle's graded notebooks.
- **Projects start early and grow** (Neuromatch, Zoomcamp's midterm, Karpathy's single growing codebase).
- **The best materials anticipate being stuck**: fast.ai's "rewind to the last point you understood", ARENA's hints and solutions, Zoomcamp's FAQ.

---

## 2. The learning science underneath

The techniques above work for reasons that cognitive science has studied for decades.
[*Teaching Tech Together*](https://teachtogether.tech/) (Greg Wilson; free, CC-BY) summarizes the evidence for technical teaching, and the
[Learning Scientists](https://www.learningscientists.org/blog/2016/8/18-1) distil it into six strategies ([posters](https://www.learningscientists.org/posters)):

| Strategy | What it means | How the courses here use it |
|---|---|---|
| **Spaced practice** | Ten hours over five days beat ten hours in one. Review a little old material every session. | Each week opens with a warm-up on material from 1 and 3 weeks earlier. There are [flashcard decks](flashcards/README.md) for daily spaced review. |
| **Retrieval practice** | Recall, not re-reading, builds memory: "the limiting factor … is not retention but recall". Solve it once from memory, then with resources, and compare. | Answer each lesson's self-check **before** opening its answer sketches. Weekly quizzes. Flashcards. |
| **Interleaving** | Mix topics (A-B-C-B-A-C) instead of blocking them (A-A-B-B-C-C). It feels harder, which is the sign that it works. | Warm-ups mix earlier topics. Labs revisit earlier functions. Review weeks mix the whole course. |
| **Elaboration** | Explain *why* an answer is right, and why the plausible alternatives are wrong. Connect new ideas to old ones. | The "Where we are" and "Where this leads" bridges in every note, and a weekly explain-it-back exercise. |
| **Concrete examples** | Abstract ideas stick when tied to specific cases. | A worked example with real numbers in every note. |
| **Dual coding** | Combine words with visuals. | The interactive "Intuition" step comes first in every lesson. |

Three more results shape how the material is *built*:

- **Cognitive load:** working memory is small. Novices learn best with **guidance**: worked examples, **faded examples** (progressively more blanks to fill in), **Parsons problems** (reorder given lines of code), and **labeled subgoals** (name the steps of a procedure). Discovery-style "figure it out" only pays off once you have prior knowledge (Kirschner, Sweller & Clark, as summarized in *Teaching Tech Together*).
  → Lab stubs come with named subgoals in their docstrings and faded hints, and get less scaffolded as each course progresses.
- **Formative assessment and misconceptions:** learners improve fastest when *broken mental models* are surfaced and corrected early. Good quiz questions have wrong answers that diagnose a specific misconception.
  → The "Pitfalls & misconceptions" section of every note, and the *(debug)* question in every lesson.
- **Spaced repetition makes memory a choice:** Michael Nielsen's [*Augmenting Long-term Memory*](https://augmentingcognition.com/ltm.html) estimates a few minutes of total review per card to remember it for years. His rule is to make a card for anything worth 10 minutes of future time. Writing *good* prompts is a skill of its own ([Matuschak](https://notes.andymatuschak.org/Writing_good_spaced_repetition_memory_prompts_is_hard)).
  → The [flashcard decks](flashcards/README.md), generated from every note's cheat sheet and self-check.

---

## 3. The design we adopted

### 3.1 Three passes over every big idea (the spiral)

| Pass | Mode | Where |
|---|---|---|
| 1. **Use it** | The whole game: a library, a pretrained model, a real dataset, in week 1 | Each course's week 1 |
| 2. **Understand it** | Intuition → derivation → worked numbers | The lesson + its [study notes](../notes/README.md) |
| 3. **Build it** | From scratch, test-driven, then compare with the library | The [labs](../labs/README.md) |

Then **use it again**, better, in the course project. Top-down and bottom-up, alternating, as every strong course actually does.

### 3.2 The weekly loop (about 8 hours)

| When | Activity | Time | Technique |
|---|---|---|---|
| Start of week | **Warm-up quiz**: 5 questions from 1 and 3 weeks ago, answered from memory | 15 min | Spacing, retrieval, interleaving |
| | **Study notes** for the week's lesson (step 0), running the code cells | 1 h | Worked examples, concrete numbers |
| | **Lesson steps**: intuition → reading → build | 4–5 h | Dual coding, guided practice |
| | **Lab**: make the failing tests pass | 1–2 h | Test-driven feedback, faded scaffolding |
| End of week | **Self-check from memory**, then compare with the answer sketches | 30 min | Retrieval, hypercorrection |
| | **Explain it back**: 5 sentences plus one sketch on the week's idea, as if teaching a colleague (post it if you learn in public) | 15 min | Elaboration, dual coding |
| | **Project milestone** (see each syllabus) | 1 h | Whole game, transfer |
| Daily (optional) | **Flashcards**: 10 minutes | 10 min | Spaced repetition |

Every 4–6 weeks there's a **review week**: an interleaved quiz over everything so far, a re-run of earlier labs from a blank file, and a project checkpoint.

### 3.3 Projects: a midterm and a capstone, both growing from week 1

Every syllabus has a **running project** that starts in week 1 (the whole game), passes a **midterm** checkpoint, and ends as a **capstone** with a rubric.
The rubric is published up front, so you know what "good" looks like before you start. That's transparent summative assessment.

### 3.4 When you're stuck (the protocol)

Adapted from fastbook's advice and the Carpentries:

1. **Rewind** to the last point where you were *sure* you understood. Re-read forward slowly from there to find the first unclear step.
2. **Shrink it:** reproduce the confusion in the smallest possible code cell, with the smallest numbers.
3. **Switch the representation:** read a different resource for the same idea (each lesson's "Go deeper"), or draw it.
4. **Time-box it** (about 45 minutes), then **move on and mark it**. Things often click after seeing what comes next. Come back during the review week.
5. **Ask precisely:** "I expected X because Y, but I got Z". Writing that sentence often solves the problem by itself.

### 3.5 Study with others (optional, and recommended)

The cohort courses (Neuromatch pods, Zoomcamp, ARENA pairs) show that peers help: **pair-program the labs**, **swap capstones for review** using the rubric (review at least two), and **learn in public** (a short weekly post of your explain-it-back).

---

## The syllabi

| Course | Weeks (at ~8 h/week) | Based on |
|---|---|---|
| [Course 0: Python for ML](00-python-for-ml.md) | 5 | [Path 0](../paths/00-python-for-ml.md) |
| [Course 1: Core ML](01-core-ml.md) | 14 | [Path 1](../paths/01-core-ml-practitioner.md) |
| [Course 2: Deep Learning](02-deep-learning.md) | 11 | [Path 2](../paths/02-deep-learning.md) |
| [Course 3: LLMs & Generative AI](03-llms-genai.md) | 12 | [Path 3](../paths/03-llms-genai.md) |
| [Course 4: ML in Production](04-ml-in-production.md) | 7 | [Path 4](../paths/04-ml-in-production.md) |
| [Course 5: Computer Vision](05-computer-vision.md) | 5 | [Path 5](../paths/05-computer-vision.md) |

The electives work as self-paced modules: take one lesson per week with the same loop.
