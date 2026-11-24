# AI collaboration

**AI used:** Claude (claude.ai, Claude Opus 5.5), from the very first prompt to the final docs.
A second, separate Claude chat was used once to cross-check ideas (Decision 8).

## Full chat history

- Main chat (from my first prompt, where I uploaded the brief PDF): **TODO: paste share link**
- Second chat used to cross-check "Unknown" handling: **TODO: paste share link**

To make a share link: open the chat on claude.ai → **Share** (top right) → **Copy link**.

## Who did what

The AI wrote most of the code and the first drafts of the docs. I chose the model and the tech stack, ran every test on my own laptop with my own downloaded data, and pasted the real results and errors back into the chat. I questioned results that looked wrong, checked one AI's claims against the actual code, and decided which suggestions to keep or reject. The commits are pushed under my name; this file is where the AI's part is declared.

## How we reached a shared understanding

1. **Same requirements first.** My first prompt uploaded the brief PDF and asked the AI to give me the full assignment, so we both worked from the same list: offline only, 50-person folder named by file, a group photo of about 10, strangers marked "unknown", results shown on the photo and as a list, UI scored, and the four required docs.
2. **Options before code.** Before anything was built, I asked *"what are the options"* and got a comparison of recognition models and app shapes. Then I chose (Decisions 1–2).
3. **Real results, not assumptions.** Each time I ran something, I pasted the real output back (errors, the full test results). The AI explained what it meant and we decided together what to change. Twice the problem turned out to be in the *test*, not the app (Decisions 5–6).
4. **Checking the AI against the code.** When a second chat described my project, I asked *"is this how we did?"* instead of trusting it. It had invented file names and a threshold (Decision 8).

---

## Decision 1: recognition model
**AI suggested:** four offline options: InsightFace `buffalo_l` (most accurate, but the weights are non-commercial only), OpenCV YuNet + SFace (light, commercially usable, weaker on small faces), `face_recognition`/dlib (older, Windows install problems) and DeepFace (heavy TensorFlow wrapper). It recommended InsightFace with SFace as a fallback.
**I decided:** InsightFace.
**Why:** the live demo runs on photos we have never seen, so accuracy on hard faces matters most, and the non-commercial licence is fine for an assignment.
**Evidence:** Excerpt 1. Comparison table in RESEARCH.md §2.

## Decision 2: tech stack
**AI suggested:** a local web app (Python backend serving a page on `localhost`) rather than Streamlit/Gradio (generic look) or a desktop app (packaging takes too long). It didn't pick frameworks.
**I decided:** **FastAPI** for the backend and **React** for the frontend.
**Why:** the UI is scored, so I wanted full control over it, especially the names drawn on the photo. Python keeps the ML simple, and everything stays on my laptop, which the offline rule requires.
**Evidence:** Excerpt 1. `backend/app/main.py`, `frontend/src/App.jsx`.

## Decision 3: `buffalo_l`, not `buffalo_s`
**AI suggested:** the large pack `buffalo_l`, loading only 2 of its 5 models (detector + recogniser).
**I decided:** I asked *"did we use bufflo_l or s?"* to check, then kept `buffalo_l` after seeing the numbers.
**Why:** on easy photos the two are almost equal (LFW 99.83 vs 99.70), but on the hard MR-ALL benchmark `buffalo_l` is far better (91.25 vs 71.87). It still takes only about 3 s per group photo on a CPU.
**Evidence:** InsightFace model guide (linked in RESEARCH.md). `scripts/download_models.py` keeps only `det_10g.onnx` and `w600k_r50.onnx`.

## Decision 4: test data
**AI suggested:** no public dataset matches the task (a named folder plus group photos of the same people), so build one. It suggested LFW from Kaggle and wrote a script that makes a 50-person folder and 5 group photos (7 known people + 3 strangers each), using a *different* photo of each person in the group than in the folder.
**I decided:** asked for the scripts, downloaded LFW myself, and ran them on my laptop.
**Why:** the demo uses 50 people with strangers mixed in, so my test should look the same. Using a different photo of each person keeps the test honest.
**Evidence:** `scripts/make_lfw_test.py`, `scripts/check_practice.py`. Results in Excerpt 2.

## Decision 5: "0/5 correct" was the test's fault, not the app's
**What happened:** my first LFW run printed `TOTAL 0/5 photos fully correct` and `17 missed`. I pasted the whole output into the chat.
**AI suggested:** reading the details, every known person (35/35) was named and every stranger (15/15) was Unknown, with 0 wrong names. The "17 missed" were **extra background faces**: LFW photos come from news pictures and often show a second person at the edge, which the app correctly called Unknown. It proposed fixing the *test scripts*, not the app.
**I decided:** fix the scripts (crop each tile to its centre face; count background faces separately) and **leave the app and threshold unchanged**.
**Why:** the app was doing the right thing. Changing the threshold to make a broken test pass would have made the real app worse.
**Evidence:** Excerpt 2. Commit "Test scripts: crop LFW tiles to one face, count background faces separately…".

## Decision 6: the threshold for "Unknown" (0.35)
**AI suggested:** measure it instead of guessing. On 300 labelled pairs, same-person scores were ≥ 0.54 and different-person scores ≤ 0.23, and one real group photo had a true match at only 0.40. It proposed 0.35, plus a "Match strictness" slider in the UI.
**I decided:** keep 0.35 after my own LFW test, and keep the slider.
**Why:** on my run, known people scored 0.50–0.83 and the highest stranger only 0.21, so 0.35 has a clear safety gap on both sides. Not one stranger was given a name, which is the worst possible error.
**Evidence:** `scripts/calibrate.py`; LFW TOTAL block (Excerpt 2); RESEARCH.md §3.

## Decision 7: one name per face
**AI suggested:** Hungarian assignment (`scipy`), so the same name can't appear on two faces and each name goes to the best-fitting face.
**I decided:** keep it.
**Why:** one person can't be in a photo twice, and "two Priyas" on screen would look broken in the demo.
**Evidence:** `backend/app/matching.py`; `tests/test_logic.py::test_one_name_per_face`.

## Decision 8: rejecting extra "Unknown" rules from a second AI
**What happened:** I asked a second chat how to handle unknown people. It said my code used `evaluate.py`, `prepare_lfw_test.py` and a 0.40 threshold, and offered to add a **margin check**, a **"low quality" label** and **Unknown_1/Unknown_2 grouping**. I pasted its answer into the main chat and asked *"is this how we did?"*
**AI said:** the main ideas matched, but the file names and the threshold didn't exist in my repo (that chat had never seen my code). Based on my own results, it advised against all three additions.
**I decided:** add none of them.
**Why:**
- *Margin check:* I already had 0 strangers named, and it could turn correct matches between lookalikes in the folder (for example siblings) into Unknown.
- *Low-quality label:* it could hide real people at the back of a group photo.
- *Unknown grouping:* only matters across several photos, and the demo uses one.

I also learned not to copy claims about my project from a chat that hasn't seen the code.
**Evidence:** Excerpt 3. RESEARCH.md §3 table.

## Decision 9: bugs found by testing
**What happened:** testing the LFW script showed people named "Person Number 1", "Person Number 3"… all merged into **one** person. The AI's file-name cleanup was stripping trailing numbers (meant for `Priya Sharma 2.jpg` copies).
**AI suggested:** only strip what operating systems add to duplicate files, `(2)` and `- Copy`, and never plain numbers.
**I decided:** accept the fix.
**Why:** merging two different people is much worse than missing a duplicate photo. Office folders could easily contain names like "Intern 1" and "Intern 2".
**Evidence:** commit "Fix: don't strip trailing numbers from names…"; `tests/test_logic.py::test_names_from_files`.

## Decision 10: keep real people's photos out of GitHub
**AI suggested:** a `.gitignore` that excludes the face models (too big for GitHub) and all test photos.
**I decided:** accepted it. The repo holds the scripts that rebuild the test data, not the photos.
**Why:** LFW and the practice photos are of real people and shouldn't be republished, and anyone can rebuild them with one command.
**Evidence:** `.gitignore`.

---

## Excerpt 1: choosing the model and the stack

> **Me:** oky what are the options
>
> **AI:** *(compared InsightFace `buffalo_l`, OpenCV YuNet + SFace, `face_recognition`/dlib and DeepFace, then local web app vs Streamlit/Gradio vs desktop app)* "**My pick:** A as the main engine, with B kept as a documented fallback. The demo on their unseen photo is a big part of the score, so accuracy matters most." … "**My pick:** a local web app, started with one double-click script."
>
> **Me:** okay use a insihtface, 2fastaapi and react as frontend

## Excerpt 2: questioning the "0/5" result

> **Me:** *(pasted the full output of `check_practice.py lfw_test`)*
> `TOTAL 0/5 photos fully correct` · `known people named correctly : 35` · `strangers marked Unknown : 15` · `known person missed (Unknown): 17` · `stranger given a name : 0` · `wrong person's name : 0`
>
> **AI:** "These results are actually perfect for the app. The test was counting wrong. … **All 35 known people were named correctly** … **All 15 strangers were marked Unknown.** … **0 wrong names.** … LFW photos are cropped from news pictures, so many tiles contain **a second person in the background** … The app found those extra faces and correctly said 'Unknown'. My checker wrongly counted them as missed known people. … **The threshold is fine.** Known people scored 0.50–0.83, and the highest stranger scored 0.21."

## Excerpt 3: checking a second AI against my code

> **Me:** *(pasted the second chat's answer recommending a margin check, quality check and Unknown_1/2 grouping)* … "Want me to add the margin check, the quality check and the Unknown_1/2 grouping to the code? is this how we did ?"
>
> **AI:** "Partly. The main ideas match what we built, but several details in that answer don't match your code. **It looks like that answer came from a different chat that hasn't seen your actual project.** It invents file names (`prepare_lfw_test.py`, `evaluate.py`, `--unknown-ratio`) and a threshold (0.40) that aren't in your repo. … **Margin check:** I'd say **no**. You have no false accepts to fix, and it adds a new risk…"
