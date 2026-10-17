# Development Journey — Agent Systems Foundations

*12 September 2026 · [Read the public HTML edition](https://az9713.github.io/11-768-ai-agents-notes/) · [Inspect the editorial study](textbook-style-report.html) · [View the public repository](https://github.com/az9713/11-768-ai-agents-notes)*

This account is reconstructed from the surviving request sequence, the four authored chapters, their source manifest, the 12-reference style report, and the final validation results. The complete command-by-command record of the earlier drafting sessions is unavailable here. Where that record is missing, this document describes an observable revision or an explicit reader request rather than inventing a tool trace.

## 1. The first brief

The project began with a link to the [CMU 11-768 playlist](https://www.youtube.com/playlist?list=PLSN0qpDfUvTM) and this request:

> this is the playlist for CMU 11-768 on agents.
> at this moment, it has 4 video lectures.
>
> please convert the transcripts to a set of lecture notes.
> use the transcript as "spine". do research on the topics/subtopics discussed in the lecture.
> add your research findings to the lecture note to harden the quality of the teaching material.
>
> Prof. neubig has only one hour for the lecture so there must be materials left out. he could not go in depth on selected topics. your research is to fill in those gaps and also to address topics that could have been included in a say 2-hour lecture. Feel free to include code and math expressions in Latex. i like precise definitions using math and logic even if materials are not presented in that way in lecture.
>
> look for Claude skill "rigorous-explainer" which MAY help in your writing. but feel free to build on that skill. it is the floor not ceiling in writing science/math/engineering learning materials.
>
> target audience are advanced graduates who actually have some experience with agents usage but not development.

The four recordings used for this edition were the first four available when the work began. The [course site](https://www.cmu-agents.com/) and the [playlist](https://www.youtube.com/playlist?list=PLSN0qpDfUvTM) remain the authoritative places to check what the instructors actually taught and whether more recordings have appeared.

| Source lecture | Recording | Role in this project |
|---|---|---|
| What are agents, and how do they work? | [Lecture 1](https://www.youtube.com/watch?v=UwfjzyLnvMg) | Agent boundary, run loop, capabilities, and system components. |
| Tool use for language-model agents | [Lecture 2](https://www.youtube.com/watch?v=jXChFB4JSyw) | Tool interfaces, call formats, protocols, and parallel operations. |
| Long-context modeling for agents | [Lecture 3](https://www.youtube.com/watch?v=AiwCCvFW1uE) | Attention, context extension, caching, and compaction. |
| Memory and skills for agents | [Lecture 4](https://www.youtube.com/watch?v=6zigF2a-2Pw) | Cross-task memory, skill representation, retrieval, and evaluation. |

The source manifest records that the English caption tracks were automatically generated. The recordings and captions supplied substantive lecture topics, distinctions, and examples as well as their order and timing; course slides and primary papers helped check technical terminology. The public repository links to the original recordings. It does not redistribute captions or slide PDFs.

## 2. What “based on CMU 11-768” means

The chapters incorporate the substance of the first four recordings, not just their syllabus. Chapter 1 develops the lecture's interaction loop and its distinction between training a model and engineering a harness. Chapter 2 takes up the lecture's grocery-cart example, code as a meta-tool, and constrained tool-call generation. Chapter 3 develops the lecture's treatment of key/value caching and context compaction. Chapter 4 develops its account of authored skills, external memory, and skill induction. The chapters re-explain these subjects and add definitions, mathematical models, worked cases, implementation comparisons, code, and exercises drawn from further research and independent analysis.

This distinction matters because the chapters sometimes pursue a topic longer than the recording does and may choose a different formalization. Of 27 numbered main sections, 25 link to relevant recording segments. That count describes navigational coverage, not a percentage of intellectual or textual reuse. A statement in these chapters should not be attributed to the instructors merely because it appears under a timestamped heading. The recording is authoritative for the lecture; the chapter is responsible for its own extensions, assumptions, and errors. The project is neither official CMU course material nor an endorsement by the instructors.

The repository name **Agent Systems Foundations** follows that boundary. It names the subject and the mathematical ambition without presenting the work as a CMU-branded course repository. The source course receives explicit attribution on the reading guide and in every chapter byline. Section timestamps remain navigational links to the recordings, not evidence that every paragraph at that location was spoken in class.

## 3. Iteration before the textbook references

The first deliverable had to be more than a transcript cleanup. It was meant for advanced readers who could use an agent but wanted to understand how one is built. The initial editorial choice was to preserve the lecture topic order while expanding each topic with formal objects, practical harness mechanisms, and external research. The first four chapters consequently share a system model: environment state, observations, history, model proposals, tool actions, authority, context, memory, cost, and verification.

The reader then requested **timestamps**, so each major section gained a link to the corresponding video segment. A subsequent correction was more consequential: the text should not recount what a professor said at a given minute. Timestamps belonged beside section headings; the chapter body needed to read like an independent graduate textbook. Self-referential phrases such as “Here is a precise model” were removed from the target style. The distinction between a timestamp as navigation and a timestamp as narrative structure became a durable rule.

The next request pushed beyond the lecture outline. The chapters were to use real agent-harness documentation and repositories, including Anthropic, OpenAI, Cursor, Hermes, and OpenClaw, while making logic and mathematics the backbone. This led to the run-state and execution-gate model, tool contracts and effect sets, context-cost equations, and memory-selection objectives. Product documentation illustrates possible implementations; it is not treated as a proof of a general agent property. Research papers and provider claims are cited where they support specific assertions.

Navigation was then made explicit: every chapter links backward, forward, and to the reading guide. A temporary private preview was also requested during the iteration. Its network address and operational details are intentionally absent from this public record; the surviving files do not independently establish the exact preview lifecycle.

Two reader corrections exposed weaknesses that a polished outline would have hidden. First, a symbol such as `inScope` had been used without a local definition. The response was to audit terms and symbols before first use, including their domains and assumptions where relevant. Second, a sentence about reporting the evaluation vector had compressed a result and its warning into one breath. The prose rule became: define success, cost, delay, and risk; report them separately; explain in a new sentence why a single rank would hide a trade-off. The correction was about reasoning clarity, not merely sentence length.

## 4. Twelve references changed the editorial method

The reader supplied twelve machine-learning references for a style study. The resulting [HTML report](textbook-style-report.html) records the sampled pages, editions or versions, online copies, and limits of the skim. The collection comprises ten textbooks or monographs, one set of theory lecture notes, and one short technical paper. It was sampled for teaching methods, not copied for distinctive wording or visual design.

| Reference | Habit transferred into the agent chapters |
|---|---|
| Mohri, Rostamizadeh, and Talwalkar, *Foundations of Machine Learning* | Organize a formal question around assumptions, result, proof, example, and exercises. |
| Bishop, *Pattern Recognition and Machine Learning* | Introduce a running example before generalization; make figures and notation serve an argument. |
| Murphy, *Probabilistic Machine Learning: An Introduction* | Use one conceptual lens, restate equations in prose, and test them on numbers. |
| Murphy, *Probabilistic Machine Learning: Advanced Topics* | Define advanced objects systematically and connect derivations with practical diagnostics. |
| Hastie, Tibshirani, and Friedman, *The Elements of Statistical Learning* | Compare methods through assumptions and failure modes, not a single score. |
| James and colleagues, *An Introduction to Statistical Learning with Applications in Python* | Begin with a concrete task, then connect explanation to code, output, and varied exercises. |
| Bach, *Learning Theory from First Principles* | State prerequisites and assumptions; explain where an asymptotic or simplified model stops being informative. |
| Mitchell, *Machine Learning* | Progress from a problem to an algorithmic trace, theory, and practice. |
| Hardt and Recht, *Patterns, Predictions, and Actions* | Connect prediction to consequential action and separate empirical findings from explanation. |
| Shalev-Shwartz and Ben-David, *Understanding Machine Learning* | State quantifiers and guarantee conditions; expose proof gaps instead of smoothing them over. |
| Telgarsky, *Deep Learning Theory Lecture Notes* | Declare the chosen theoretical viewpoint and its omitted cases honestly. |
| Phuong and Hutter, *Formal Algorithms for Transformers* | Give algorithms explicit inputs, outputs, parameters, dimensions, and notation. |

The twelve samples yielded nine working rules: start with a consequential problem; maintain a recurring conceptual model; define locally; derive and interpret claims; reuse a running case; put operational algorithms after the model; compare assumptions and failure modes; state evidence classes and limits; and use exercises with prose that gives distinct ideas room. The style report is a comparative skim, not a judgment on every page of these works.

The reader then asked for the `rigorous-explainer` skill itself to be updated so the twelve-reference findings would override conflicting presentation rules. This made a concrete difference. A figure became useful only if it clarified a dependency, state transition, comparison, or quantitative result; there was no figure quota. A consequential problem outranked a compulsory everyday anecdote. Proofs needed their consequential steps visible, not every routine algebraic manipulation. The reader next required edition information for all twelve references, then replaced local-PDF links with freely accessible online editions. The [MIT machine-learning reading list](https://gradml.mit.edu/info/books/) helped locate several of those editions.

## 5. Applying the revised skill to the four chapters

The final editorial pass used the updated `rigorous-explainer` guidance across all four chapters. It did not restart the project from zero: the notes already had mathematical definitions, citations, and a substantial topic sequence. The pass strengthened the parts where textbook method changed the reasoning.

In Chapter 1, an order-cancellation case now runs from authorization to hidden-state inference and an executable trace. The formal admission gate depends on trusted **evidence** rather than the actual external state, which the harness usually cannot inspect. The chapter also distinguishes a client-side status observation from an atomic server-side precondition check. In Chapter 2, a grocery-cart case separates retrieval, cart creation, and placing an order by their effects and authority. A timeout after cart creation becomes a concrete demonstration of why “unknown outcome” does not mean “no side effect.”

In Chapter 3, a coding agent that must not publish provides a recurring test of context selection and compaction. The worked 100-call example now derives 1,685,000 submitted input tokens without compaction and 565,000 under an explicitly stated 20-call reset rule. The count is not a claim about billed price or task quality. Chapter 4 carries a return-policy skill through fact storage, skill selection, temporal conflicts, and transfer testing. Its Bayesian example shows why a judge with seemingly respectable sensitivity and false-positive rate can have only about 8.3% precision when truly useful skills have a 1% prior frequency.

All four chapters gained worked exercise solutions. The prose was expanded where an equation, an interpretation, and a limitation had been compressed together. The HTML kept its existing dark, print-friendly reading layout and MathJax rendering; the source Markdown remains editable.

## 6. Tools, architecture, and boundaries

| Tool or component | Role in this build |
|---|---|
| Automatically generated caption tracks and public course slides | Established topic coverage and video timing; not published as copied source files. |
| Official papers, specifications, and product documentation | Supplied evidence for research extensions and concrete harness examples. |
| `rigorous-explainer` | Guided definitions, derivations, worked cases, limitations, and exercises after the twelve-reference update. |
| `dev-journey` | Supplied the present retrospective structure, including failed turns and verification limits. |
| Markdown plus `render_notes.py` | Authored the chapters once, rendered standalone HTML, generated chapter navigation and the index. |
| Bundled MathJax | Rendered LaTeX locally without requiring a network connection when reading the HTML. |
| `validate_notes.py`, mathematical checks, and browser checks | Verified package structure, timestamps, links, equations, and rendered pages. |
| `gh` command-line interface | Created and published the public GitHub repository after a staged-file privacy audit. |

The content pipeline is straightforward: public lecture recordings and research sources inform the authored Markdown; the renderer transforms that Markdown into HTML chapters and a reading guide; the public package includes the editorial study and this journey. A small manifest lists each video source and its caption provenance. The raw caption files, JSON caption dumps, downloaded slide PDFs, and private preview configuration stay outside version control. The published repository is a reading artifact and a reproducible renderer, not an archive of the course's media.

No subagents were used in the final revision and publication pass. Work was performed inline. The public document does not record a personal browser session, local account name, home directory, or private network address.

## 7. What went wrong and what the fixes established

1. **An undefined predicate survived an earlier draft.** The reader identified `inScope` as used before definition. The repair was not a glossary alone: each chapter needed local explanations of terms and symbols before their first substantive use. Rule learned: a notation appendix cannot repair an undefined first use.
2. **The first narrative risked sounding like a lecture recap.** A timestamp could easily become “at this minute the lecturer says” prose. The reader rejected that form. The repair confined video links to section headings and made each body section an independent explanation. Rule learned: source navigation and source narration are different functions.
3. **Formalism initially lagged behind the engineering discussion.** Provider examples were informative but could have become a product list. A common model of state, observation, authority, tool effects, and verification made each example answer a defined question. Rule learned: attach every product claim to the part of the model it illustrates.
4. **One evaluation sentence hid four dimensions.** Success, cost, delay, and risk were defined and reported separately before discussing a decision rule. Rule learned: a scalar rank is a choice of weights, not the measurement itself.
5. **The first style report linked to local PDFs.** That would fail for a public reader and expose workspace structure. The report was revised to identify editions and link to online copies, with the MIT reading list used where appropriate. Rule learned: an inspectable citation must resolve outside the author's computer.
6. **The authorization equation initially used the hidden state as if the harness knew it.** The final gate uses trusted evidence. For consequential operations, the downstream service must also check its precondition at commit time to close the observation–execution race. Rule learned: an agent's belief and a service's authoritative state are not interchangeable.
7. **The compaction example stated a saving without computing it.** The final version states whether summaries replace or accumulate, derives both totals, and names the uncounted costs. Rule learned: a numerical example needs a complete accounting convention.
8. **A public release could have implied course authorship or copied the course archive.** Bylines now say “Independent study chapter,” the reading guide explains the course's exact role, and raw captions and slides are excluded. Rule learned: attribution must identify both the borrowed source and the independently authored extension.
9. **The public-package validator initially rejected a valid anchor link.** After the bylines gained `index.html#sources`, the check treated that entire value as a filename and raised an assertion. The validator now parses the file path and fragment separately, then checks that the target section exists. Rule learned: validating links requires URL parsing, even for local relative links.

An in-app browser later declined a local-file navigation under its URL policy. No workaround was attempted through that browser. The existing project checks had already rendered the chapters in a separate test run, and the final release was checked by the package and static validators. This is a verification boundary, not evidence that the browser-visible local tab itself was retested after every wording change.

## 8. Verification, costs, and limits

The first-edition package check found four chapters, the expected source video IDs, timestamp links only in major section headings, working backward/forward/index navigation, and locally resolving links. Mathematical delimiter and syntax checks reported no errors in the generated HTML. The browser test observed the chapter pages and found no MathJax error nodes. Each chapter's worked solutions rendered as an ordered list with the expected number of answers. The nine-, ten-, and thirteen-chapter extensions repeat these checks across the expanded manifest and navigation chain.

These are structural and rendering checks. They do not prove that every scientific claim is true, that every external link will remain live, or that every equation is the only useful way to formalize an agent system. Product documentation can change after the date recorded here. The source captions were generated automatically and may contain transcription mistakes. The textbook report sampled selected pages rather than reading all twelve works cover to cover.

No paid API total, token bill, or reliable wall-clock total was recorded for the earlier drafting sessions, so none is invented here. The public release uses the repository host and does not depend on a temporary private preview. The authored material has no blanket license grant covering third-party recordings, captions, slides, or textbook PDFs; readers should follow each source's own terms.

## 9. Where the project stands

The [live reading guide](https://az9713.github.io/11-768-ai-agents-notes/) links the standalone chapters, the [style report](textbook-style-report.html), and this document. The [public repository](https://github.com/az9713/11-768-ai-agents-notes) contains the authored notes, HTML, renderer, validation code, and a source manifest. It excludes raw transcripts, downloaded slides, private preview configuration, and local book files. GitHub Pages reported a successful build, and the reading guide, this journey, a chapter, and the style report each returned HTTP 200 during the first-edition release check.

The first edition covered the first four recordings. On 25 September 2026, the same pipeline was extended through Lecture 9. The five added chapters treat planning and coordination, coding agents, computer-use agents, supervised fine-tuning, and reinforcement-learning foundations. A later parity audit found that their first drafts had the correct topics but developed them more briefly than Chapters 1–4. They were rebuilt from private source maps around persistent running cases, formal models, worked checks, integrated primary research, executable definitions, and full exercise discussions. The revised chapters preserve timestamped section navigation and the boundary between recorded material and independent extensions.

On 17 October 2026, Chapter 10 added the lecture on deep research agents. Unlike the earlier rounds, this round used subagents. They first gathered the captions, slides, and reading list into a private source map, and read the slide figures as images, because several numbers appear only in figures and some slide and speech numbers disagree. A citation check fetched every course reading and settled those discrepancies against the cited papers before drafting. One agent then drafted the chapter and its harness additions, a second reviewed it in a fresh context, and a third performed a science-editing pass; the fixes were applied before the chapter was approved. Every external link was fetched and its returned title compared with its label. The renderer now derives its counts from its chapter list, and the validator derives the chapter count and recording durations from the source manifest, instead of fixed constants. Agent token totals were not recorded.

Later on 17 October 2026, Chapters 11–13 added the lectures on advanced RL algorithms, RL training systems, and agent safety. The three chapters were drafted in parallel, each by its own agent, from the private source maps and citation checks prepared in the Chapter 10 round. Each draft then went through the same fresh-context review and science-editing pass, and its drafter applied or rejected each finding with a recorded reason. Integration added nine functions to the training module, a new `safety` module for Chapter 13, and tests that bring the suite to 58. Every external link in the three chapters returned HTTP 200. Five of those links returned no page title to the script, so their titles were compared with the sources by the agents instead. Two limits remain. The dense H100 bf16 rate in Chapter 12 comes from a third-party post, not from NVIDIA. Several papers in Chapters 11 and 13 were checked at the abstract level only. Chapter 13 kept about 190 words of lecture material that the editing pass had proposed to cut, so it is the longest chapter at 5,728 words. The lecturer of Chapter 13 is named from the video description and the speaker's own statements, because the title slide shows no name. After publication, a second model, OpenAI's Codex, reviewed the three chapters in a read-only sandbox. It found 21 defects that the first reviews had missed, mostly claims stated as always true that hold only under a condition, plus one code defect. Each finding was checked, all 21 were confirmed and fixed, and the reports are in the [review folder](https://github.com/az9713/11-768-ai-agents-notes/tree/main/review). Agent token totals were not recorded.
