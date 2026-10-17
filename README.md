# CMU 11-768 AI Agents — Independent Study Edition

**Agent Systems Foundations** is a set of independent, research-extended study chapters based on **CMU 11-768: AI Agents**.

**[Open the live HTML reading guide](https://az9713.github.io/11-768-ai-agents-notes/)** · [Development journey](https://az9713.github.io/11-768-ai-agents-notes/development-journey.html) · [Twelve-reference editorial study](https://az9713.github.io/11-768-ai-agents-notes/textbook-style-report.html)

The first thirteen available lectures of [CMU 11-768: AI Agents](https://www.cmu-agents.com/) supplied more than a syllabus. Their recordings, captions, and slides informed the chapters' topic order, technical concepts, distinctions, and some examples. The chapters re-explain that material and add independently developed mathematical models, research, code, cases, and exercises. Ninety-six of 98 main sections link to relevant recording segments. This count measures navigation, not the fraction of chapter content taken from the lectures. The [CMU 11-768 reading guide](index.html#sources) gives a chapter-by-chapter source map. These are **independent study notes, not official CMU 11-768 course notes**. They are not endorsed by Carnegie Mellon University or the instructors. Consult the original recordings for what was actually taught.

The thirteen CMU 11-768 study chapters cover run states and authorization, tool contracts, long-context computation, memory, planning, coding agents, computer use, supervised fine-tuning, reinforcement learning, deep research agents, advanced RL algorithms, RL training systems, and agent safety. They are aimed at advanced readers who have used agents and want to understand how agent systems are developed. Each chapter provides local definitions, derivations, worked cases, limitations, exercises, and solutions. A cumulative [Python harness](agent_lab/README.md) implements central equations and boundaries. The [development journey](DEVELOPMENT-JOURNEY.md) records the requests and corrections that shaped the first edition; the [style report](textbook-style-report.html) explains what was learned from twelve machine-learning references and two technical-writing guides.

## What this project adds beyond the transcripts

The lecture transcripts provide the topical spine and preserve the sequence of ideas. This project develops that source material into graduate-level teaching chapters through the following contributions:

- **Mathematical and logical formalization.** Informal descriptions of agent behavior are expressed through state-transition models, typed tool contracts, authorization predicates, optimization objectives, probability models, and explicit invariants. Every symbol is defined before use.
- **Research extensions.** Primary papers, official documentation, and production agent systems supply omitted background, deeper derivations, implementation tradeoffs, failure modes, and current engineering practices.
- **Worked examples and exercises.** Each chapter includes concrete cases, calculations, exercises, and solutions that test the formal concepts instead of merely restating lecture claims.
- **Executable agent harness.** The standard-library [`agent_lab`](agent_lab/README.md) package turns central equations and safety boundaries into cumulative Python code. By Chapter 13 it covers authorization, tool execution, context selection, memory, planning, evaluation, supervised learning, reinforcement-learning diagnostics, research-agent evaluation and retrieval measures, policy-optimization and RL-systems arithmetic, and safety measures.
- **Textbook-style exposition.** The prose was revised using patterns extracted from twelve machine-learning textbooks and two technical-writing guides. Section bodies read as independent explanations; timestamps remain navigation links to the relevant recording segments.
- **Explicit provenance.** The [source map](index.html#sources) distinguishes traceable lecture material from independently developed extensions. Raw captions and course slides are not redistributed.

These enhancements are independently authored. They should be evaluated as interpretations and extensions of the course material rather than as statements by Carnegie Mellon University or the instructors.

The repository contains authored Markdown, standalone HTML, the standard-library companion harness and tests, a renderer, a validation script, and a small [source manifest](sources/manifest.json). It does **not** contain raw captions, transcript dumps, downloaded course slides, textbook PDFs, or private preview settings. Source recordings and research references are linked from the pages.

To regenerate and check the HTML with Python:

```sh
python -m pip install -r requirements.txt
python render_notes.py
python validate_notes.py
python -m agent_lab.order_demo
python -m unittest discover -s tests -v
```

The optional browser check uses a local Google Chrome installation through Playwright:

```sh
python browser_check.py
```

The thirteen videos, course slides, and linked books remain the works of their respective rights holders. Their publication online does not grant this repository permission to redistribute them.

## License

The independently authored material in this repository (the chapter text, the `agent_lab` code and tests, and the render and validation scripts) is released under the [MIT License](LICENSE). The license covers only that material. It does not cover the CMU 11-768 recordings, slides, or course name, which belong to their rights holders. It does not cover the linked papers and books. The bundled MathJax file in `assets/` keeps its own license (`assets/mathjax-LICENSE`).
