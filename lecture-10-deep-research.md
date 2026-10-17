# Lecture 10 — Deep research agents: evaluation, training, and retrieval

*Independent study chapter · Based on a CMU 11-768: AI Agents lecture by Akari Asai, 24 September 2026 · [Course and attribution](index.html#sources)*

## 0. A research question has no single gold document [00:00](https://www.youtube.com/watch?v=nKUBrXFQBUM&t=0s)

A user asks an agent: "Can AI agents synthesize scientific literature as well as human experts?" The agent must find studies that compare agents with experts, check which fields they tested, and write a qualified answer with citations. The consequential error is a confident claim attached to a citation that does not support it. A reader who trusts the citation inherits the error.

Call this request the **parity question**; it is the running case of the chapter. A **deep research agent** answers such a question by repeated search, reading, and synthesis, and returns a long-form **report**. A **citation** points from a claim in the report to a source the reader can inspect. A claim is **supported** when the cited source states or entails it.

A lookup such as Herbert Simon's Nobel year, 1978, sits on one page. **Retrieval-augmented generation (RAG)** suffices there. A retriever runs once, and a **language model (LM)** answers from the retrieved text, as the [RAG survey](https://arxiv.org/abs/2312.10997) describes. The parity question's evidence is spread over papers that each answer a narrower question.

In the [OpenScholar study](https://www.nature.com/articles/s41586-025-10072-4), GPT-4o hallucinated citations 78–90% of the time. Experts preferred OpenScholar-8B and OpenScholar-GPT-4o answers over expert-written ones 51% and 70% of the time, against 32% for GPT-4o. Its benchmark covers only computer science, physics, neuroscience, and biomedicine. A report that cites it for "parity holds in every field" makes an unsupported claim.

### The research loop is an instance of the run model

Chapter 1 models a run with state $x_t=(s_t,h_t,m_t,b_t,u_t)$, proposal $z_t$, gate $G_\phi$, and a verifier. Let $x$ be the question, as in chapter 9; subscripted $x_t$ stays chapter 1's run state. The hidden state $s_t$ is the document collection's content, and $h_t$ holds reasoning, queries, and returned text. The budget $b_t$ counts tool calls, tokens, and money; $u_t$ grants read-only tools.

Three actions suffice. The action $\operatorname{search}(q)$ returns ranked snippets for query $q$, which is not chapter 1's workflow state $q_t$. The action $\operatorname{browse}(d)$ returns document $d$, and $\operatorname{answer}(y)$ ends the run with report $y$. The **stop rule** ends the run at the first answer, when $b_t$ reaches zero, or at step $T_{\max}$.

In one run, the first search returns a benchmark ranking AI systems and an expert evaluation. The agent searches again for tested tasks and domains. Its report reads: "Promising on tested tasks; broader parity with experts remains unproven".

**Reflection** is reasoning text in $h_t$, part of proposal $z_t$, that changes the next query. Let $\beta_t$ be chapter 1's belief over hidden states, here over which conclusion the literature supports. Evidence multiplies the odds by its likelihood ratio. A ranking of AI systems is about as likely whether parity holds or not, so its ratio is near one. An expert comparison is the discriminating evidence. A fixed RAG pipeline is a workflow in chapter 1's sense, $a_t=f(q_t,o_t)$. A deep research agent chooses each query from the history.

The verifier has no reference string to compare against. Chapter 1 noted that a factual answer may require sources that support each claim. Section 2 builds that check.

## 1. Short-answer benchmarks trade realism for checkability [03:58](https://www.youtube.com/watch?v=nKUBrXFQBUM&t=238s)

Natural Questions pairs real search queries with Wikipedia answers, such as "who wrote the score for the force awakens," answered by John Williams on one page. Four limits of such a benchmark define four design axes.

| Axis | Lookup end | Deep research end | Measurable test |
|---|---|---|---|
| **Search complexity** | one search | evidence joined across sources | does one document contain the answer? |
| **Domain expertise** | general | expert | do writing and checking need an expert? |
| **Answer form** | short entity | long-form synthesis | does a string match decide correctness? |
| **Evidence support** | not inspected | cited sources checked | does the score read the sources? |

**Exact match (EM)** scores a predicted answer $\hat y$ against a reference $y^*$: one when the strings agree after normalization, zero otherwise. The common convention lowercases and removes punctuation and articles; the sources here do not state one. EM is cheap for entities but blind to evidence. A model can recall "John Williams" from pretraining, which is **memorization**, and EM cannot tell recall from search.

[BrowseComp](https://arxiv.org/abs/2504.12516) has 1,266 questions built by **reverse construction**. A writer starts from a known answer, hides its name, and combines factual clues. A question is kept only if LMs with search fail and five Google searches do not suffice. One asks for a fictional character who breaks the fourth wall, was helped by selfless ascetics, is known for humor, and had a 1960s–1980s show with fewer than 50 episodes. The answer is Plastic Man.

Why is such a question hard to find but easy to check? Let $N_{\rm cand}$ be the number of candidate entities, and let clue $\ell$ hold for a fraction $\chi_\ell$ of them. If the $n_{\rm clue}$ clues are independent, one wrong candidate satisfies all of them with probability $\prod_\ell\chi_\ell$. Linearity of expectation over the $N_{\rm cand}-1$ wrong candidates gives

$$\mathbb E[\text{wrong survivors}]=(N_{\rm cand}-1)\prod_{\ell=1}^{n_{\rm clue}}\chi_\ell.$$

With $N_{\rm cand}=10^5$ and five clues at $\chi_\ell=0.1$, the count of wrong survivors is Binomial$(10^5-1,10^{-5})$, close to Poisson with mean one. The answer is then unique with probability $e^{-1}\approx0.37$; a sixth clue raises it to $e^{-0.1}\approx0.90$. Each clue is searchable alone, no index stores the intersection, and EM checks the short answer. Positively correlated clues leave more survivors, so uniqueness needs its own check.

| System (BrowseComp, reported 2025) | Accuracy (%) | 95% half-width (points) |
|---|---:|---:|
| GPT-4o | 0.6 | 0.4 |
| GPT-4o with browsing | 1.9 | 0.8 |
| OpenAI o1 (medium) | 9.9 | 1.6 |
| Deep Research (trained for BrowseComp-style tasks) | 51.5 | 2.8 |
| Humans, reference matches on 1,255 attempted | 25.3 | 2.4 |

The half-widths are $1.96\sqrt{\hat p(1-\hat p)/N_q}$ for $N_q$ independent questions. GPT-4.5, at 0.9%, is omitted. The 1.3-point GPT-4o gap exceeds its difference half-width, $\sqrt{0.4^2+0.8^2}\approx0.9$. Every other adjacent gap exceeds its own more than three times. Because all systems answer the same questions, a paired test is the correct check. The abstract states the benchmark's boundary. It sidesteps a true user query distribution, long answers, and ambiguity.

A live web adds cost, latency, and a changing index, so reruns differ. [BrowseComp-Plus](https://arxiv.org/abs/2508.06600) uses a **frozen corpus**. The o3 model gathers evidence pages, and humans verify spans and clues. The builders add **hard negatives**, related pages that miss a clue. The result is 830 questions over 100,195 documents, with a 95% half-width near 3.4 points at 50% accuracy. The frozen corpus isolates the retriever. With one prompt, GPT-5 scores 55.9% with BM25, a word-matching retriever, and 70.1% with the Qwen3-Embedding-8B retriever, using fewer searches.

On the second axis, [FinSearchComp](https://arxiv.org/abs/2509.13160) used 70 financial experts to write 635 questions. [MedBrowseComp](https://arxiv.org/abs/2505.14963) has over 1,000 human-curated medical questions.

Search budgets limit these measurements. A 500-token prompt that grows by 1,100 tokens per call holds $500+1{,}100\,n_{\rm call}$ tokens after $n_{\rm call}$ calls. Six calls fit an 8,192-token window, and the seventh reaches 8,200. Read from a figure in the [Tongyi DeepResearch report](https://arxiv.org/abs/2510.24701), Tongyi's BrowseComp accuracy is about 0% at an 8K context limit and about 43.5% at 128K. The token counts are illustrative.

## 2. Long-form answers need rubrics, and rubrics need audits [13:53](https://www.youtube.com/watch?v=nKUBrXFQBUM&t=833s)

A **reference-similarity metric** scores an answer by closeness to a reference. **Recall-Oriented Understudy for Gisting Evaluation (ROUGE)** counts shared word sequences. [BERTScore](https://arxiv.org/abs/1904.09675), after Bidirectional Encoder Representations from Transformers (BERT), matches contextual embeddings. [Bilingual Evaluation Understudy with Representations from Transformers (BLEURT)](https://arxiv.org/abs/2004.04696) is learned from ratings.

One word can flip truth, as in "parity is unproven" versus "parity is proven". [A study of long-form question-answering evaluation](https://aclanthology.org/2023.acl-long.181/) used 109 expert comparisons between answer pairs. ROUGE picked the expert-preferred answer 58% of the time, BERTScore 57%, and BLEURT 62%, against 50% for chance. The chance standard error is $\sqrt{0.25/109}\approx0.048$. These sit 1.7, 1.5, and 2.5 standard errors above chance. Only BLEURT clears the two-sided 5% level, and ROUGE passes only a one-sided test. BLEURT still misorders 38% of pairs.

A rubric replaces similarity with criteria. Chapter 7 defines weights $w_j\ge0$, checks $c_j\in\{0,1\}$, and $R_{\rm rubric}=\sum_jw_jc_j/\sum_jw_j$. A **judge**, usually an LM, decides each check on report $y$: $c_j=\operatorname{Judge}(\text{crit}_j,y)$, where $\text{crit}_j$ is criterion $j$'s text. A parity rubric gives weight 2 to "Reports a direct comparison with human experts". It gives weight 1 to "Qualifies the conclusion by task and domain". The candidate "Experts preferred OpenScholar answers in one study [1]. This holds in every field [1]" passes only the first.

```python
from agent_lab.evaluation import rubric_score

score = rubric_score(
    {"direct comparison": 2, "qualified by task and domain": 1},
    {"direct comparison": True, "qualified by task and domain": False},
)
assert abs(score - 2 / 3) < 1e-12
```

The check assumes binary judgments and fixed weights. The score measures only listed criteria. The false sentence costs a point because the rubric anticipated it. By chapter 7's distinction, an unsupported claim may belong among hard constraints.

Expert rubrics are costly, so [ResearchQA](https://arxiv.org/abs/2509.00496) mines surveys from 75 fields into 21K queries and 160K rubric items. A **parametric rubric** is one an LM writes from its own weights, without retrieved sources. An expert audit found 15% of parametric items vacuous, erroneous, or unclear. An item citing a nonexistent paper rewards a hallucinated citation. A **search-grounded rubric** is written from retrieved sources instead. A third source is expert writing: [DeepResearch Bench II](https://arxiv.org/abs/2601.08536) derives 9,430 binary items from expert articles, and its best agents satisfy fewer than 50%.

### Judge agreement needs a chance correction and an interval

In the OpenScholar evaluation, over 12 questions, 2 systems, and 2 expert raters per answer, an LM judge agreed with an expert on 79% of rubric judgments. Two experts agreed on 80%. **Observed agreement** $p_o$ is the fraction of identical labels, chance matches included. Let raters A and B mark an item satisfied with probabilities $\bar c_A$ and $\bar c_B$. Independent raters agree by chance with probability

$$p_e=\bar c_A\bar c_B+(1-\bar c_A)(1-\bar c_B).$$

**Cohen's kappa** maps chance agreement to zero and perfect agreement to one, provided $p_e<1$.

$$\kappa=\frac{p_o-p_e}{1-p_e}.$$

The function `agent_lab.evaluation.cohen_kappa` computes it from two label sequences. Base rates decide what 79% means. If both raters mark 70% of items satisfied, $p_e=0.58$ and $\kappa=0.50$. Disagreement needs at least one rater to say "unsatisfied," so $p_o\ge1-(1-\bar c_A)-(1-\bar c_B)$. At 88% for both, $p_o\ge0.76$ permits 79%, and $p_e\approx0.789$ gives $\kappa\approx0.01$, chance level. No base rates are reported, so $\kappa$ is unknown.

If the 24 question–system answers are the independent units, 79% carries a 95% half-width of $1.96\sqrt{0.79\cdot0.21/24}\approx0.16$. Items within an answer are correlated. The one-point gap shows the judge is not grossly worse than a second expert here. It does not establish equivalence. The [study of LMs as judges](https://arxiv.org/abs/2306.05685) documents position, verbosity, and self-enhancement biases.

### Citation support is a separate measurement

A rubric judges the report. A citation check judges each link. [DeepResearch Bench](https://arxiv.org/abs/2506.11763) introduces **Factual Abundance and Citation Trustworthiness (FACT)**. FACT extracts **claim–citation pairs**, fetches each cited page, and asks a judge whether the page supports the claim. Let the report contain claims $k=1,\ldots,N$, let claim $k$ cite $n_k\ge0$ sources, and let $v_{k,r}\in\{0,1\}$ record support by source $r$. With at least one pair, **citation accuracy** and **effective citations** are

$$\operatorname{CA}=\frac{\sum_k\sum_rv_{k,r}}{\sum_kn_k},\qquad\operatorname{EC}=\sum_k\sum_rv_{k,r}.$$

In a two-claim example, "OpenScholar was compared with human answers [1]" is supported and "OpenScholar matches experts in every field [1]" is not: $\operatorname{CA}=1/2$, $\operatorname{EC}=1$. Neither asks whether each claim has support. Define **citation recall**, which DeepResearch Bench does not report:

$$\operatorname{CR}=\frac1N\sum_{k=1}^N\mathbf 1\Big[\max_rv_{k,r}=1\Big],$$

where $\mathbf 1[\cdot]$ is one when its condition holds and a maximum over no sources is zero. Take five claims: two with one supported citation each, the "every field" claim with an unsupported one, one with two supported citations, and one uncited. Then $\operatorname{CA}=4/5$, $\operatorname{EC}=4$, and $\operatorname{CR}=3/5$.

```python
from agent_lab.evaluation import citation_metrics

claims = [[True], [True], [False], [True, True], []]
assert citation_metrics(claims) == (0.8, 4, 0.6)
without_bad_link = [[True], [True], [], [True, True], []]
assert citation_metrics(without_bad_link) == (1.0, 4, 0.6)
```

The function assumes correct extraction and judgments. The second assertion exposes an incentive: deleting the bad citation raises accuracy to 1.0 while the false claim stays. Only recall reveals it. Neither measures source quality, and the judge errs. Take a true-positive rate of 0.95, a false-positive rate of 0.10, and 40% of pairs unsupported. Measured accuracy is $0.95\cdot0.60+0.10\cdot0.40=0.61$ against a true 0.60. Yet $0.04/0.61\approx6.6\%$ of accepted pairs are unsupported. These rates are illustrative.

## 3. Training data comes from checkable questions and filtered trajectories [24:09](https://www.youtube.com/watch?v=nKUBrXFQBUM&t=1449s)

A rollout alternates model text, such as the reasoning "Find expert comparisons" and the call `search("expert comparisons")`, with tool text the harness appends. Let trajectory $\tau_i$ be a token sequence $(\tau_{i,1},\ldots,\tau_{i,L_i})$. Let $m_{i,j}=1$ when the policy generated token $j$, giving **model tokens**, and $m_{i,j}=0$ for question and **tool tokens**. Chapter 8's masked loss and Section 4's losses train only model tokens. The [Search-R1](https://arxiv.org/abs/2503.09516) abstract names retrieved-token masking as a condition for stable training. Training tool tokens would teach a model of the search engine, not a policy. A step with 40 reasoning tokens, a 15-token call, and a 1,200-token result trains only $55/1{,}255\approx4.4\%$ of its tokens. The counts are illustrative.

The Tongyi report uses three stages. **Mid-training** continues pretraining on agent trajectories. **Supervised fine-tuning (SFT)** imitates teacher trajectories. **Reinforcement learning (RL)** learns from the policy's own rollouts. Each stage needs questions with checkable answers.

A **knowledge graph** stores facts as triples (subject, relation, object), such as (Allen Newell, faculty member of, Carnegie Mellon). A **random walk** from Newell collects connected facts: the 1975 Turing Award shared with Herbert Simon, Simon's 1978 Nobel Prize, and their 1972 book *Human Problem Solving*. Hiding the target and rewriting relations as clues yields a question. It asks: "Which CMU researcher coauthored a book on human problem solving with a future economics Nobel laureate, and shared the Turing Award with that colleague three years before the Nobel Prize?" The answer is Allen Newell. [WebSailor-V2](https://arxiv.org/abs/2509.13305) builds high-uncertainty tasks by such structured sampling and information obfuscation.

The procedure mirrors reverse construction and its failure. The Newell question looks four-hop but falls in one or two searches: one finds Simon from the book and prize, and one finds both 1975 recipients. BrowseComp's five-search rule would reject it, though it remains a usable easy training item. Synthesis needs a difficulty filter and a uniqueness check over the graph.

### Filtering selects what imitation learns

In filtered teacher SFT, a stronger **teacher** model generates trajectories, and a filter keeps some. The student trains on them with chapter 8's masked loss, as in [WebDancer](https://arxiv.org/abs/2505.22648). The retained set is $\mathcal D^+=\{\tau:\operatorname{EM}(\operatorname{ans}(\tau),y^*)=1,\ \operatorname{valid}(\tau)=1,\ \operatorname{coherent}(\tau)=1\}$, where $\operatorname{ans}(\tau)$ is the final answer and coherence is a judgment. Here EM is exact match, not the expectation-maximization of chapter 9's ReST-EM. This is chapter 9's filtered objective with the teacher as sampler.

The filter shifts the training distribution. Suppose half the questions are hard, and the teacher solves 60% of easy and 10% of hard ones, one trajectory each. Hard questions then form $0.5\cdot0.10/(0.5\cdot0.60+0.5\cdot0.10)=0.05/0.35\approx14\%$ of the retained set, not 50%. A lucky correct string reached by an incoherent path passes EM, so only the coherence judgment catches it. These numbers are illustrative.

Mid-training instead continues next-token prediction on agent trajectories mixed with general text. [Scaling Agents via Continual Pre-training](https://arxiv.org/abs/2509.13310) applies one SFT set to a Qwen3 base and to its mid-trained AgentFounder base. Pass@1, chapter 6's pass@$k$ at $k=1$, rose from 28.6% to 39.9% on BrowseComp-en and from 35.6% to 43.3% on BrowseComp-zh. On GAIA, a benchmark for General AI Assistants, it rose from 71.8% to 72.8%. The abstract confirms 39.9% and 43.3% for AgentFounder. Only the GAIA pair is assigned from the figure legend. That is one comparison. No interval is printed, so the 1.0-point GAIA change is unresolved.

## 4. Exact-match reward turns short answers into an RL signal [31:15](https://www.youtube.com/watch?v=nKUBrXFQBUM&t=1875s)

RL trains on the policy's own attempts, so it sees the histories its own errors create. **Reinforcement learning with verifiable rewards (RLVR)** computes reward by a program that checks the outcome. For a short answer,

$$R(\tau_i)=\operatorname{EM}\big(\operatorname{ans}(\tau_i),y^*\big).$$

If three Newell rollouts answer Herbert Simon, Allen Newell, and Allen Newell, their rewards are 0, 1, and 1, as an illustration. Search-R1 reports relative gains, as the abstract words them, of 41% for Qwen2.5-7B and 20% for Qwen2.5-3B over retrieval-augmented baselines on seven datasets. EM gives zero to an alias such as "A. Newell" and one to a lucky guess. It never inspects the evidence path.

### The clipped objective weights model tokens by a probability ratio

Let $n$ be the number of rollouts per question. Chapter 9 writes $G$, reserved here for the gate $G_\phi$. **Group Relative Policy Optimization (GRPO)**, from [DeepSeekMath](https://arxiv.org/abs/2402.03300), compares each rollout with the other $n-1$. Chapter 9 defines the centered advantage $\hat A_i=R_i-\bar R$. DeepSeekMath also divides by the group standard deviation. For $n=3$ rewards $(0,1,1)$, the centered advantages are $(-2/3,1/3,1/3)$. With the population deviation they standardize to about $(-1.41,0.71,0.71)$; chapter 9's sample deviation gives $(-1.15,0.58,0.58)$.

Let $\pi_{\theta_{\rm old}}$ be the sampling policy. For a model token, define the policy ratio

$$r_{i,j}(\theta)=\frac{\pi_\theta(\tau_{i,j}\mid x,\tau_{i,<j})}{\pi_{\theta_{\rm old}}(\tau_{i,j}\mid x,\tau_{i,<j})},$$

where $\tau_{i,<j}$ is every earlier token, tool tokens included. This is the token-level form of chapter 9's ratio $r_t(\theta)$, with $\pi_{\theta_{\rm old}}$ in place of $\mu$. Let $M_i=\sum_jm_{i,j}$, let $\epsilon>0$ be chapter 9's clip width, and let $\beta_{\rm KL}\ge0$ weight a **Kullback-Leibler (KL)** penalty. The coefficient $\beta_{\rm KL}$ is unrelated to the belief $\beta_t$. The surrogate is

$$L^{\rm GRPO}(\theta)=\mathbb E\Big[\frac1n\sum_{i=1}^n\frac1{M_i}\sum_jm_{i,j}\Big(\min\big(r_{i,j}\hat A_i,\operatorname{clip}(r_{i,j},1-\epsilon,1+\epsilon)\hat A_i\big)-\beta_{\rm KL}D_{{\rm KL},i,j}\Big)\Big].$$

The expectation runs over questions and sampled groups. The term $D_{{\rm KL},i,j}$ is chapter 9's divergence from reference $\pi_{\rm ref}$ at token $j$'s history. The factor $1/M_i$ averages within a trajectory; batch-level token averaging is a different objective. Tool tokens leave the sum but stay in the conditioning history. For positive $\hat A_i$, clipping stops the gain once $r_{i,j}>1+\epsilon$; for negative $\hat A_i$, once $r_{i,j}<1-\epsilon$. Only the mask is specific to search.

Search makes degenerate groups common. Chapter 9 gives $p^n+(1-p)^n$ as the probability that all rewards tie. At $p=0.1$ and $n=8$ it is about $0.43$. About 43% of such groups give zero advantages, and only the KL term updates them.

Rollout length also varies. Suppose eight rollouts finish between 10 and 120 seconds, mean 55. A synchronous trainer is then busy about $55/120\approx46\%$ of each step, as an illustration. An **asynchronous rollout** scheme scores each trajectory as it finishes and batches completed ones. The Tongyi report also caches tool results. The cost is **staleness**: a rollout begun under $\pi_{\theta_{\rm old}}$ can finish after an update. It stays valid only if the trainer logs its sampling probabilities for $r_{i,j}$, as chapter 9 requires for off-policy reuse.

## 5. Rubric rewards extend RL to open-ended reports [35:46](https://www.youtube.com/watch?v=nKUBrXFQBUM&t=2146s)

EM cannot score the parity question, whose report needs coverage, relevance, factuality, and supported claims. A **rubric reward** scores rollout $\tau_i$ by a rubric applied to its report $y_i$. [Deep Research Tulu (DR Tulu)](https://arxiv.org/abs/2511.19399) trains an 8-billion-parameter agent this way.

Rubric rewards appear with three normalizations: by the total weight, by the positive weight, or none. The DR Tulu paper's Eq. (1), in version 3, divides by the positive-weight sum. Its judge outputs 0, 0.5, or 1. Let $\Xi_t(x)$ index the items active for question $x$ at step $t$. The letter $\Xi$ is unused elsewhere in the series. With graded checks $c_j(y)\in\{0,\tfrac12,1\}$,

$$R_t(\tau)=\frac{\sum_{j\in\Xi_t(x)}w_jc_j(y)}{W_t(x)},\qquad W_t(x)=\sum_{j\in\Xi_t(x):\,w_j>0}w_j.$$

With all weights positive, this is $R_{\rm rubric}$ with graded checks. The equation permits negative weights. How DR Tulu weights negative rubrics is not given in the sources cited.

Does normalization matter for GRPO? Suppose all $n$ rollouts in a group are judged against one rubric set. The paper scores all responses with the current active rubrics. Then $W_t(x)$ is a group constant. With $S_i=W_t(x)R_i$, the group mean is $\bar S=W_t(x)\bar R$, so

$$S_i-\bar S=W_t(x)\,(R_i-\bar R).$$

Centered advantages scale by $W_t(x)$, and so does the group standard deviation. Standardized advantages are therefore identical under all three forms whenever the deviation is positive. Without standardization, questions with more rubric weight get larger updates. As a check, let three parity reports pass item 1 only, both items, and neither. Normalized rewards $(2/3,1,0)$ give centered advantages $(1/9,4/9,-5/9)$; unnormalized rewards $(2,3,0)$ give $(1/3,4/3,-5/3)$. Both standardize, with the population deviation, to about $(0.27,1.07,-1.34)$.

### Evolving rubrics make the reward non-stationary

DR Tulu's **reinforcement learning with evolving rubrics (RLER)** changes $\Xi_t(x)$ during training. Initial rubrics are written with search and persist. At each step, an LM contrasts the sampled reports and writes new rubrics. A **positive rubric** rewards a correct, specific claim that some rollout made. A **negative rubric** penalizes a specific error that some rollout made. A **rubric buffer** keeps the persistent and useful collected rubrics. On the parity question, a rollout claiming "OpenScholar matches experts in every field" can trigger the negative rubric "The response claims parity across all fields". While that rubric stays in the buffer, every later rollout that repeats the claim loses reward, which a rubric fixed in advance would have to anticipate.

The cost is a **non-stationary reward**, one whose definition changes during optimization. The step-$t$ objective is $J_t(\theta)=\mathbb E[R_t(\tau)]$. Its $\Xi_t(x)$ depends on earlier rollouts, hence on earlier parameters. Chapter 9's policy-gradient identity assumes a reward independent of $\theta$ except through the trajectory. The rubric set is fixed while a group is scored, so each estimate is valid for $J_t$, but the target moves. A rising training-reward curve is therefore not comparable across steps. The generator and judge are LMs, so their errors become reward. In Section 2, 15% of parametric items were flawed. No flaw rate is reported for evolving rubrics.

DR Tulu trains with 8 rollouts per prompt and 32 prompts per batch. Its ablations, read from the figures, average HealthBench, ScholarQABench v2, and DeepResearch Bench.

| Reward, from the same SFT start | Score at 2,500 steps (%) |
|---|---:|
| No RL | about 50.0 |
| Random rewards | about 50.9 |
| Initial rubrics only | about 59.5 |
| Evolving rubrics | about 61.2 |

Random rewards stay within about one point of no RL, so the reward's content matters. Evolving rubrics lead initial ones by up to 2.7 points, at 2,000 steps. A second figure tests SFT as a **warm start**: at 600 RL steps, no SFT scores about 38.5% and full SFT about 54.2%; the undertrained start lies between.

Read from the cost figure, DR Tulu-8B costs about 0.0018 US dollars per query against about 1.8 for OpenAI Deep Research. Its score sits about 0.5 above OpenAI Deep Research and 0.5 below GPT-5 with search, inside reading error. The abstract reports a 0.7% average gain. A system lies on the **Pareto frontier** when no other is both cheaper and higher scoring. DR Tulu, the cheapest plotted system, is on it by cost alone. This figure averages four benchmarks, the ablation three.

## 6. Retrieval quality bounds what the agent can find [45:00](https://www.youtube.com/watch?v=nKUBrXFQBUM&t=2700s)

An agent can cite only what its search returns, from a local collection or a web search application programming interface (API). Section 1 measured the stakes. With the same prompt, GPT-5 rose from 55.9% to 70.1% when BM25 was replaced by Qwen3-Embedding-8B.

**BM25**, short for Okapi Best Matching 25, is a lexical scoring function. For each query term, it multiplies a saturating, length-normalized term-frequency factor by an inverse-document-frequency weight that favors rare terms, and sums. It needs no training. The [BEIR benchmark](https://arxiv.org/abs/2104.08663) found it a robust zero-shot baseline. A **dual encoder** maps queries and documents to vectors through encoders $\operatorname{enc}_q$ and $\operatorname{enc}_d$ with parameters $\psi$, and scores

$$\operatorname{sim}(q,d)=\operatorname{enc}_q(q;\psi)^\top\operatorname{enc}_d(d;\psi).$$

Documents are encoded offline into a **vector index**. A query returns its nearest vectors, chapter 4's top-$k$ rule. [Dense Passage Retrieval (DPR)](https://aclanthology.org/2020.emnlp-main.550/) reports 9–19 points higher top-20 accuracy than Lucene BM25.

A three-document example uses the Newell facts. The query is "Who shared the Turing Award with Simon?" Document [1], "Newell and Simon: 1975 Turing Award," is relevant. Document [2], "Simon: economics Nobel Prize, 1978," shares "Simon," so it is a hard negative. Document [3], "CMU campus buildings and directions," is an easy negative.

### The contrastive loss is a softmax cross-entropy

Let $d^+$ be the relevant document and $\mathcal N(q)$ a set of negatives. Over the candidates $\{d^+\}\cup\mathcal N(q)$, define

$$p_\psi(d\mid q)=\frac{\exp\operatorname{sim}(q,d)}{\exp\operatorname{sim}(q,d^+)+\sum_{d^-\in\mathcal N(q)}\exp\operatorname{sim}(q,d^-)}.$$

The **contrastive loss** is the cross-entropy against a target with all mass on $d^+$.

$$\mathcal L_{\rm contrast}(\psi)=-\log p_\psi(d^+\mid q)=-\operatorname{sim}(q,d^+)+\log\sum_d\exp\operatorname{sim}(q,d).$$

This form is widely called InfoNCE after the [contrastive predictive coding paper](https://arxiv.org/abs/1807.03748), which uses a probabilistic contrastive loss with negative sampling. Differentiate by one candidate's score. The first term gives $-1$ for $d^+$ only, and the log-sum-exp gives the softmax probability. Hence

$$\frac{\partial\mathcal L_{\rm contrast}}{\partial\operatorname{sim}(q,d)}=p_\psi(d\mid q)-\mathbf 1[d=d^+],\qquad\nabla_\psi\mathcal L_{\rm contrast}=\sum_d\big(p_\psi(d\mid q)-\mathbf 1[d=d^+]\big)\nabla_\psi\operatorname{sim}(q,d).$$

The positive is pulled up with weight $1-p_\psi(d^+\mid q)$, and each negative is pushed down in proportion to its current probability. A negative far below the positive receives almost no gradient, so hard negatives carry the signal.

Let the Newell scores be $(2,1,0)$. The exponentials $7.389$, $2.718$, and $1$ sum to $11.107$. The probabilities are $(0.665,0.245,0.090)$, and the loss is $-\log0.665\approx0.408$. The score gradient $(-0.335,0.245,0.090)$ pushes the hard negative about 2.7 times harder than the easy one.

```python
from agent_lab.training import contrastive_loss

loss = contrastive_loss([2.0, 1.0, 0.0], positive_index=0)
assert abs(loss - 0.4076) < 1e-4
assert abs(contrastive_loss([12.0, 11.0, 10.0], 0) - loss) < 1e-12
```

The function assumes one positive and raw inner-product scores without a temperature. The second assertion checks shift invariance, which holds because the derivative along the all-ones direction is zero (Exercise 2). **In-batch negatives** reuse the other $B-1$ queries' positives in a minibatch at no extra encoding cost. DPR adds BM25 hard negatives. Either source can yield **false negatives**, relevant documents labeled negative, whose gradient pushes a correct document down.

### Reasoning-aware retrieval conditions on the agent's information need

The retriever usually sees only the latest query $q_t$. On the parity question, "expert comparisons" does not say what is compared with what. The preceding reasoning, "Find expert comparisons; check the tested tasks," does. Let $\zeta_t$ be that reasoning, the part of proposal $z_t$ written just before $q_t$.

[AgentIR](https://arxiv.org/abs/2603.04384) jointly embeds $\zeta_t$ and $q_t$, which it calls reasoning-aware retrieval. Its DR-Synth method generates labels for intermediate searches from standard question-answering datasets. DR-Synth keeps the top 45 of the 50 results for an intermediate query, inserts 5 global gold documents, and reranks with the global question and answer in view. Top ranks become positives and bottom ranks negatives; the cut-off rank is not given.

On BrowseComp-Plus with the Tongyi-DeepResearch agent, the abstract reports 68% accuracy for AgentIR-4B, 50% for embedders twice its size, and 37% for BM25. Read from the figure, Tongyi's mean search calls fall from about 32.9 with BM25 to 25.9 with AgentIR. An ablation separates reasoning input from DR-Synth training. All values below are printed, not read by eye.

| Retriever (accuracy, %) | Tongyi-DR | gpt-oss-120B | GLM-4.7 | Tongyi-DR (visit) |
|---|---:|---:|---:|---:|
| Query only, untrained | 48.7 | 47.6 | 50.5 | 50.2 |
| Query plus reasoning, untrained | 55.5 | 51.3 | 50.9 | 54.0 |
| Query only, DR-Synth trained | 59.4 | 59.2 | 57.5 | 59.5 |
| Both (AgentIR) | 66.3 | 67.0 | 64.7 | 68.1 |

No benchmark is named for this ablation, and the "visit" column is not defined. Its 68.1 matches the abstract's 68% on BrowseComp-Plus. The Tongyi-DR 66.3 also matches the 66.4 read by eye from the BrowseComp-Plus scatter. The trained query-only row reappears exactly as the history ablation's baseline.

For gpt-oss-120B, reasoning alone adds 3.7 points and training alone 11.6, a sum of 15.3. Both add 19.4, an interaction of 4.1. The interaction is 0.1 for Tongyi-DR, 6.8 for GLM-4.7, and 4.8 for the visit column. On GLM-4.7, the encoder needed training before the reasoning helped: 0.4 points untrained, 7.2 trained. On Tongyi-DR, reasoning adds about 7 points with or without training. If the four accuracies are independent estimates on 830 questions, an interaction carries a 95% half-width near 6.8 points. No interval is printed, and paired questions would narrow it.

A second ablation fixes DR-Synth training and varies the input. On Tongyi-DR, query only scores 59.4, prior queries 63.1, with reasoning 63.1, and with documents added 60.0. Current reasoning alone scores 66.3. Adding documents falls below prior queries in all four columns. The source lecture attributes the drop to long inputs that exceed what the embedder represents well. The table does not test that explanation.

## 7. Evaluation limits every reported number [66:17](https://www.youtube.com/watch?v=nKUBrXFQBUM&t=3977s)

Short-answer evaluation is measurable. BrowseComp accuracy rose from 1.9% for GPT-4o with browsing to 51.5% for a system trained on such tasks. The DR Tulu abstract notes that most open agents train with short-form verifiable rewards. Graph-synthesized questions imitate the construction of the benchmark that scores them, so a gain may measure fit to that procedure rather than research ability. No benchmark here measures whether a report improves a later decision.

**Contamination** is the presence of evaluation answers in material the system can access. The official BrowseComp file stores questions and answers encrypted, which keeps that file out of plain-text crawls but not copies decrypted elsewhere. Memorization of such copies is one route. Running the agent with search disabled probes it. Retrieval is the other: a live-search agent can find a decrypted copy. A **blocklist** of dataset hosts removes known copies, and the harness should log and flag listed visits. A frozen corpus without benchmark files closes this route.

### Scope and limits of deep research evaluation and training

Three results are formal. The uniqueness argument assumes independent clues. The normalization result needs one rubric set per group. The contrastive gradient is exact for a finite softmax. Everything else is empirical, mostly from one paper, one comparison, or a figure read by eye. A result should report the corpus or search API and its date, the retriever, the judge, citation accuracy with recall, cost, and contamination controls.

### Exercises

1. A parity report has four claims. Claim 1 cites two sources, one supporting. Claim 2 cites one supporting source. Claim 3 cites one non-supporting source. Claim 4 cites nothing. Compute CA, EC, and CR. Then compute the rubric score for weights $(3,2,1)$ and checks $(1,0,1)$.
2. Prove that the score gradient of $\mathcal L_{\rm contrast}$ sums to zero and that adding one constant to every score leaves the loss unchanged.
3. Two raters have observed agreement $p_o=0.85$; rater A marks 80% of items satisfied and rater B 75%. Compute $p_e$ and $\kappa$, and explain why $p_o$ alone cannot show that an LM judge can replace an expert.
4. Design a held-out evaluation for an agent trained with evolving rubrics on long-form questions whose evidence comes from the BrowseComp-Plus corpus. Name three controls, state what each rules out, and say what to report beside the rubric score.

### Solutions and discussion

1. The pairs are $(1,0)$ for claim 1, $(1)$ for claim 2, and $(0)$ for claim 3: four pairs, two supported. So $\operatorname{CA}=2/4=0.5$ and $\operatorname{EC}=2$. Claims 1 and 2 have support and claims 3 and 4 do not, so $\operatorname{CR}=2/4=0.5$. The rubric score is $(3+0+1)/6=4/6\approx0.667$.
2. Component $d$ of the score gradient is $p_\psi(d\mid q)-\mathbf 1[d=d^+]$. Summing over candidates gives $\sum_dp_\psi(d\mid q)-1=0$. Replacing every score by $\operatorname{sim}+a$ changes the first term to $-\operatorname{sim}(q,d^+)-a$ and the second to $\log\big(e^a\sum_d\exp\operatorname{sim}(q,d)\big)=a+\log\sum_d\exp\operatorname{sim}(q,d)$. The changes cancel.
3. $p_e=0.80\cdot0.75+0.20\cdot0.25=0.65$, so $\kappa=(0.85-0.65)/0.35\approx0.57$. Of the 85% raw agreement, 65 points are expected from base rates alone. Replacement also needs the expert–expert $\kappa$, intervals from the number of independent questions, and bias checks such as verbosity.
4. Hold out questions whose evidence documents and source BrowseComp items were absent from training; this rules out answer reuse. Confirm the corpus holds no benchmark files and log retrieved documents; this rules out reading reference answers. Score with a fixed rubric set and a judge not used in training; this rules out credit for adapting to the training judge. Report citation accuracy and recall, search calls, and cost per question beside the rubric score, with 95% intervals.

The first answer shows why the citation quantities differ: claim 1 counts for recall although half its citations fail, and claim 4 lowers recall but not accuracy. A high rubric score does not imply supported citations.

The second answer shows that only score differences matter: the pull on the positive equals the total push on the negatives.

The third answer shows that base rates control agreement. When most items are satisfied, unrelated raters still agree often.

The fourth answer combines both contamination routes with non-stationarity: a drifting training reward cannot measure progress, but a fixed, independent evaluation can.

### Further reading

- [BrowseComp](https://arxiv.org/abs/2504.12516): reverse-constructed questions that are hard to find and easy to verify.
- [BrowseComp-Plus](https://arxiv.org/abs/2508.06600): a frozen corpus with verified evidence and hard negatives, which isolates the retriever.
- [DR Tulu](https://arxiv.org/abs/2511.19399): reinforcement learning with evolving rubrics for long-form research.
- [AgentIR](https://arxiv.org/abs/2603.04384): reasoning-aware retrieval and DR-Synth data for intermediate searches.
