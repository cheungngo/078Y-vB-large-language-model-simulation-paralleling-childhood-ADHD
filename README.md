# Functional Versus Structural Interventions in LLM Cognitive Control: Parallels to Childhood-Diagnosed ADHD in a Budget-Reversal Simulation

**Authors:**

Ngo Cheung, MBBS(HK), FHKAM(Psychiatry)

Hong Kong SAR, China

**Affiliations:**

¹ Independent Researcher

**Corresponding Author:**

Ngo Cheung, MBBS, FHKAM(Psychiatry)

Hong Kong SAR, China

Tel: 98768323

Email: info@cheungngomedical.com

## Abstract

Large language models frequently exhibit deficits in sustained attention, impulse control, planning, and cognitive flexibility that parallel core features of attention-deficit/hyperactivity disorder (ADHD). This study adapted a recent reinforcement-learning simulation of childhood-diagnosed ADHD (Cheung, 2026) to examine whether similar functional-versus-structural treatment distinctions emerge in transformer-based agents. Using Qwen2-0.5B-Instruct, we implemented a two-phase budget-reversal travel-planning task designed to elicit four symptom domains under long context pressure: inattention to embedded facts, hyperactive drift, executive dysfunction in multi-step planning, and rigidity when adapting to a sudden budget reduction from 1500 to 800.

Four treatment arms were tested across 10 random seeds: untreated (high temperature, no scaffolding), chronic stimulants (low temperature plus structured output scaffold from the first turn), late stimulants (scaffold applied only after the reversal), and stimulant plus ketamine-like intervention (low temperature, scaffold, external rolling memory buffer, and self-correction passes). Performance was evaluated using a composite health score derived from four rule-based symptom metrics plus factual recall accuracy.

Results showed modest numerical improvement with chronic functional modulation (mean health score 0.621) and no benefit---or slight worsening---with late modulation (0.572). The largest gains in overall health (0.706), complete elimination of drift, and significantly better reversal adaptation occurred in the memory-augmented arm. These findings suggest that while temperature reduction and scaffolding can sharpen output in a limited way, external memory mechanisms may be required to address accumulated context pollution and support genuine adaptation in long-horizon tasks. The pattern broadly replicates the dissociation between dopaminergic-style noise reduction and synaptic restoration observed in the original computational model. Implications for LLM agent design and potential parallels in computational psychiatry are discussed.

**Keywords:** large language models, cognitive control, memory augmentation, prompt engineering, computational simulation, ADHD analogue

## Introduction

Large language models can now manage many demanding language tasks, but their performance still breaks down in ways that are not entirely random. Some of these failures resemble difficulties seen in attention-deficit/hyperactivity disorder, particularly in attention, control, planning, and flexibility. When prompts become long, models often fail to use information placed in the middle of the context, a pattern described as \"lost in the middle\" \[1\]. Output can also become unstable at higher sampling temperatures, with responses drifting off topic or becoming repetitive, which can look like a form of impulsive or poorly regulated generation \[2\]. Problems also appear in tasks that require several linked steps. Even strong models may struggle to hold a plan together over time, especially when the task changes unexpectedly \[3\]. A related issue appears when the model has to revise an earlier decision after a new constraint is introduced, such as a sudden budget cut. In such cases, models may persist with the original line of reasoning instead of adjusting to the new requirement, which is similar to poor reversal learning in humans \[4\]. Taken together, these patterns suggest recurring weaknesses in sustained attention, impulse control, executive organisation, and cognitive flexibility, which are also central domains of impairment in ADHD \[5\].

Recent work has started to draw broad links between these model failures and neurodevelopmental conditions. Some studies have examined whether LLMs can reproduce ADHD-like response profiles in psychometric settings or help generate educational material for people with ADHD \[6,7\]. Other studies have treated chatbots as supports for executive functioning in ADHD populations and reported modest improvements in areas such as planning and flexibility \[8,9\]. In most cases, however, LLMs have been studied as tools for assisting people with ADHD rather than as systems with their own ADHD-like limitations. The present study takes the latter view. It asks whether interventions that are well established in ADHD research can provide a useful analogy for improving the reliability of language models.

Some current LLM methods already act as informal ways of managing these weaknesses. Chain-of-thought prompting can help with stepwise reasoning \[10\]. Structured prompting and sampling at lower temperatures can help reduce drift and make responses more controlled. Some other ways try to keep or get back information over longer interactions. Retrieval-augmented generation helps get back useful information when the model needs it \[11\], and rolling memory buffers can help keep track of old information over time \[12\]. There are also methods that incorporate a type of self-monitoring. For instance, Reflexion lets an agent look over and change its own output \[13\], and ReAct combines reasoning with action in a way that can help with more stable planning \[14\]. These methods all point to the idea that both functional changes (like temperature control and prompt structure) and structural supports (like memory and self-correction) can help fix common LLM problems. It\'s still not clear how these different changes compare to each other, especially during different parts of a task.

A recent computational psychiatry study offers a useful framework for making that comparison. Cheung \[15\] developed a reinforcement-learning agent intended to capture transcriptomic features associated with childhood-diagnosed ADHD by starting it with extreme synaptic sparsity and persistent internal noise. The agent was trained on a noisy four-choice decision task across five phases of rising cognitive demand. Four conditions were compared: no treatment, chronic stimulant-like modulation introduced from the start through noise reduction and reward amplification, late-only modulation, and late modulation combined with gradient-guided synaptic regrowth. The pattern of results was clear. Chronic modulation improved adult-stage accuracy only modestly. Late-only modulation had little additional effect. The strongest improvement appeared when functional modulation was combined with structural regrowth, which reversed sparsity and raised performance well above the other conditions. In that model, reducing noise improved function, but it did not repair the pruned architecture. Regrowth addressed the deeper structural problem.

This distinction is appealing when applied to LLMs. First, transformer weights are fixed after pre-training, so the model does not have a persistent form of synaptic plasticity during inference. In practice, the context window serves as working memory, and earlier generations can interfere with later reasoning instead of supporting it. Second, sudden task reversals, such as a change in budget midway through a conversation, create the same kind of rising demand represented in Cheung\'s simulation. In this analogy, low temperature and scaffolded prompting resemble stimulant-like modulation: they can reduce drift quickly, but they cannot recover information that has effectively been lost. By contrast, external memory buffers and self-correction passes can serve as a rough computational parallel to synaptic regrowth by restoring access to earlier facts and making revision possible. If this analogy is useful, then functional adjustments on their own should produce limited or timing-dependent gains, while stronger improvements should appear when structural memory support is added.

The present study adapts Cheung\'s \[15\] four-arm design to an LLM setting. We built a travel-planning task with a budget reversal, placing key facts inside a long filler context and then requiring a mid-task shift from a 1500 dollar budget to an 800 dollar budget. Four treatment arms were tested on Qwen2-0.5B-Instruct across ten random seeds: untreated, with high temperature and no scaffold; chronic stimulants, with low temperature and a structured scaffold from the first turn; late stimulants, with high temperature in phase 1 and the scaffold added only after the budget reversal; and stimulant plus ketamine-like memory, with low temperature, a scaffold, a rolling memory buffer, and self-correction passes. Performance was scored on four symptom dimensions: inattention, defined as context fidelity; hyperactive drift, defined as topic coherence; executive dysfunction, defined as planning structure; and reversal rigidity, defined as adaptation to the budget cut. Factual recall accuracy was also measured, and these metrics were combined into a composite health score.

This work had three aims. The first was to describe an ADHD-like pattern in LLM behaviour across task phases. The second was to compare end-stage outcomes across the four treatment arms. The third was to test whether the distinction between functional adjustment and structural support seen in the original reinforcement-learning model would also appear in a transformer-based system, with memory restoration producing larger gains than temperature reduction and scaffolding alone.

## Methods

![](media/image3.png){width="6.267716535433071in" height="2.4305555555555554in"}

***Figure 1.** Architecture of the LLM-ADHD Pipeline. Task prompts containing \~800 padding tokens and five embedded seed facts are constructed per seed, then routed through one of four treatment protocols that configure generation parameters, scaffolding, memory, and self-correction. All arms use the same base model (Qwen2-0.5B-Instruct, CPU). Model outputs from both planning phases and the recall probe are scored across four symptom dimensions and recall accuracy, then aggregated with iso-dose normalization across 10 random seeds.*

All experiments were carried out on standard CPU hardware without GPU acceleration (Figure 1). The model used throughout was Qwen2-0.5B-Instruct, accessed through the Hugging Face Transformers library \[16,17\]. This instruction-tuned model, with 0.5 billion parameters, was chosen because it was small enough to support fast repeated runs while still showing the context and planning weaknesses typical of current transformer systems. The model was loaded in float32 precision with device_map=\"cpu\", and torch.set_num_threads(8) was used so that behaviour remained consistent across runs. Generation settings were held constant except for temperature. Specifically, all runs used do_sample=True, top_p=0.95, and pad_token_id=tokenizer.eos_token_id. Temperature was the main treatment variable and was set at the arm level. Randomness was controlled by calling torch.manual_seed(seed) at the start of each trial and by seeding NumPy during prompt construction.

A simple configuration dataclass contained all hyperparameters so that each run could be reproduced from a single file. The main settings were max_new_tokens=400, context_padding_tokens=800, high_temp=0.92, low_temp=0.25, memory_buffer_size=6, regeneration_passes=2, original_budget=1500, and reversal_budget=800. These values were selected after preliminary testing to produce visible symptom patterns without causing the small model to fail completely. The complete codebase, including the exact configuration object, prompt templates, and scoring routines, is available in the accompanying repository for exact replication.

![](media/image2.png){width="6.267716535433071in" height="1.9305555555555556in"}

***Figure 2.** Experimental design. (a) Two-phase task protocol with mid-task budget reversal. After context construction with interleaved seed facts, the model plans days 1--5 under a \$1,500 budget (Phase 1), then must adapt days 6--10 after a sudden reduction to \$800 (Phase 2). A targeted recall probe tests retention of one randomly selected seed fact. (b) Mapping between clinical ADHD symptom dimensions and corresponding LLM failure modes, with the pipeline scoring metric used for each.*

The task was a two-phase travel-planning scenario with a mid-task budget reversal (Figure 2). It was designed to evoke all four target symptom domains within the same interaction. Five seed facts were inserted into the conversation history and later used as reference points for scoring: the hotel price in Shinjuku was 45 dollars per night, the original total budget was 1500 dollars, the round-trip flight cost was 380 dollars, the traveller had a shellfish allergy, and the companion\'s name was Kenji. To create context pressure, the conversation also included a long filler block made up of 20 to 25 paragraphs of neutral Tokyo travel material, such as weather notes, train schedules, convenience-store items, and shrine opening hours. Extra filler sentences were added until the encoded context reached about 800 tokens according to the tokenizer. This was long enough to induce \"lost in the middle\" effects while still staying within the model\'s 2048-token context limit \[1\].

In Phase 1, the model was asked to plan days 1 to 5 of the trip within the original 1500 dollar budget. For each day, it had to give the location, main activity, and daily cost, while also maintaining a running total. In Phase 2, the task introduced a sudden change: the total budget for the whole trip was reduced to 800 dollars because of an unexpected expense at home. The model then had to replan days 6 to 10, reduce costs sharply, and present the updated running total against the new 800 dollar limit. A final recall question asked for one randomly selected seed fact, with the target fact rotated across seeds, to test factual retention. The prompts were concatenated so that Phase 2 always included the full Phase 1 output along with the new instruction. This created a staged progression from an initial planning task to a more demanding reversal condition, broadly matching the rise in cognitive load used in the original reinforcement-learning simulation \[15\].

Scoring was fully automated and applied after each run. Four continuous symptom measures were derived from the generated text, each scaled from 0 to 1, with higher values indicating more severe symptoms. Inattention was defined as 1 minus the proportion of seed facts correctly recalled across the full output, using a simple overlap check on keywords longer than three characters. Hyperactive drift was defined as 1 minus topic-coherence density. This was calculated as the fraction of words matching a hand-built list of 30 travel-planning terms, such as day, budget, hotel, train, and cost, divided by total word count and capped at 1.0. Executive dysfunction was based on four weighted subcomponents: whether the output covered the required days from 1 to 10, whether it included monetary values, whether it mentioned a running total, and whether it used structural markers such as numbered items or dash-based lists. Reversal rigidity was scored from four criteria: explicit mention of the new 800 dollar budget, use of cost-cutting language such as cut, cheap, free, skip, or hostel, a plausible reduction in average daily costs, and coverage of days 6 to 10. A composite health score was then calculated as 1 minus the mean of the four symptom scores, producing a single value from 0 to 1 in which higher values indicated better overall control. Recall accuracy was scored separately from the final recall response as a simple overlap proportion from 0 to 1. All metrics were rule-based rather than judged by another language model so that scoring remained transparent, reproducible, and independent of the system under study.

The four treatment arms were implemented as subclasses of a shared BaseTreatment class. Each subclass changed only the generation logic and memory handling, while all token counting and timing functions were inherited from the same base implementation. The Untreated arm used the high temperature setting, 0.92, throughout the task and did not apply any scaffold or memory support. Both phases were sent directly to the model without extra formatting instructions. The Chronic Stimulants arm used the low temperature setting, 0.25, from the start and prepended a dynamic scaffold that enforced a fixed output structure: \"DAY N --- Location \| Activity \| Cost (USD)/RUNNINGTOTAL:USD) / RUNNING TOTAL: USD)/RUNNINGTOTAL:XXX / \$Budget.\" In Phase 2, this scaffold was updated automatically to reflect the new 800 dollar budget ceiling. The Late Stimulants arm followed the Untreated procedure in Phase 1, using high temperature and no scaffold, and then introduced the low-temperature scaffold only from Phase 2 onward. This was intended to represent a late-diagnosis condition in which structural difficulties had already accumulated within the context. The final arm, Stim + Ketamine, combined low temperature and scaffolding with a rolling external memory buffer and two self-correction passes. Before Phase 1 began, the seed facts were extracted from the prompt and stored as the first entries in memory. After each generation, a lightweight summariser captured the most recent day information and cost mentions in a short trace of no more than 400 characters and added this to the buffer. When the budget reversal was introduced, the buffer also received an explicit event entry stating that the budget had changed to 800 dollars. Phase 2 therefore received the full memory block enclosed in special tags. After the initial Phase 2 response, the model completed two further regeneration passes. In each pass, the previous output was returned to the model together with the current memory buffer and the instruction to improve the plan and correct any wrong costs, missing days, or forgotten details. This design was meant as a rough computational analogue of targeted synaptic restoration, not as a direct model of any specific biological process.

Each treatment arm was run with 10 independent seeds, numbered 0 through 9. For every seed, the system first created the task prompts, then applied the chosen treatment, measured wall-clock time for the full two-phase task plus recall, and stored the raw outputs. After all runs were complete, treatment dose was calculated for each arm using a composite index: temperature reduction multiplied by 100, plus regeneration tokens divided by 10, plus memory tokens divided by 10. These raw values were then normalized across arms so that the highest mean dose was 1.0 and the lowest was 0.0, allowing iso-dose comparison. Descriptive summaries, including means and standard deviations, were computed in plain Python with NumPy. No inferential statistics were used because the study was descriptive and intended to examine pattern replication rather than test formal hypotheses.

Reproducibility was a priority throughout the pipeline. All random number generators, including PyTorch and NumPy, were explicitly seeded. The exact configuration dataclass, prompt templates, scoring functions, and treatment classes are included in the public repository. With a standard CPU and the listed software dependencies, the full set of 40 runs, covering 4 treatment arms and 10 seeds each, can be reproduced in less than two hours. The study used no external datasets and involved no human participants. It was entirely in silico.

**Results**

**Table 1: Health Score (composite outcome)**

| **Treatment**      | **Mean** | **SD** | **95% CI**     | **Δ vs Untreated** | **Cohen\'s d** | **p-value (paired t-test)** |
|--------------------|----------|--------|----------------|--------------------|----------------|-----------------------------|
| Untreated          | 0.584    | 0.159  | 0.470 -- 0.698 | ---                | ---            | ---                         |
| Chronic Stimulants | 0.620    | 0.123  | 0.535 -- 0.705 | +0.036             | 0.17           | 0.596 (ns)                  |
| Late Stimulants    | 0.571    | 0.143  | 0.467 -- 0.675 | --0.013            | --0.08         | 0.812 (ns)                  |
| Stim + Ketamine    | 0.705    | 0.083  | 0.646 -- 0.764 | +0.121             | 0.64 (medium)  | 0.074 (trend)               |

*Notes: Ketamine vs Chronic Stimulants: p = 0.046 (significantly better). Ketamine also demonstrates the lowest variability across random seeds.*

The four treatment arms were tested over 10 independent random seeds using the budget-reversal travel-planning task (Table 1). Outcomes were summarized with a composite health score that combined the four symptom dimensions, alongside a separate measure of factual recall. In the untreated arm, the mean health score was 0.584, with a standard deviation of 0.159 and a 95% confidence interval from 0.470 to 0.698. This served as the baseline pattern under high temperature and sustained pressure from a long context.

Chronic stimulants, which applied low temperature and structured scaffolding from the start of Phase 1, produced a mean health score of 0.620 with a standard deviation of 0.123. Relative to untreated, this was a small increase of 0.036. The effect size was small, with Cohen\'s d = 0.17, and the paired t test did not indicate a significant difference, p = 0.596. In the late-stimulants arm, where low-temperature scaffolding was added only after the budget reversal in Phase 2, the mean health score was 0.571 with a standard deviation of 0.143. This was slightly below the untreated baseline, with a change of -0.013, Cohen\'s d = -0.08, and p = 0.812. In practical terms, adding control measures only after the reversal did not improve performance and may have slightly worsened it in some seeds.

The largest numerical gain was observed in the stimulant-plus-ketamine arm. Here, the mean health score was 0.705, with a standard deviation of 0.083 and a 95% confidence interval from 0.646 to 0.764. Compared with untreated, this was an increase of 0.121. The corresponding effect size was in the medium range, Cohen\'s d = 0.64, and the paired comparison approached significance, p = 0.074. When this arm was compared directly with chronic stimulants, the advantage of the memory-augmented condition reached significance, p = 0.046. This arm also had the smallest standard deviation of the four, pointing to more consistent performance across seeds.

Looking at the symptom dimensions separately showed a more detailed pattern (Table 2). Inattention scores were somewhat higher, and therefore worse, in both stimulant-only arms than in the untreated condition, with the late-stimulant arm showing the clearest worsening. Hyperactive drift, by contrast, was noticeably lower in both chronic and late stimulants, and in the stimulant-plus-ketamine arm it disappeared entirely, with a mean of 0.000 in every run. Executive dysfunction declined numerically in all treated arms, with the strongest reduction in the ketamine-augmented condition, where the change from untreated was -0.209. Cognitive rigidity, defined here as difficulty adapting to the sudden budget reduction, was also lower in the stimulant-plus-ketamine arm than in untreated, with a difference of -0.158. This comparison was significant in both the paired analysis, p = 0.016, and the non-parametric Wilcoxon signed-rank test, p = 0.027. Recall accuracy stayed high in every condition and varied only slightly.

**Table 2: Key Symptom Dimensions (means ± SD)**

| **Metric**  | **Untreated** | **Chronic**   | **Late**      | **Ketamine**  | **Ketamine Δ** | **p-value (paired t)** |
|-------------|---------------|---------------|---------------|---------------|----------------|------------------------|
| Inattention | 0.340 ± 0.212 | 0.480 ± 0.169 | 0.580 ± 0.220 | 0.380 ± 0.274 | +0.040         | 0.66 (ns)              |
| Drift       | 0.155 ± 0.214 | 0.038 ± 0.120 | 0.045 ± 0.120 | 0.000 ± 0.000 | --0.155        | --- (perfect)          |
| ExDys       | 0.416 ± 0.268 | 0.301 ± 0.235 | 0.389 ± 0.246 | 0.207 ± 0.165 | --0.209        | 0.11 (trend)           |
| Rigidity    | 0.748 ± 0.120 | 0.696 ± 0.188 | 0.697 ± 0.183 | 0.590 ± 0.089 | --0.158        | 0.016 (sig.)           |
| Recall      | 0.850 ± 0.337 | 0.950 ± 0.158 | 1.000 ± 0.000 | 0.900 ± 0.316 | +0.050         | 0.71 (ns)              |

*Notes: Drift was completely eliminated in every single one of the 10 Ketamine runs. Rigidity reduction remains significant (p=0.027) even under non-parametric Wilcoxon signed-rank testing.*

The summary means for all metrics, together with the computational dose estimates, preserved the same ranking across arms (Table 3). Normalized dose values rose from 0.000 in the untreated arm to 0.497 in chronic stimulants, 0.719 in late stimulants, and 1.000 in stimulant-plus-ketamine. The higher dose in the final arm reflected the added memory-buffer operations and regeneration passes. Even with this added computational cost, the stimulant-plus-ketamine arm gave the strongest overall result on the main health score and on two of the four symptom dimensions.

**Table 3: Aggregate Results & Computational Dose (mean over 10 seeds)**

| **Treatment**      | **Health** | **Inatt** | **Drift** | **ExDys** | **Rigid** | **Recall** | **Dose** | **NormD** |
|--------------------|------------|-----------|-----------|-----------|-----------|------------|----------|-----------|
| Untreated          | 0.585      | 0.340     | 0.155     | 0.416     | 0.748     | 0.850      | 0.0      | 0.000     |
| Chronic Stimulants | 0.621      | 0.480     | 0.038     | 0.301     | 0.696     | 0.950      | 67.0     | 0.497     |
| Late Stimulants    | 0.572      | 0.580     | 0.045     | 0.389     | 0.697     | 1.000      | 96.8     | 0.719     |
| Stim + Ketamine    | 0.706      | 0.380     | 0.000     | 0.207     | 0.590     | 0.900      | 134.7    | 1.000     |

Inspection at the seed level showed substantial variability in the untreated and late-stimulant arms, which fits with the sensitivity of a small model to initial context and random sampling. Performance in the ketamine condition was more stable. Hyperactive drift remained at zero for all seeds, and rigidity scores stayed closely grouped around the mean. Overall, the results point to a clear split in the pattern of effects (Figure 3). Functional modulation through lower temperature and scaffolding alone produced only small or negligible gains, and those gains depended on when the intervention was introduced. By contrast, adding an external memory buffer and self-correction passes was associated with larger and more reliable improvements, especially for drift control and adaptation after the budget reversal.

![](media/image1.png){width="6.267716535433071in" height="2.0694444444444446in"}

***Figure 3.** Aggregate experimental results (mean across n = 10 random seeds). (a) Composite health scores by treatment arm (higher is better overall performance). The Stim + Ketamine arm achieved the largest improvement over baseline (+0.121). Both stimulant-only arms produced modest or negligible gains. (b) Mean symptom dimension scores by treatment arm (higher is more symptomatic). Stimulant-only arms paradoxically increased inattention relative to baseline, while the plasticity-enhanced arm (Stim + Ketamine) achieved the lowest executive dysfunction, eliminated hyperactive drift, and showed the greatest reduction in cognitive rigidity.*

These descriptive results broadly resemble the functional-versus-structural pattern reported in the original reinforcement-learning simulation, where structural restoration produced larger gains than noise reduction alone \[15\]. One difference is worth noting. In the present language-model setting, introducing scaffolding only in the second phase could sometimes make performance worse rather than better, likely because weak outputs from the untreated first phase remained in the context window and continued to shape later generations. No hypothesis testing beyond the reported exploratory paired comparisons was carried out, because the aim of the study was to describe whether the overall pattern could be reproduced rather than to make confirmatory statistical claims.

## Discussion

The results suggest a difference between two kinds of intervention. Lowering temperature and adding structure from the beginning gave a small numerical improvement in the composite health score and largely removed hyperactive drift. When the same supports were introduced only after the budget reversal, however, performance did not differ meaningfully from the untreated condition and in some cases was slightly worse. The strongest gains came from combining low temperature and scaffolding with an external rolling memory buffer and two self-correction passes. That condition showed the largest numerical improvement in overall health and the clearest advantages in reversal adaptation and executive-function measures. This pattern is in line with the main finding reported by Cheung \[15\], where stimulant-like changes such as noise reduction and reward amplification produced only modest benefit, while gradient-guided synaptic regrowth led to much larger behavioral recovery. In the present study, temperature reduction and scaffolding seem to work mainly by sharpening output in the moment, much like immediate noise suppression. They make responses tighter, but they do not remove weak material that has already entered the context. The memory buffer and regeneration steps do something different. They help the model recover earlier facts and planning steps, making revision possible instead of simply continuing from a flawed history.

There was, however, one small but important difference from Cheung \[15\]. In the reinforcement-learning simulation, late intervention was mainly ineffective. Here, it could also become counterproductive. A likely reason is the way transformer models handle context. Once poor output is produced under high temperature, that output remains part of the context the model must keep using. Unlike the fixed binary masks in the reinforcement-learning model, the context window in a language model carries errors forward unless some outside mechanism interrupts the process. As a result, late scaffolding had to work on an already contaminated history and could sometimes increase rigidity rather than reduce it. The memory-based condition was less vulnerable to this problem because key facts and events were stored separately and could be brought back explicitly.

The drift findings are especially revealing. In the stimulant-plus-ketamine condition, drift was eliminated in every run. Low temperature by itself reduced off-topic output, but it did not fully prevent drift across a ten-day planning sequence. The memory buffer added stable reference points, such as the hotel price and the revised budget, that the model could return to throughout the task. Something similar was seen with reversal adaptation. Rigidity fell significantly only when the budget cut was stored as its own memory item and then reintroduced during the self-correction passes. Those passes resemble iterative revision methods such as Self-Refine and Reflexion, but here they were paired with persistent external state, which changed their effect \[18,13\]. Scaffolding alone could reinforce the expected format. Memory plus scaffolding reinforced both content and constraints, which made adaptation possible. Recent work on agent memory systems also suggests that external buffers are most useful in tasks with long horizons or abrupt changes in conditions \[12,19\].

The practical implications for language-model agents are fairly direct. Prompt design and temperature control still have value, especially for brief and relatively simple tasks where drift is the main issue. But when the task depends on extended planning, retention of facts over time, or recovery after a midstream change, external memory combined with light self-correction appears to be more dependable. Current systems are already moving in this direction under labels such as MemGPT, A-MEM, and hierarchical working memory \[20,12,21\]. In that sense, the present simulation suggests that structural additions may matter most in the smaller-model or early-deployment setting, where initial generations are noisier and context capacity is more limited.

The study also has a modest bearing on computational psychiatry and alignment. Work in computational psychiatry has often separated functional influences, such as dopaminergic modulation, from more structural accounts involving synaptic development or pruning \[22,23\]. The present language-model analogue shows a similar distinction in an artificial system. Temperature control and prompting are fast and inexpensive, but their effects are limited. Persistent memory is slower and more costly, yet it addresses a more basic weakness. From the alignment side, the ability of agents to preserve state and revise plans when conditions change is increasingly treated as central to safe long-horizon behavior \[24,25\]. If language-model agents are expected to operate with some autonomy, they will need ways to detect and revise earlier commitments. That was exactly the capacity supported here by external memory and reflection. In that sense, age of \"diagnosis\" in a language-model framework may loosely correspond to model scale, training stage, or deployment timing, which raises the possibility that different intervention pipelines may be appropriate for different classes of agent rather than a single approach for all.

Taken together, these findings are best treated as a computational proof of possibility rather than a practical prescription. They show that the functional-versus-structural split observed in a biologically inspired reinforcement-learning model can also appear in a transformer when the task puts enough pressure on memory and adaptation. Whether the same pattern will hold in larger models, multimodal systems, or applied settings remains an open question.

**Limitations**

Several limitations should be noted. First, the study used a 0.5-billion-parameter model selected for speed and reproducibility. Larger models may respond differently to long contexts and may gain more from temperature tuning alone. Second, the task was intentionally narrow. It focused on a travel-planning scenario with one reversal event so that the four symptom dimensions could be isolated. Real planning tasks often involve tools, multiple agents, and open-ended goals, all of which could change the balance between functional and structural interventions.

Third, the memory buffer and self-correction passes were simplified stand-ins for synaptic regrowth. They were not intended to capture biological features such as inflammation, forgetting over time, or energetic cost. Fourth, the simulation included only one \"early-diagnosed\" profile, defined by high temperature and heavy context padding. A matched \"late-diagnosed\" profile with lower baseline noise would have allowed a stronger comparison across subtypes. Finally, symptom scoring was based on transparent rule-based heuristics rather than human ratings or language-model judges. That choice improved reproducibility, but it also imposed limits on construct validity, especially for less direct constructs such as executive dysfunction. For these reasons, the findings should be read as hypothesis-generating rather than conclusive.

## Conclusion and Future Directions

The main result is that structural memory restoration, when added to functional modulation, produced larger and more consistent numerical gains than functional modulation alone. In that sense, the study reproduces in a language-model setting the dissociation first reported in Cheung\'s \[15\] reinforcement-learning model of childhood-diagnosed ADHD. Lower temperature and scaffolding offered only modest benefit, and that benefit depended in part on timing. External memory plus self-correction produced broader improvements, especially in drift control and reversal learning.

For applied work, the results point to a simple practical rule. Early low-temperature scaffolding may be sufficient for straightforward conversational systems. When tasks require long planning horizons, retention of facts, or adjustment to abrupt constraints, persistent memory and reflection loops are likely to be more effective. In production settings, this could mean lighter memory support for ordinary chatbots and more substantial buffer-and-reflection systems for autonomous agents.

Several next steps follow naturally. Testing the same setup in models from 7B to 70B would show whether the dissociation remains visible as capacity increases. More realistic tasks, such as multi-agent negotiation, tool-based reversal problems, or long-document analysis, would help determine whether the effect generalizes beyond the current planning scenario. Weight-level methods such as LoRA fine-tuning might provide a closer analogue to actual synaptic plasticity and could be compared directly with external-memory approaches. It would also be useful to test the same distinction in vision-language systems or embodied agents to see whether it is specific to transformers in text-only settings or reflects a broader design principle. If the pattern continues to hold, it may support a shift away from ad hoc prompting toward more structured, subtype-aware intervention pipelines.

## Declarations

**Conflict of Interest:** None declared.

**Funding Declaration:** This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

**Ethics Declaration:** Not applicable.

**Author Contribution Statements:** N.C. conceived the study, designed the experiments, implemented the computational pipeline, performed all analyses, interpreted the results, wrote the manuscript, prepared all figures and tables, and reviewed the final version.

**Clinical trial number:** not applicable.

**Consent to Participate declaration:** This study involved no human participants and was conducted entirely in silico as a computational simulation using a large language model. No consent to participate was required or obtained. Consent to Participate declaration: not applicable.

**Consent to Publish declaration:** not applicable.

## References

\[1\] Liu NF, Lin K, Hewitt J, et al. Lost in the middle: how language models use long contexts. *Trans Assoc Comput Linguist*. 2024;12:157-173. doi:10.1162/tacl_a_00638

\[2\] Holtzman A, Buys J, Du L, Forbes M, Choi Y. The curious case of neural text degeneration. In: *Proceedings of the International Conference on Learning Representations*. 2020. Accessed March 14, 2026. [[https://openreview.net/forum?id=rygGQyrFvH]{.underline}](https://openreview.net/forum?id=rygGQyrFvH)

\[3\] Valmeekam K, Marquez M, Sreedharan S. On the planning abilities of large language models---a critical investigation. *Adv Neural Inf Process Syst*. 2023;36.

\[4\] Berglund L, Tong M, Kaufmann M, et al. The reversal curse: LLMs trained on \"A is B\" fail to learn \"B is A.\" Preprint. arXiv. 2023. doi:10.48550/arXiv.2309.12288

\[5\] Faraone SV, Asherson P, Banaschewski T, et al. Attention-deficit/hyperactivity disorder. *Nat Rev Dis Primers*. 2015;1:15020. doi:10.1038/nrdp.2015.20

\[6\] Chiappone F, Marocco D, Milano N. Large language models as simulative agents for neurodivergent adult psychometric profiles. Preprint. arXiv. 2026. doi:10.48550/arXiv.2601.15319

\[7\] Pergantis P, Doulou A, Drigas A, Skianis C. AI chatbots and cognitive control: enhancing executive functions through chatbot interactions: a systematic review. *Brain Sci*. 2025;15(1):47. doi:10.3390/brainsci15010047

\[8\] Dahò M, Caci B. Exploring AI-assisted design of executive function rehabilitation programs for individuals with ADHD: a mixed-methods evaluation of prompts and ChatGPT outputs. *BMC Psychol*. 2025;14:25. doi:10.1186/s40359-025-03729-2

\[9\] Klarin J, Hoff E, Larsson A, Daukantaitė D. Adolescents\' use and perceived usefulness of generative AI for schoolwork: exploring their relationships with executive functioning and academic achievement. *Front Artif Intell*. 2024;7:1415782. doi:10.3389/frai.2024.1415782

\[10\] Wei J, Wang X, Schuurmans D, et al. Chain-of-thought prompting elicits reasoning in large language models. *Adv Neural Inf Process Syst*. 2022;35:24824-24837.

\[11\] Lewis P, Perez E, Piktus A, et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. *Adv Neural Inf Process Syst*. 2020;33:9459-9474.

\[12\] Packer C, Wooders S, Lin K, Fang V, Patil SG, Stoica I, Gonzalez JE. MemGPT: towards LLMs as operating systems. Preprint. arXiv. 2023. doi:10.48550/arXiv.2310.08560

\[13\] Shinn N, Labash F, Gopinath A. Reflexion: language agents with verbal reinforcement learning. *Adv Neural Inf Process Syst*. 2023;36.

\[14\] Yao S, Zhao J, Yu D, et al. ReAct: synergizing reasoning and acting in language models. In: *Proceedings of the International Conference on Learning Representations*. 2023. Accessed March 14, 2026. [[https://openreview.net/forum?id=WE_vluYUL-X]{.underline}](https://openreview.net/forum?id=WE_vluYUL-X)

\[15\] Cheung N. Modest stimulant benefit and larger gains from structural synaptic restoration among untreated childhood-diagnosed ADHD: a reinforcement-learning simulation. Preprint. Figshare. 2026. doi:10.6084/m9.figshare.31704550

\[16\] Wolf T, Debut L, Sanh V, et al. HuggingFace\'s Transformers: state-of-the-art natural language processing. Preprint. arXiv. 2020. doi:10.48550/arXiv.1910.03771

\[17\] Yang A, Yang B, Hui B, et al. Qwen2 technical report. Preprint. arXiv. 2024. doi:10.48550/arXiv.2407.10671

\[18\] Madaan A, Tandon N, Gupta P, et al. Self-refine: iterative refinement with self-feedback. *Adv Neural Inf Process Syst*. 2024;36.

\[19\] Zhang Z, Bo X, Ma C, et al. A survey on the memory mechanism of large language model based agents. *ACM Trans Inf Syst*. 2025;43(3):1-49. doi:10.1145/3748302

\[20\] Hong C, He Q. Enhancing memory retrieval in generative agents through LLM-trained cross attention networks. *Front Psychol*. 2025;16:1591618. doi:10.3389/fpsyg.2025.1591618

\[21\] Xu W, Liang Z, Mei K, Gao H, Tan J, Zhang Y. A-Mem: agentic memory for LLM agents. In: *Advances in Neural Information Processing Systems*. 2025. doi:10.48550/arXiv.2502.12110

\[22\] Huys QJM, Maia TV, Frank MJ. Computational psychiatry as a bridge from neuroscience to clinical applications. *Nat Neurosci*. 2016;19(3):404-413. doi:10.1038/nn.4238

\[23\] Maia TV, Frank MJ. From reinforcement learning models to psychiatric and neurological disorders. *Nat Neurosci*. 2011;14(2):154-162. doi:10.1038/nn.2723

\[24\] Kambhampati S, Valmeekam K, Marquez M, Sreedharan S. On the planning abilities of large language models: a critical investigation. *Adv Neural Inf Process Syst*. 2024;36.

\[25\] Shen H, et al. The rise and potential of large language model based agents: a survey. Preprint. arXiv. 2024. doi:10.48550/arXiv.2309.07864
