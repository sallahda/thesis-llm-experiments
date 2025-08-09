# LLM Vulnerability Narrative Experiments

This repository contains the experimental code for evaluating LLM-generated vulnerability narratives across three dimensions: clarity, accuracy, and stakeholder alignment.

## Research Context

These experiments support the thesis research on automated stakeholder-specific vulnerability reporting. The code tests three LLMs (Claude Sonnet, Nova Pro, Llama 3B) with four prompt engineering strategies across three visualization types.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. Configure AWS credentials for Bedrock access

## Core Files

- `accuracy_experiment.py` - Evaluates factual accuracy of narratives using GPT-4 judge
- `clarity_experiment.py` - Measures readability using Flesch-Kincaid scoring
- `stakeholder_experiment.py` - Assesses alignment with stakeholder needs
- `judge_llm.py` - GPT-4 evaluation wrapper for accuracy and stakeholder assessments
- `experiment_runner.py` - Orchestrates all experiments
- `results_analysis.py` - Comprehensive analysis and visualization of results

## Usage

### Run Individual Experiments

```bash
# Clarity experiment (Flesch-Kincaid readability)
python clarity_experiment.py

# Accuracy experiment (GPT-4 judge)
python accuracy_experiment.py

# Stakeholder alignment experiment (GPT-4 judge)
python stakeholder_experiment.py
```

### Run All Experiments

```bash
python experiment_runner.py
```

### Analyze Results

```bash
python results_analysis.py
```

## Experimental Design

- **Models**: Claude 3.7 Sonnet, Amazon Nova Pro, Meta Llama 3B
- **Prompt Strategies**: General, Role-based, Few-shot, Chain-of-thought
- **Visualizations**: Stacked bar chart, heatmap, treemap
- **Total Combinations**: 36 narratives (3×4×3)

## Key Findings

The experiments reveal trade-offs between narrative qualities:
- Amazon Nova Pro with Few-shot prompting optimizes for clarity
- Role-based prompting enhances stakeholder alignment
- No single configuration maximizes all metrics simultaneously
