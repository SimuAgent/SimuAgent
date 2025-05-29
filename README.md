# SimuAgent: An LLM-based Simulink Modeling Assistant

SimuAgent is an LLM-powered modeling and simulation agent designed for Simulink. This repository contains the code and resources related to the SimuAgent framework, with a focus on its training mechanisms and tool integration.

**Key Features & Concepts:**

* **Lightweight Python Representation:** Simulink models are represented as Python dictionaries, making them more amenable to LLM processing and manipulation.
* **Plan-Execute Architecture:** SimuAgent employs a lightweight plan-execute paradigm for task decomposition and execution.
* **Reflection-GRPO (ReGRPO):** A novel reinforcement learning algorithm that augments Group Relative Policy Optimization (GRPO) with self-reflection traces. This provides richer intermediate feedback, accelerating convergence and improving robustness, especially in tasks requiring tool use. (See Fig2.png for ReGRPO architecture).
* **Tool Integration:** The framework is designed to allow the LLM to leverage external tools for tasks like block searching and model analysis.

![SimuAgent Workflow](docs/SimuAgent.png)
*Fig1: Comparison between SimuAgent and conventional workflows*

![ReGRPO Architecture](docs/ReGRPO.png)
*Fig2: ReGRPO architecture*

## Quick Start

To get a basic understanding of how to run ReGRPO:

1.  **Without tools:**
    ```bash
    python quick_start.py
    ```
2.  **With tool integration:**
    ```bash
    python quick_start_with_tools.py
    ```

## Current Status & Future Work

This repository contains the initial implementation of SimuAgent and the ReGRPO algorithm. We are actively working on refining the framework, expanding the toolset, and improving performance on more complex modeling tasks.

*(Note: This project is currently under active development, and some components may be subject to change.)*
