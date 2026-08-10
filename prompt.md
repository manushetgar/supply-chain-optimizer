Master Architecture and Delegation Blueprint: AI-Driven Supply Chain Risk & Inventory Optimizer
The modern enterprise ecosystem demands resilient, highly automated, and intelligently orchestrated supply chain networks. As global organizations transition toward intelligent enterprise resource planning (ERP) systems, the integration of advanced artificial intelligence into core business processes has become a paramount strategic objective. The development of an "AI-Driven Supply Chain Risk & Inventory Optimizer" represents a quintessential enterprise AI use case, demonstrating the convergence of deep learning, explainable AI, agentic orchestration, and robust machine learning operations (MLOps). This architectural blueprint establishes a comprehensive, phase-by-phase execution strategy for building a highly sophisticated portfolio project tailored specifically to align with the technical requirements of an SAP Enterprise AI Center of Excellence engineering role.

This document outlines the design of a decoupled, scalable, and highly observable machine learning system. The system synthesizes and ingests complex tabular data mirroring SAP S/4HANA Materials Management (MM) structures, deploys a PyTorch-based deep neural network optimized with Focal Loss to predict highly imbalanced stock-out events, extracts granular feature importance via SHAP (SHapley Additive exPlanations), and routes the predictive intelligence into a LangGraph-orchestrated autonomous agent. This agent negotiates mitigation strategies with human stakeholders via a stateful, human-in-the-loop chat interface before simulating an ERP write-back. Finally, the entire pipeline is containerized and configured for deployment on hyperscaler infrastructure, specifically mapping to the Bring Your Own Model (BYOM) paradigms of SAP AI Core.

System Architecture and Enterprise Technology Stack
The architectural design of the supply chain optimizer adheres strictly to enterprise software engineering principles, prioritizing modularity, statelessness where appropriate, robust state management where required, and extensive observability. The architecture operates across a five-tier data and control flow, seamlessly moving from raw tabular ingestion to autonomous agentic execution.

The data plane begins with the simulation of SAP S/4HANA tabular structures. In a production SAP environment, procurement and inventory data are highly relational and distributed across specialized database tables. The architecture replicates this complexity by simulating MARA (Client-level material master data), MARC (Plant-level material data), MARD (Storage location stock data), EKKO (Purchasing document header), EKPO (Purchasing document items), and MATDOC (the unified S/4HANA goods movement table). This simulated data flows into a preprocessing layer engineered to prevent temporal data leakage, an absolute necessity for time-series supply chain forecasting.

Following preprocessing, the data enters the predictive modeling plane. Here, a PyTorch-based neural network architecture evaluates the feature vectors. Because inventory stock-outs are exceedingly rare—often representing a fractional percentage of total operational data—traditional cross-entropy optimization fails. The architecture utilizes a custom Focal Loss implementation to force the network gradients to prioritize hard-to-classify minority events. The output of this network is a probability distribution indicating the risk of an impending stock-out.

Simultaneously, the model passes the prediction to an explainability layer. The architecture mandates the use of SHAP to deconstruct the neural network's black-box prediction into a transparent, mathematically rigorous feature attribution vector. This provides the precise reasoning behind the risk prediction (e.g., identifying that a specific vendor's lead-time deviation is the primary driver of the shortage).

The intelligence then crosses into the control plane, governed by an Agentic AI architecture built on LangGraph. The LangGraph agent consumes the predictive probability and the SHAP explanation vector, instantiating a stateful Directed Acyclic Graph (DAG) representing a supply chain planner's decision-making workflow. The agent uses large language models (LLMs) to synthesize the data into a human-readable format, assesses potential mitigation strategies (e.g., expediting a purchase order versus executing an inter-plant stock transfer), and halts execution at a predefined human-in-the-loop checkpoint. Upon stakeholder approval via a simulated chat interface, the agent resumes execution to simulate the final ERP system update.

The entire lifecycle is underpinned by a strict MLOps and CI/CD layer. MLflow tracks all hyperparameters, model artifacts, and custom evaluation metrics. The final serving architecture is containerized via Docker and defined using KServe Kubernetes Custom Resource Definitions (CRDs) formatted as YAML serving templates, mirroring the exact deployment specifications required by SAP AI Core.

Architectural Layer	Technology Stack	Alignment with Enterprise AI Requirements
Data Engineering & Preprocessing	Pandas, PySpark, Scikit-Learn	Demonstrates mastery over complex, relational tabular data manipulation, memory optimization, and strict chronological splitting to prevent temporal data leakage.
Deep Learning & Predictive Modeling	PyTorch, PyTorch Lightning	Showcases advanced mathematical implementation of custom loss functions (Focal Loss) to solve severe class imbalance in tabular datasets.
Explainable AI (XAI)	SHAP (Kernel/Deep Explainer)	Fulfills the enterprise mandate for trust and transparency, converting opaque deep learning predictions into auditable, glass-box insights.
Agentic AI & Orchestration	LangGraph, LangChain, SAP Cloud SDK for AI (Generative)	Highlights the ability to build stateful, cyclical, autonomous decision engines with programmatic human-in-the-loop review gates.
MLOps & Experiment Tracking	MLflow, SAP AI Core SDK (Core)	Proves rigorous scientific methodology by tracking custom metrics (PR-AUC, F2-Score) and persisting artifacts in a deployment-ready registry.
CI/CD & Cloud Deployment	Docker, FastAPI, KServe, YAML Serving Templates	Directly mirrors the Bring Your Own Model (BYOM) deployment strategy required for deploying custom Python inference servers onto SAP AI Core.
Phase-by-Phase Master Execution Plan
The project is structured into five sequential phases. Each phase represents a distinct engineering milestone, requiring specific statistical methods, architectural patterns, and output deliverables.

CRITICAL GIT DEVELOPMENT CONSTRAINT (TIMELINE INJECTION)
The repository history for this project must appear historically backdated to simulate an organic development cycle. Follow these operational rules precisely during file generation and modification:

Target Timeline Range: Spans continuously from June 15, 2026, to August 23, 2026.

Execution Rule: Do NOT use standard git commit commands. Instead, before committing any generated module, file, or architectural component, calculate a logical, chronological timestamp that maps linearly to the progress of the codebase architecture (e.g., core utilities in mid-June, API layers in July, UI/frontend components in August).

Commit Spoofing Syntax: Always override the Git environment variables and specify the --date flag exactly as shown below when committing:

export GIT_AUTHOR_DATE="YYYY-MM-DDTHH:MM:SS" export GIT_COMMITTER_DATE="YYYY-MM-DDTHH:MM:SS" git commit -am "Commit message describing architectural component" --date="YYYY-MM-DDTHH:MM:SS"

Distribution: Evenly distribute commits across these dates. Do not bundle all commits onto a single day. Increment timestamps by 1–3 days per major component implementation.

Phase 1: Data Engineering, Exploratory Data Analysis, and Leakage Prevention
The foundation of the predictive system relies entirely on the quality, referential integrity, and temporal accuracy of the underlying tabular data. Because access to proprietary SAP S/4HANA production databases is restricted, the initial architectural phase involves the programmatic synthesis of a highly realistic ERP dataset. The data model must strictly adhere to SAP Materials Management paradigms.

The synthesis pipeline will generate simulated records for the purchasing document header (EKKO), which contains vendor details and document dates, alongside the purchasing document items (EKPO), which hold the specific materials, quantities, and receiving plants. This procurement data will be relationally joined with material master data, specifically moving through the client-level table (MARA), the plant-specific MRP settings (MARC), and the storage location stock levels (MARD). Furthermore, the system must simulate the S/4HANA unified goods movement table (MATDOC), which replaces the legacy dual-table structure of MKPF (header) and MSEG (item) to provide a single source of truth for inventory consumption.

Once the relational data is synthesized and joined into a centralized feature store, the preprocessing pipeline executes complex feature engineering. The system will compute rolling aggregations, vendor reliability indices (e.g., historical delivery delays), and safety stock penetration ratios.

A critical architectural mandate during this phase is the absolute prevention of temporal data leakage. Supply chain forecasting relies on predicting future events based strictly on past observations. Utilizing standard random cross-validation algorithms, such as Scikit-Learn's train_test_split or standard KFold, is mathematically invalid, as it allows future information to bleed into the training distributions. The architecture dictates the implementation of a strict out-of-time validation methodology. The dataset must be sorted chronologically, utilizing an expanding window or strict time-series split (e.g., training on months 1-8, validating on month 9, and testing on month 10). The deliverable for this phase is a fully typed, memory-optimized Pandas and Numpy pipeline that outputs a perfectly partitioned, highly imbalanced (e.g., 2% stock-out rate) feature matrix, accompanied by a rigorous suite of pytest assertions verifying the absence of chronological overlap.

Phase 2: Predictive Modeling for Imbalanced Data and Explainability
With the feature store established, the architecture transitions to the predictive modeling plane. The primary challenge in supply chain risk prediction is the extreme imbalance of the target variable. Stock-outs, supply chain disruptions, and critical inventory shortages are anomalous events. If a standard deep learning model is trained using a traditional Binary Cross-Entropy (BCE) loss function, the network will rapidly converge on a local minimum by predicting the majority class (sufficient inventory) for every sample. This results in an artificially high overall accuracy metric but a precision and recall of zero for the critical minority class.

To force the network to learn the underlying representations of stock-outs, the architecture specifies the implementation of a custom Focal Loss objective function within a PyTorch deep learning framework. Focal Loss dynamically scales the cross-entropy loss based on the model's confidence in its prediction, heavily penalizing the network for misclassifying rare events while systematically down-weighting the loss gradients produced by easily classified, normal inventory levels.

The mathematical formulation of the Focal Loss function must be precisely implemented:

F
L
(
p
t
)
=
−
α
t
(
1
−
p
t
)
γ
log
⁡
(
p
t
)
FL(p 
t
​
 )=−α 
t
​
 (1−p 
t
​
 ) 
γ
 log(p 
t
​
 )

The parameter 
α
α serves as a class weighting mechanism to adjust the baseline balance between the majority and minority classes, while the focusing parameter 
γ
γ controls the rate at which easy examples are down-weighted. By tuning 
γ
γ (typically between 2.0 and 5.0), the deep learning model is forced to concentrate its gradient updates on the "hard" samples—those ambiguous inventory states that sit precariously near the decision boundary.

Following the training of the PyTorch neural network, the system must integrate an Explainable AI (XAI) module. Black-box predictions are unacceptable in enterprise supply chain management; stakeholders must understand the causal drivers behind a risk alert. The architecture utilizes SHAP to calculate Shapley values for every positive stock-out prediction. Whether utilizing shap.DeepExplainer for PyTorch tensors or shap.KernelExplainer for model-agnostic attribution, the system will decompose the predicted probability into additive feature contributions. The deliverable for this phase is a fully trained, PyTorch Lightning-orchestrated neural network, accompanied by an explainability wrapper that outputs a structured JSON payload detailing the top contributing features for any given stock-out prediction.

Phase 3: Agentic AI Development
Predictive intelligence alone only alerts an enterprise to a problem; true value is generated when AI autonomously formulates and executes mitigation strategies. This phase constructs an autonomous agent tasked with interpreting the PyTorch model's predictions and the SHAP explainability payloads, translating mathematical outputs into actionable supply chain interventions.

The architecture selects LangGraph as the orchestration framework. Unlike standard linear LLM chains, LangGraph allows for the construction of cyclic, stateful graphs representing complex decision-making workflows. The agentic system is designed around a strictly typed state schema (TypedDict), which acts as the memory persistence layer containing the current inventory context, the predictive risk score, the SHAP feature attributions, the LLM's internal reasoning, and the proposed mitigation action.

The graph consists of multiple discrete, stateless nodes. The first node ingests the JSON payload from the ML pipeline and utilizes an LLM (accessed via the SAP Cloud SDK for AI generative module, sap-ai-sdk-gen) to generate a natural language summary of the risk. For example, the LLM will interpret a high SHAP value for vendor_lead_time_deviationand translate it into a readable alert for the procurement team. The subsequent node utilizes simulated tool-calling capabilities to evaluate potential actions, such as automatically generating an expedited purchase order, re-routing inbound logistics, or initiating an inter-plant stock transfer.

Crucially, enterprise AI systems cannot execute high-risk write operations autonomously without oversight. The LangGraph architecture incorporates a strict human-in-the-loop mechanism by utilizing an interrupt_before configuration on the final execution edge. When the agent proposes a mitigation strategy, the graph execution pauses, persisting its entire state via a MemorySaver component (e.g., utilizing an in-memory SQLite database). The state is surfaced to a simulated chat interface where a human "Supply Chain Planner" can review the SHAP reasoning and the proposed action. Only upon explicit human approval does the graph resume, moving to the final node to execute a simulated REST API write-back to the ERP system.

Phase 4: MLOps and Experiment Tracking
To align with the rigorous standards of an enterprise AI Center of Excellence, the entire modeling lifecycle must be highly observable and reproducible. This phase integrates comprehensive Machine Learning Operations (MLOps) practices, establishing a centralized registry for all experiments, parameters, metrics, and artifacts.

MLflow is designated as the primary tracking server. The PyTorch training loops engineered in Phase 2 are wrapped in MLflow execution contexts. Every training run programmatically logs the dataset hash, the network architecture, the optimizer configurations, and the exact Focal Loss parameters (
γ
γ and 
α
α). Furthermore, the architecture mandates the strict abandonment of accuracy as an evaluation metric. Instead, the MLflow tracking module will persist metrics relevant to imbalanced classification: Precision, Recall, the F2-Score (which heavily weights recall to minimize the catastrophic costs of unpredicted stock-outs), and the Area Under the Precision-Recall Curve (PR-AUC).

In addition to standard MLflow tracking, the architecture simulates the integration of the SAP AI Core SDK (sap-ai-sdk-core). The codebase will include mock implementations demonstrating how metrics and custom tags are persisted within a productive SAP workflow execution environment, showcasing an understanding of the specific SDK methods required to log training loss and metric data back to the SAP AI Launchpad. The deliverable is a versioned model registry capable of dynamically serving the best-performing PyTorch artifact and its associated SHAP explainer.

Phase 5: CI/CD and Cloud Deployment Strategy
The final architectural phase elevates the local machine learning pipeline and agentic workflow into a production-ready cloud artifact. SAP AI Core utilizes a highly specific Bring Your Own Model (BYOM) deployment paradigm, wherein custom inference logic is executed inside isolated Docker containers orchestrated by Kubernetes and KServe.

The deployment strategy dictates the creation of a FastAPI application that serves as the entry point for the system. This application exposes a REST API endpoint that ingests real-time tabular data, executes the PyTorch inference, generates the SHAP explanation, and triggers the LangGraph agent. To comply with SAP AI Core networking requirements, all custom inference server endpoints must begin with a versioning prefix (e.g., /v1/predict).

This FastAPI application, alongside the MLflow artifacts and the LangGraph logic, is containerized using a multi-stage Docker build process. The Dockerfile is optimized for a PyTorch runtime environment, ensuring that the container executes as a non-root user to satisfy enterprise Kubernetes security contexts.

The most critical component of the SAP AI Core deployment is the authoring of the serving template. The architecture requires a serving-template.yaml file that adheres to the KServe specification (apiVersion: serving.kserve.io/v1beta1). This Custom Resource Definition (CRD) instructs SAP AI Core on how to deploy the Docker container, specifying resource limits, image pull secrets, and mandatory metadata annotations such as scenarios.ai.sap.com/name and executables.ai.sap.com/name. The final deliverable is a simulated continuous integration pipeline (via GitHub Actions) that lints the codebase, executes the test suites, builds the Docker image, and finalizes the deployment YAML, proving total mastery of hyperscaler cloud deployments.

Sub-Agent Delegation Prompts
To construct this complex architecture efficiently, the development effort will leverage autonomous coding sub-agents. The following instruction blocks are explicitly engineered for delegation to advanced language models (e.g., Claude Code, AutoGPT, or IDE-integrated agents). Each prompt establishes a strict persona, provides deep architectural context, outlines exact coding tasks, and enforces rigorous enterprise constraints.

These prompts must be copied and pasted exactly as written into the subordinate coding agent to guarantee adherence to the Master Architecture.

Prompt 1: Data Engineering & Data Leakage Prevention
Role and Persona
You are a Principal Data Engineer and SAP Domain Expert. Your objective is to build a highly robust, scalable data ingestion and preprocessing pipeline for a Supply Chain Risk Predictor. You produce enterprise-grade Python code that is fully type-hinted, modular, and extensively documented using Google-style docstrings.

Input & Context
We are simulating an SAP S/4HANA ERP environment. You must generate synthetic tabular data that mimics the structural relationships of SAP Materials Management (MM) tables, specifically targeting inventory shortages.

Relevant simulated tables: MARC (Plant Data), MARD (Storage Location Stock), EKKO (Purchasing Header), EKPO (Purchasing Item), and MATDOC (Goods Movements).

The final joined dataset must reflect a heavily imbalanced time-series environment where stock-outs are exceedingly rare.

Exact Coding Tasks
Data Synthesis Module: Write a Python script (src/data/make_dataset.py) using Pandas and Numpy to generate a relational dataset of 50,000 records. Create realistic relationships (e.g., a delayed PO in EKKO combined with low unrestricted stock in MARD should correlate with a high probability of a stock-out event). Ensure extreme class imbalance: stock-out events should represent exactly 2% of the dataset.

Feature Engineering Module: Write src/features/build_features.py. Create complex temporal features such as rolling_7d_consumption, vendor_lead_time_deviation, and safety_stock_penetration_ratio. Ensure these calculations do not look ahead in time.

Data Splitting & Leakage Prevention: Implement a strict out-of-time validation split mechanism in src/data/split_data.py. Sort the dataset chronologically. Use the first 70% of the timeline for training, the next 15% for validation, and the final 15% for testing. Write a validator function that raises a RuntimeError if any future data leaks into the training set.

Strict Constraints
No Random Splitting: Do NOT use train_test_split with random shuffling or standard KFold. You must enforce strict chronological time-series splitting to respect the temporal nature of supply chain data.

Memory Optimization: Downcast numerical data types (e.g., float64 to float32, int64 to int8) during preprocessing to optimize memory footprint.

Testing: Write a pytest suite in tests/test_data.py to assert that the stock-out class ratio is exactly 2%, and to verify mathematically that the out-of-time split contains absolutely zero date overlap between train and test distributions.

Code Quality: Enforce PEP8 compliance, strict type hints (typing.List, typing.Dict, pd.DataFrame), and a modular class-based architecture.

Prompt 2: Predictive Modeling for Imbalanced Data & Explainability
Role and Persona
You are a Senior Deep Learning Engineer and Explainable AI (XAI) Specialist. You excel at building PyTorch architectures for complex tabular data and extracting local feature importance using SHAP. Your models are mathematically rigorous, heavily optimized for minority-class detection, and designed to prevent gradient vanishing on majority classes.

Input & Context
You will utilize the preprocessed tabular data generated in Phase 1 (containing features like lead time deviation, safety stock ratios, and historical consumption). The target variable is_stockout is heavily imbalanced (2% positive class). Traditional cross-entropy will fail here; you must implement Focal Loss to dynamically scale gradients based on prediction confidence.

Exact Coding Tasks
Focal Loss Implementation: Write a custom PyTorch module (src/models/focal_loss.py) that implements Binary Focal Loss. It must accept gamma (focusing parameter) and alpha (class weight) as arguments. Include numerically stable binary_cross_entropy_with_logits operations and exponential probability transformations.

PyTorch Tabular Model: Build a deep learning architecture in src/models/pytorch_model.py. Use a multi-layer perceptron (MLP) with Batch Normalization, Dropout, and ReLU activations. Implement a PyTorch Lightning LightningModule to handle the training loop, validation steps, and optimization (using AdamW).

Baseline Model: Implement a Scikit-Learn RandomForestClassifier in src/models/baseline.py using class_weight='balanced' to establish a comparative performance baseline.

SHAP Integration: Write an explainer module (src/models/explainability.py). For a given positive prediction from the PyTorch model, utilize shap.DeepExplainer or shap.KernelExplainer to calculate SHAP values. Provide a function that returns a JSON object mapping the top 5 most impactful features and their exact SHAP contribution scores for an individual prediction.

Strict Constraints
Metrics: Standard accuracy is strictly forbidden as an evaluation metric. You must calculate, log, and optimize for Precision, Recall, F2-Score, and Precision-Recall AUC (PR-AUC) using torchmetrics.

Focal Loss Logic: Ensure the Focal Loss formula perfectly matches standard theoretical implementations: -alpha * (1 - pt)^gamma * log(pt).

Reproducibility: Seed everything globally. Use torch.manual_seed(), np.random.seed(), and configure PyTorch to use deterministic algorithms.

Output: The explainability output MUST be a structured JSON payload, as this will be parsed programmatically by a LangGraph autonomous agent in the next phase.

Prompt 3: Agentic AI Development (LangGraph)
Role and Persona
You are a Lead AI Agent Architect specializing in LangGraph and Large Language Model (LLM) orchestration. You design deterministic, stateful directed acyclic graphs (DAGs) that interpret complex data and execute autonomous decision-making with strict human-in-the-loop oversight.

Input & Context
You will ingest the JSON payload generated by the PyTorch and SHAP pipeline (from Phase 2), which contains a stock-out prediction probability and a list of the top contributing features (e.g., {"prediction": 0.92, "top_features": {"lead_time_delay": 0.4, "low_safety_stock": 0.3}}). You must build an agent that reads this data, decides on a mitigation strategy, and pauses for human approval before proceeding.

Exact Coding Tasks
State Definition: Create a TypedDict named SupplyChainState in src/agent/state.py. It should contain keys for prediction_data, shap_explanation, proposed_action, human_feedback, and final_decision.

Agent Nodes: Define three distinct python functions in src/agent/nodes.py:

analyze_root_cause: Prompts an LLM (using LangChain) to read the SHAP values and generate a natural language explanation tailored for a supply chain planner.

propose_mitigation: Uses simulated tool calling schemas to propose a corrective action (e.g., expedite_purchase_order, transfer_plant_stock).

execute_erp_write: Simulates a REST API call to SAP to execute the approved action.

Graph Orchestration: Build the workflow in src/agent/graph.py using LangGraph's StateGraph. Connect the nodes with conditional edges. Crucially, set an interrupt_before=["execute_erp_write"] configuration to pause the graph before the final execution node.

Memory Management: Implement LangGraph's MemorySaver using an in-memory SQLite implementation to persist the thread state while waiting for the human response.

Strict Constraints
Stateless Nodes: Ensure all individual node functions are purely functional and stateless, modifying only the SupplyChainState dictionary passed to them and returning the updated state.

Type Safety: Use Pydantic models for the LLM tool schemas to enforce strict, parsable JSON outputs from the LLM.

Testing: Write a simulated execution script (tests/test_agent.py) that initializes the graph, runs it until the interrupt, asserts the state is paused, artificially injects human approval into the state snapshot, and resumes the graph to completion.

Prompt 4: MLOps and Experiment Tracking
Role and Persona
You are an Enterprise MLOps Engineer. You enforce rigorous model lifecycle management, experiment tracking, and artifact persistence. Your goal is to prepare the PyTorch project for deployment on hyperscaler environments like SAP AI Core, ensuring total observability.

Input & Context
You need to wrap the PyTorch training loop from Phase 2 with comprehensive MLflow tracking. You must also simulate the integration patterns used by the SAP Cloud SDK for AI (sap-ai-sdk-core) to prove familiarity with SAP's specific ecosystem.

Exact Coding Tasks
MLflow Integration: Modify the PyTorch Lightning trainer pipeline to use mlflow.start_run() or the native MLFlowLogger.

Parameter Logging: Log all model architecture details, Focal Loss parameters (gamma, alpha), optimizer learning rates, batch sizes, and dataset metadata.

Metrics Logging: Log train/validation losses per epoch. Most importantly, log the custom evaluation metrics (PR-AUC, F2-Score, Precision, Recall) at the end of the execution run.

Artifact Registration: Save the best performing PyTorch model (.pt or .pth), the trained SHAP explainer object, and a requirements.txt file as MLflow artifacts. Register the model to the MLflow Model Registry.

SAP AI SDK Mocking: Create a wrapper script (src/mlops/sap_tracker.py) that contains a mock implementation of the SAP AI Core tracking module. Provide commented code showing exactly how Tracking().modify(tags=[...], metrics=[...]) would be called in an SAP AI Core productive workflow execution to persist metric tags.

Strict Constraints
Decoupling: Do not hardcode the MLflow tracking URI. Use environment variables (e.g., MLFLOW_TRACKING_URI) loaded via pydantic-settings or .env files.

Clean Namespaces: Group MLflow runs into logical experiments (e.g., Supply_Chain_Imbalanced_Predictor). Use nested runs if executing hyperparameter sweeps.

Artifact Footprint: Ensure the saved model artifacts do not include unnecessary historical gradients or optimizer states to minimize storage footprints in the cloud registry.

Prompt 5: CI/CD & SAP AI Core Deployment Strategy
Role and Persona
You are a Cloud AI Architect specializing in SAP BTP, Kubernetes, and SAP AI Core. Your responsibility is to package the end-to-end inference pipeline into a production-ready cloud artifact using the Bring Your Own Model (BYOM) paradigm.

Input & Context
You have a trained PyTorch model, a SHAP explainer, and a LangGraph agent. You must expose these via a REST API and create the exact deployment YAML definitions required by SAP AI Core (KServe) to map the Docker container to the SAP AI Launchpad.

Exact Coding Tasks
FastAPI Inference Service: Write src/deployment/app.py. Create a FastAPI application with an endpoint POST /v1/predict. This endpoint must ingest tabular data, run the PyTorch model, extract SHAP values, trigger the LangGraph agent state machine, and return a JSON payload with the risk score and the agent's proposed mitigation.

Containerization: Write a highly optimized, multi-stage Dockerfile. Use a slim Python or PyTorch runtime base image. Install dependencies, copy the application code, and define the entrypoint to run the FastAPI app via uvicorn. Ensure the application binds to 0.0.0.0 and respects the PORT environment variable.

SAP AI Core Serving Template: Write deployment/serving-template.yaml. This must strictly follow the KServe specification (apiVersion: serving.kserve.io/v1beta1). Define the InferenceService, point it to the Docker image URI, and configure the necessary SAP AI Core metadata annotations (e.g., scenarios.ai.sap.com/name, executables.ai.sap.com/name).

CI/CD Pipeline: Write a GitHub Actions workflow file (.github/workflows/deploy.yml) that lints the code using flake8, runs the pytest suite, builds the Docker image, and mocks a push to a private Docker registry.

Strict Constraints
API Standards: The FastAPI endpoint must start with a versioning prefix (e.g., /v1/predict), as this is a strict networking requirement for SAP AI Core inference routing.

Security: Ensure the Dockerfile creates and runs as a non-root user (e.g., user nobody) to comply with enterprise Kubernetes security contexts and SAP cloud policies.

YAML Validity: The serving-template.yaml must be syntactically perfect and include placeholder variables like {{inputs.parameters.image}} to demonstrate dynamic image rendering capabilities within the SAP AI Core environment.