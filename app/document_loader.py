import os
import logging
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from app.config import settings

logger = logging.getLogger(__name__)

SAMPLE_DOCUMENTS = [
    (
        "intro_to_ml",
        """Introduction to Machine Learning

Machine learning is a branch of artificial intelligence that enables computers to learn from data
without being explicitly programmed for every task. Instead of following hard-coded rules,
machine learning algorithms build mathematical models from training data to make predictions
or decisions.

The field was formally introduced by Arthur Samuel in 1959, who defined it as a "field of study
that gives computers the ability to learn without being explicitly programmed." Today, machine
learning powers applications from email spam filters to self-driving cars and medical diagnosis.

There are three primary paradigms: supervised learning (learning from labeled examples),
unsupervised learning (discovering patterns in unlabeled data), and reinforcement learning
(learning through interaction with an environment to maximize reward). Each paradigm is suited
to different types of problems and data availability.

Key concepts in machine learning include features (input variables), labels (output variables),
training datasets, test datasets, overfitting (when a model memorizes training data but fails
to generalize), and underfitting (when a model is too simple to capture underlying patterns).
Model evaluation typically uses metrics like accuracy, precision, recall, F1-score, and
area under the ROC curve depending on the task type.
""",
    ),
    (
        "neural_networks",
        """Neural Networks and Deep Learning

Neural networks are computing systems inspired by the biological neural networks in animal brains.
They consist of layers of interconnected nodes (neurons) that process information using
connectionist approaches to computation. A basic neural network has an input layer, one or more
hidden layers, and an output layer.

Deep learning refers to neural networks with many hidden layers (hence "deep"). These deep
architectures can automatically learn hierarchical representations of data. Convolutional Neural
Networks (CNNs) excel at image recognition by learning spatial hierarchies of features.
Recurrent Neural Networks (RNNs) handle sequential data like text and time series.

Training neural networks involves forward propagation (passing data through the network to get
predictions), calculating a loss function (measuring prediction error), and backpropagation
(computing gradients and updating weights using gradient descent or variants like Adam, RMSprop).

Key challenges include the vanishing gradient problem (gradients becoming too small in deep
networks), computational cost, and the need for large labeled datasets. Techniques like batch
normalization, dropout regularization, and residual connections (skip connections) help address
these challenges and enable training of very deep networks.
""",
    ),
    (
        "transformers",
        """The Transformer Architecture

The Transformer architecture, introduced in the 2017 paper "Attention Is All You Need" by Vaswani
et al., revolutionized natural language processing and became the foundation for modern large
language models. Unlike previous sequential models, transformers process all tokens in parallel
using self-attention mechanisms.

The core innovation is the attention mechanism, which allows the model to weigh the importance of
different parts of the input when producing each part of the output. Self-attention computes
attention scores between all pairs of tokens in a sequence, capturing long-range dependencies
without the vanishing gradient problems of RNNs.

A transformer consists of an encoder (which processes the input) and a decoder (which generates
the output). Each encoder/decoder block contains a multi-head attention layer (multiple attention
heads running in parallel) and a position-wise feed-forward network, with residual connections
and layer normalization throughout.

Positional encodings are added to token embeddings since transformers have no inherent notion of
position. The multi-head attention allows the model to attend to information from different
representation subspaces simultaneously, capturing different types of relationships in the data.
BERT uses only the encoder, GPT uses only the decoder, while T5 uses the full encoder-decoder.
""",
    ),
    (
        "large_language_models",
        """Large Language Models

Large Language Models (LLMs) are transformer-based neural networks trained on massive text
corpora with billions to trillions of parameters. They exhibit emergent capabilities—abilities
not present in smaller models—including in-context learning, chain-of-thought reasoning,
and instruction following.

Prominent LLMs include GPT-4 (OpenAI), Claude (Anthropic), Gemini (Google), and Llama (Meta).
These models are pre-trained on diverse internet text using next-token prediction as the training
objective. Pre-training is followed by fine-tuning on instruction-following datasets and
alignment techniques like RLHF (Reinforcement Learning from Human Feedback) to make them helpful,
harmless, and honest.

Scaling laws describe how model performance improves predictably with more parameters, more data,
and more compute. The Chinchilla scaling laws suggest that for a given compute budget, training
a smaller model on more data is more efficient than training a larger model on less data.

Key capabilities include text generation, summarization, translation, question answering, code
generation, and reasoning. Limitations include hallucination (generating plausible but false
information), knowledge cutoff dates, context window limitations, and sensitivity to prompt
phrasing. Techniques like retrieval augmentation and tool use help address some limitations.
""",
    ),
    (
        "rag_systems",
        """Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) is an AI architecture that enhances large language models
by grounding their responses in retrieved external knowledge. Instead of relying solely on
parametric knowledge learned during training, RAG systems dynamically retrieve relevant documents
at inference time and include them in the context provided to the LLM.

The RAG pipeline has two main components: a retrieval system and a generation system. The
retrieval system encodes documents into dense vector embeddings, stores them in a vector database,
and performs similarity search to find relevant documents for a given query. The generation
system (an LLM) then synthesizes an answer using both the query and the retrieved documents.

RAG addresses several limitations of standalone LLMs: it reduces hallucination by anchoring
responses to retrieved facts, enables knowledge updates without retraining (just update the
document store), supports domain-specific applications, and allows citation of sources.

Advanced RAG variants include HyDE (Hypothetical Document Embeddings), which generates a
hypothetical answer to improve retrieval; re-ranking, which applies a cross-encoder to reorder
retrieved results; and agentic RAG, which uses tool use and multi-step reasoning. Evaluation
metrics like faithfulness, answer relevancy, context precision, and context recall (from the
Ragas framework) quantify RAG system quality.
""",
    ),
    (
        "vector_embeddings",
        """Vector Embeddings in Machine Learning

Vector embeddings are dense, low-dimensional numerical representations of high-dimensional data
(text, images, audio). They encode semantic meaning in a continuous vector space where similar
items are geometrically close together. Word2Vec, GloVe, and FastText were early word embedding
models; modern sentence and document embeddings from models like BERT, Sentence-BERT, and
text-embedding-ada-002 capture richer contextual semantics.

For text, embedding models map sentences or paragraphs to fixed-length vectors (typically 384
to 1536 dimensions). These embeddings capture semantic relationships: "king - man + woman ≈ queen"
in the embedding space. Cosine similarity and dot product are common distance metrics for
comparing embeddings.

Embeddings are fundamental to modern NLP pipelines. They power semantic search (finding
semantically similar documents rather than exact keyword matches), recommendation systems,
clustering, classification, and retrieval-augmented generation. Fine-tuning embedding models on
domain-specific data can significantly improve performance for specialized applications.

Google's text-embedding-gecko and embedding-001 models, OpenAI's text-embedding-ada-002, and
open-source models like all-MiniLM-L6-v2 represent state-of-the-art embedding options. Matryoshka
Representation Learning (MRL) creates embeddings that maintain quality even when truncated,
enabling flexible trade-offs between storage, speed, and quality.
""",
    ),
    (
        "vector_databases",
        """Vector Databases

Vector databases are specialized storage systems optimized for storing, indexing, and querying
high-dimensional vector embeddings. Unlike traditional databases that excel at exact lookups,
vector databases support approximate nearest neighbor (ANN) search—finding vectors most similar
to a query vector efficiently at scale.

Popular vector databases include Pinecone, Weaviate, Qdrant, Milvus, and ChromaDB. ChromaDB is
an open-source, embedded vector database that supports local persistence and is popular for
development and smaller deployments. It uses HNSW (Hierarchical Navigable Small World) indexing
for fast ANN search.

Key features of vector databases: efficient similarity search with sub-millisecond latency at
millions of vectors, metadata filtering (combining vector search with attribute filters),
CRUD operations on vectors, scalability, and integrations with ML frameworks. HNSW, IVF
(Inverted File Index), and PQ (Product Quantization) are common indexing strategies with
different trade-offs between speed, accuracy, and memory.

Vector databases are the backbone of RAG systems, semantic search engines, recommendation systems,
and image/audio search applications. When choosing a vector database, consider: query latency
requirements, scale (number of vectors), filtering needs, deployment constraints (cloud vs.
on-premise), and integration with your tech stack.
""",
    ),
    (
        "nlp_fundamentals",
        """Natural Language Processing

Natural Language Processing (NLP) is a subfield of AI focused on enabling computers to understand,
interpret, and generate human language. It bridges computational linguistics and machine learning
to process and analyze large amounts of natural language data.

Core NLP tasks include tokenization (splitting text into words/subwords), part-of-speech tagging,
named entity recognition (identifying persons, organizations, locations), sentiment analysis,
machine translation, text summarization, and question answering. Modern NLP uses transformer-based
models that have achieved superhuman performance on many benchmarks.

The NLP pipeline typically involves preprocessing (cleaning, tokenization, stopword removal),
feature extraction (TF-IDF, word embeddings, or contextual embeddings from BERT/GPT), model
training or fine-tuning, and post-processing. Subword tokenization methods like Byte Pair
Encoding (BPE) and WordPiece handle out-of-vocabulary words and multilingual text effectively.

Evaluation metrics vary by task: BLEU and ROUGE for generation tasks, F1-score for classification
and span extraction, perplexity for language modeling. The GLUE and SuperGLUE benchmarks
standardized NLP evaluation. Recent work focuses on few-shot learning, multilingual models,
and improving robustness and fairness of NLP systems.
""",
    ),
    (
        "computer_vision",
        """Computer Vision

Computer vision is an AI field that enables machines to derive meaningful information from images,
videos, and other visual inputs. It encompasses image classification, object detection, image
segmentation, optical character recognition, and video understanding.

Convolutional Neural Networks (CNNs) dominated computer vision from 2012 with AlexNet's ImageNet
victory through architectures like VGG, ResNet, and EfficientNet. Vision Transformers (ViT)
introduced in 2020 apply the transformer architecture directly to image patches, achieving
state-of-the-art results on many benchmarks and enabling unified vision-language models.

Key techniques include: data augmentation (random cropping, flipping, color jitter) to prevent
overfitting, transfer learning from ImageNet pre-trained models, attention mechanisms for
focusing on relevant image regions, and self-supervised learning (contrastive methods like
CLIP, SimCLR) to learn visual representations without labeled data.

Applications span autonomous driving (detecting pedestrians, signs, lane markings), medical
imaging (detecting tumors, analyzing X-rays), retail (product recognition), manufacturing
(defect detection), and augmented reality. Multimodal models like GPT-4V, Gemini, and DALL-E
combine vision and language understanding, enabling image captioning, visual question answering,
and text-to-image generation.
""",
    ),
    (
        "reinforcement_learning",
        """Reinforcement Learning

Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make
decisions by interacting with an environment to maximize cumulative reward. Unlike supervised
learning, RL learns from trial and error without labeled examples, discovering strategies
through exploration and exploitation.

The RL framework involves: an agent (the learner/decision-maker), an environment (what the agent
interacts with), states (representations of the environment), actions (choices available to the
agent), and rewards (scalar feedback signals). The agent's goal is to learn a policy—a mapping
from states to actions—that maximizes expected cumulative reward.

Key algorithms include Q-Learning (tabular), Deep Q-Networks (DQN, which combines Q-learning
with deep neural networks), Policy Gradient methods (REINFORCE, PPO, A3C), and Actor-Critic
methods that combine value and policy learning. Model-based RL additionally learns a model of
the environment for planning.

RL has achieved superhuman performance in games (Atari, Chess, Go, StarCraft II), robot
locomotion and manipulation, and resource optimization. RLHF (Reinforcement Learning from
Human Feedback) is central to aligning LLMs with human preferences. Challenges include sample
efficiency, reward shaping, sparse rewards, and sim-to-real transfer for robotics.
""",
    ),
    (
        "supervised_learning",
        """Supervised Learning

Supervised learning is the most common machine learning paradigm, where a model learns a mapping
from inputs to outputs using labeled training examples. The algorithm learns by comparing its
predictions to the true labels and adjusting its parameters to minimize prediction error.

Classification problems predict discrete categories (spam/not spam, disease/no disease, digit 0-9),
while regression problems predict continuous values (house price, temperature, stock return).
Common algorithms include linear/logistic regression, decision trees, random forests, gradient
boosting (XGBoost, LightGBM), support vector machines, k-nearest neighbors, and neural networks.

The training pipeline: split data into train/validation/test sets, preprocess features (scaling,
encoding categoricals, handling missing values), select and train a model, tune hyperparameters
on validation set (grid search, random search, Bayesian optimization), and evaluate final
performance on held-out test set. Cross-validation provides more robust performance estimates.

Key challenges include class imbalance (use oversampling, undersampling, or class weights),
high dimensionality (use feature selection or dimensionality reduction), label noise, and
distribution shift between train and production data. Feature engineering—creating informative
input features—often has more impact on performance than algorithm choice.
""",
    ),
    (
        "unsupervised_learning",
        """Unsupervised Learning

Unsupervised learning discovers hidden patterns in data without labeled examples. Instead of
learning a predefined mapping, unsupervised algorithms find inherent structure—clusters,
low-dimensional representations, or generative factors—in the input data.

Clustering algorithms group similar data points: K-Means partitions data into K clusters
minimizing within-cluster variance; DBSCAN finds density-based clusters of arbitrary shape;
hierarchical clustering builds a tree of nested clusters. Gaussian Mixture Models take a
probabilistic approach, modeling data as a mixture of Gaussian distributions.

Dimensionality reduction compresses high-dimensional data to fewer dimensions while preserving
structure. PCA (Principal Component Analysis) finds orthogonal directions of maximum variance.
t-SNE and UMAP are non-linear methods that preserve local neighborhood structure, excellent
for visualization. Autoencoders use neural networks to learn compressed representations.

Generative models learn the underlying data distribution to generate new samples. Variational
Autoencoders (VAEs) learn a structured latent space. Generative Adversarial Networks (GANs)
train a generator and discriminator in adversarial fashion. Diffusion models iteratively denoise
random noise to generate high-quality images (Stable Diffusion, DALL-E 3). Applications include
anomaly detection, customer segmentation, recommendation, and data augmentation.
""",
    ),
    (
        "transfer_learning",
        """Transfer Learning

Transfer learning is a machine learning technique where a model trained on one task is adapted
(fine-tuned) for a different but related task. Rather than training from scratch, transfer
learning leverages knowledge already encoded in pre-trained model weights, dramatically reducing
data and compute requirements for new tasks.

The pre-train/fine-tune paradigm dominates modern deep learning. A model is first pre-trained
on a large general dataset (ImageNet for vision, large text corpora for NLP), learning broadly
useful features. It is then fine-tuned on a smaller task-specific dataset by continuing training
with a lower learning rate, sometimes freezing early layers and only updating later ones.

Types of transfer learning: feature extraction (use pre-trained model as fixed feature extractor),
fine-tuning (update all or part of pre-trained weights), and domain adaptation (adapt to a
different data distribution). Zero-shot and few-shot transfer push this further—large foundation
models like GPT-4 and CLIP can generalize to unseen tasks with no or minimal examples.

BERT demonstrated that pre-training on masked language modeling and next sentence prediction,
then fine-tuning on downstream tasks, achieves state-of-the-art NLP results. Parameter-Efficient
Fine-Tuning (PEFT) methods like LoRA (Low-Rank Adaptation) and prefix tuning update only a
small fraction of parameters while achieving performance comparable to full fine-tuning,
making adaptation of billion-parameter models practical.
""",
    ),
    (
        "generative_ai",
        """Generative AI

Generative AI refers to AI systems that can create new content—text, images, audio, video, code,
or 3D models—that resembles human-created content. These systems learn the statistical patterns
and structure of training data to generate novel, coherent outputs.

The generative AI landscape includes: Large Language Models (GPT-4, Claude, Gemini) for text
generation; image generation models (DALL-E 3, Midjourney, Stable Diffusion); code generation
tools (GitHub Copilot, CodeLlama); audio synthesis models (ElevenLabs, MusicLM); and video
generation (Sora, Gen-2). Multimodal models handle multiple modalities simultaneously.

Foundation models are large pre-trained models that serve as the base for many downstream
applications. They exhibit emergent capabilities—abilities not explicitly trained for—that
appear at scale. In-context learning allows foundation models to perform new tasks given just
a few examples in the prompt, without any weight updates.

The generative AI ecosystem includes: model providers (Anthropic, OpenAI, Google, Meta),
inference infrastructure (API providers, local inference via llama.cpp), orchestration frameworks
(LangChain, LlamaIndex), vector databases, and evaluation tools. Key challenges include
hallucination, bias, safety (preventing harmful outputs), intellectual property concerns,
and the environmental cost of training and inference at scale.
""",
    ),
    (
        "ai_evaluation",
        """AI Evaluation Metrics and Ragas

Evaluating AI systems, especially generative AI and RAG systems, requires specialized metrics
beyond traditional accuracy measures. Ragas (RAG Assessment) is a framework specifically designed
to evaluate RAG pipeline quality across multiple dimensions.

Faithfulness measures whether the generated answer is factually consistent with the retrieved
context. A high faithfulness score means the model only makes claims that are supported by the
provided documents, not hallucinating. It is computed by decomposing the answer into statements
and checking each against the context.

Answer Relevancy measures how pertinent the generated answer is to the given question. It
generates multiple questions from the answer and measures their similarity to the original
question—answers that directly address the question score higher than vague or tangential ones.

Context Precision measures whether the retrieved context ranks the most relevant information
higher. It evaluates whether relevant chunks appear at the top of the retrieved results.
Context Recall measures the proportion of the ground-truth answer that can be attributed to
the retrieved context—high recall means the retrieved documents contain all the information
needed to answer the question correctly.

Beyond Ragas, LLM evaluation includes: BLEU and ROUGE for summarization and translation,
BERTScore for semantic similarity, G-Eval (LLM-as-judge), MT-Bench for instruction following,
and human evaluation as the gold standard. Continuous evaluation in production (monitoring
answer quality, detecting drift, A/B testing) is essential for maintaining RAG system quality.
""",
    ),
    (
        "bias_and_fairness",
        """Bias and Fairness in AI

AI bias refers to systematic errors in model outputs that create unfair outcomes for certain
groups. Bias can enter at multiple stages: biased training data (historical discrimination
encoded in data), biased labels (subjective human annotation), biased metrics (optimizing
overall accuracy can mask poor performance on minorities), and biased deployment.

Types of bias include: representation bias (training data under-represents certain groups),
measurement bias (features proxy for protected attributes), aggregation bias (one model for
heterogeneous subgroups), and evaluation bias (benchmarks that don't represent real-world
diversity). Intersectionality—compounding bias across multiple protected attributes—often
produces the most severe disparities.

Fairness definitions include demographic parity (equal prediction rates across groups),
equalized odds (equal true/false positive rates), individual fairness (similar individuals
treated similarly), and counterfactual fairness (same outcome if protected attribute changed).
These definitions are often mathematically incompatible, requiring trade-off decisions.

Mitigation techniques: pre-processing (resampling, reweighting training data), in-processing
(fairness constraints in training objective), and post-processing (calibrating outputs by group).
Fairness auditing tools like IBM AI Fairness 360, Microsoft Fairlearn, and Google's What-If Tool
help practitioners identify and measure bias. Regulatory frameworks like the EU AI Act and
US Executive Order on AI increasingly require fairness assessments for high-risk AI applications.
""",
    ),
    (
        "prompt_engineering",
        """Prompt Engineering

Prompt engineering is the practice of designing and optimizing input prompts to elicit desired
outputs from large language models. As LLMs are prompt-sensitive, carefully crafted prompts
can dramatically improve output quality, accuracy, and reliability without any model fine-tuning.

Key techniques include: zero-shot prompting (giving the task directly without examples),
few-shot prompting (providing 2-10 examples of the desired input-output format), chain-of-thought
(CoT) prompting (asking the model to reason step-by-step before answering), and self-consistency
(sampling multiple reasoning paths and majority voting). Structured output prompting uses formats
like JSON to get machine-parseable responses.

System prompts define the model's role, persona, and behavioral guidelines. Role prompting
("You are an expert Python developer") improves performance on domain-specific tasks.
Constitutional AI prompts instruct models to critique and revise their own outputs for safety
and helpfulness. ReAct (Reasoning + Acting) combines chain-of-thought with tool use calls.

Advanced techniques: Tree of Thoughts explores multiple reasoning branches, Active Prompting
identifies uncertain examples for human annotation, Automatic Prompt Optimization uses
meta-prompting to improve prompts. Prompt injection attacks—where user input overrides system
instructions—are a key security concern. Best practices include clear, specific instructions,
providing context and constraints, using delimiters to separate sections, and iterative testing.
""",
    ),
    (
        "fine_tuning_llms",
        """Fine-Tuning Large Language Models

Fine-tuning adapts a pre-trained LLM to specific domains, tasks, or behavior patterns by
continuing training on a curated dataset. While foundation models have broad capabilities,
fine-tuning improves performance on specialized tasks, enforces specific output formats, and
encodes domain-specific knowledge not present in pre-training data.

Full fine-tuning updates all model parameters but requires significant GPU memory and compute.
For a 7B parameter model, full fine-tuning needs multiple A100 GPUs and hundreds of GBs of
VRAM. Parameter-Efficient Fine-Tuning (PEFT) methods address this by updating only a small
fraction of parameters: LoRA (Low-Rank Adaptation) injects trainable low-rank matrices into
attention layers; QLoRA combines LoRA with 4-bit quantization to fine-tune 65B models on a
single 48GB GPU.

Instruction tuning trains on (instruction, response) pairs to improve instruction following.
RLHF (Reinforcement Learning from Human Feedback) uses human preference data to align model
behavior with human values. DPO (Direct Preference Optimization) achieves similar alignment
without a separate reward model, making it simpler to implement.

Fine-tuning dataset quality matters more than quantity. Curating 1,000 high-quality examples
often outperforms 100,000 noisy ones. Techniques like data deduplication, quality filtering,
and synthetic data generation (using a stronger model to generate training examples) are critical.
Evaluation after fine-tuning should use held-out test sets and automated benchmarks, checking
for catastrophic forgetting of general capabilities.
""",
    ),
    (
        "ai_ethics",
        """AI Ethics and Responsible AI

AI ethics encompasses the principles, guidelines, and practices for developing and deploying
AI systems that are safe, fair, transparent, accountable, and beneficial. As AI systems become
more powerful and pervasive, ensuring they align with human values and do not cause harm is
increasingly critical.

Core ethical principles: beneficence (AI should benefit humanity), non-maleficence (avoid harm),
autonomy (respect human agency and decision-making), justice (fair distribution of AI benefits
and risks), and explicability (AI decisions should be understandable and justifiable). These
principles are instantiated in frameworks like the EU AI Act, IEEE Ethically Aligned Design,
and company AI principles from Google, Microsoft, and Anthropic.

AI safety research addresses risks from advanced AI systems: alignment (ensuring AI pursues
intended goals), interpretability (understanding what AI systems learn and why they make
decisions), robustness (maintaining safe behavior under distribution shift and adversarial attacks),
and long-term risks from potentially transformative AI systems.

Practical responsible AI practices: model cards (documenting model purpose, limitations, and
evaluation results), datasheets for datasets, impact assessments before deployment, red-teaming
(adversarial testing for harmful outputs), human oversight for high-stakes decisions, and
incident response processes. Organizations like the Alignment Research Center, Center for AI
Safety, and Partnership on AI advance safety research and policy.
""",
    ),
    (
        "langchain_framework",
        """The LangChain Framework

LangChain is an open-source framework for building applications powered by large language models.
It provides standardized abstractions and composable components for common LLM application
patterns including conversational agents, document Q&A, summarization pipelines, and autonomous
agents.

Core LangChain concepts: Chains (sequences of LLM calls and other operations), Agents (LLMs
that dynamically choose actions using tools), Memory (maintaining conversation context),
Retrievers (fetching relevant documents from a vector store), and Document Loaders (ingesting
data from files, APIs, databases). LCEL (LangChain Expression Language) provides a declarative
syntax for composing these components into pipelines.

LangChain integrates with major LLM providers (OpenAI, Anthropic, Google, Hugging Face),
vector stores (ChromaDB, Pinecone, Weaviate, FAISS), document loaders (PDF, CSV, web scraping),
and tools (search engines, code execution, APIs). LangSmith provides observability—tracing,
logging, and evaluation—for LangChain applications in production.

The framework has evolved significantly: LangChain 0.1.x unified the package structure,
0.2.x improved stability and LCEL, and 0.3.x deprecated legacy chains in favor of LCEL-based
alternatives. For RAG applications, LangChain's RetrievalQA chain and ConversationalRetrievalChain
are widely used. The ecosystem includes LangGraph for stateful multi-agent workflows and
LangServe for deploying chains as REST APIs.
""",
    ),
]

DEFAULT_QA_PAIRS = [
    {
        "question": "What is machine learning?",
        "answer": "Machine learning is a branch of artificial intelligence that enables computers to learn from data without being explicitly programmed. It builds mathematical models from training data to make predictions or decisions, with three primary paradigms: supervised learning, unsupervised learning, and reinforcement learning.",
        "contexts": [
            "Machine learning is a branch of artificial intelligence that enables computers to learn from data without being explicitly programmed for every task.",
            "There are three primary paradigms: supervised learning (learning from labeled examples), unsupervised learning (discovering patterns in unlabeled data), and reinforcement learning (learning through interaction with an environment to maximize reward).",
        ],
        "ground_truth": "Machine learning is a branch of AI that enables computers to learn from data without explicit programming, using three main paradigms: supervised, unsupervised, and reinforcement learning.",
    },
    {
        "question": "What is RAG and why is it useful?",
        "answer": "RAG (Retrieval-Augmented Generation) is an AI architecture that enhances large language models by retrieving relevant external documents at inference time. It reduces hallucination, enables knowledge updates without retraining, and supports domain-specific applications with source citations.",
        "contexts": [
            "Retrieval-Augmented Generation (RAG) is an AI architecture that enhances large language models by grounding their responses in retrieved external knowledge.",
            "RAG addresses several limitations of standalone LLMs: it reduces hallucination by anchoring responses to retrieved facts, enables knowledge updates without retraining, and allows citation of sources.",
        ],
        "ground_truth": "RAG is an AI architecture that combines retrieval of relevant documents with LLM generation, reducing hallucination and enabling knowledge updates without full model retraining.",
    },
    {
        "question": "How do transformers work?",
        "answer": "Transformers process all tokens in parallel using self-attention mechanisms that weigh the importance of different input parts. They consist of encoder and decoder blocks with multi-head attention layers, feed-forward networks, residual connections, and layer normalization.",
        "contexts": [
            "The Transformer architecture processes all tokens in parallel using self-attention mechanisms, unlike previous sequential models.",
            "The core innovation is the attention mechanism, which allows the model to weigh the importance of different parts of the input when producing each part of the output.",
        ],
        "ground_truth": "Transformers use self-attention mechanisms to process all input tokens in parallel, with encoder/decoder blocks containing multi-head attention and feed-forward layers.",
    },
    {
        "question": "What are vector embeddings?",
        "answer": "Vector embeddings are dense, low-dimensional numerical representations of high-dimensional data that encode semantic meaning. Similar items are geometrically close in the embedding space. They are used in semantic search, recommendation systems, and RAG pipelines.",
        "contexts": [
            "Vector embeddings are dense, low-dimensional numerical representations of high-dimensional data (text, images, audio). They encode semantic meaning in a continuous vector space where similar items are geometrically close together.",
            "Embeddings are fundamental to modern NLP pipelines. They power semantic search, recommendation systems, clustering, classification, and retrieval-augmented generation.",
        ],
        "ground_truth": "Vector embeddings are numerical representations of data that encode semantic meaning in a vector space, enabling semantic search and powering RAG systems.",
    },
    {
        "question": "What is fine-tuning in LLMs?",
        "answer": "Fine-tuning adapts a pre-trained LLM to specific domains or tasks by continuing training on curated data. Parameter-efficient methods like LoRA and QLoRA update only a fraction of parameters, making fine-tuning of large models practical on consumer hardware.",
        "contexts": [
            "Fine-tuning adapts a pre-trained LLM to specific domains, tasks, or behavior patterns by continuing training on a curated dataset.",
            "LoRA (Low-Rank Adaptation) injects trainable low-rank matrices into attention layers; QLoRA combines LoRA with 4-bit quantization to fine-tune 65B models on a single 48GB GPU.",
        ],
        "ground_truth": "Fine-tuning adapts pre-trained LLMs to specific tasks by additional training on curated datasets, with PEFT methods like LoRA making it computationally efficient.",
    },
    {
        "question": "What are the key Ragas evaluation metrics?",
        "answer": "Ragas evaluates RAG systems using four main metrics: faithfulness (answer consistency with retrieved context), answer relevancy (pertinence to the question), context precision (ranking of relevant information), and context recall (coverage of ground-truth information in retrieved context).",
        "contexts": [
            "Faithfulness measures whether the generated answer is factually consistent with the retrieved context.",
            "Answer Relevancy measures how pertinent the generated answer is to the given question.",
            "Context Precision measures whether the retrieved context ranks the most relevant information higher.",
            "Context Recall measures the proportion of the ground-truth answer that can be attributed to the retrieved context.",
        ],
        "ground_truth": "Ragas metrics include faithfulness, answer relevancy, context precision, and context recall—each measuring a different aspect of RAG pipeline quality.",
    },
    {
        "question": "What is prompt engineering?",
        "answer": "Prompt engineering is the practice of designing input prompts to elicit desired outputs from LLMs. Key techniques include zero-shot and few-shot prompting, chain-of-thought reasoning, and role prompting. It improves model performance without requiring fine-tuning.",
        "contexts": [
            "Prompt engineering is the practice of designing and optimizing input prompts to elicit desired outputs from large language models.",
            "Key techniques include: zero-shot prompting, few-shot prompting, chain-of-thought (CoT) prompting, and self-consistency.",
        ],
        "ground_truth": "Prompt engineering involves crafting input prompts to improve LLM outputs using techniques like few-shot examples and chain-of-thought reasoning.",
    },
    {
        "question": "What is transfer learning?",
        "answer": "Transfer learning uses a model trained on one task as a starting point for a different task. In NLP, models pre-trained on large text corpora are fine-tuned on specific tasks. PEFT methods like LoRA make transfer learning efficient for billion-parameter models.",
        "contexts": [
            "Transfer learning is a machine learning technique where a model trained on one task is adapted (fine-tuned) for a different but related task.",
            "BERT demonstrated that pre-training on masked language modeling and next sentence prediction, then fine-tuning on downstream tasks, achieves state-of-the-art NLP results.",
        ],
        "ground_truth": "Transfer learning adapts models pre-trained on large datasets to new tasks, reducing data and compute requirements through fine-tuning.",
    },
    {
        "question": "What are vector databases used for?",
        "answer": "Vector databases store and query high-dimensional vector embeddings using approximate nearest neighbor search. They are used in RAG systems, semantic search engines, recommendation systems, and image search. ChromaDB, Pinecone, and Weaviate are popular options.",
        "contexts": [
            "Vector databases are specialized storage systems optimized for storing, indexing, and querying high-dimensional vector embeddings.",
            "Vector databases are the backbone of RAG systems, semantic search engines, recommendation systems, and image/audio search applications.",
        ],
        "ground_truth": "Vector databases store embeddings and support similarity search, powering RAG systems, semantic search, and recommendation engines.",
    },
    {
        "question": "What is reinforcement learning?",
        "answer": "Reinforcement learning is a paradigm where an agent learns to make decisions by interacting with an environment to maximize cumulative reward. Key algorithms include DQN, PPO, and Actor-Critic methods. RLHF uses RL to align LLMs with human preferences.",
        "contexts": [
            "Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment to maximize cumulative reward.",
            "RLHF (Reinforcement Learning from Human Feedback) is central to aligning LLMs with human preferences.",
        ],
        "ground_truth": "Reinforcement learning trains agents to maximize reward through environment interaction, with RLHF being key to modern LLM alignment.",
    },
]


def load_and_index_documents(file_paths: list[str]) -> int:
    import os as _os

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview",
        google_api_key=settings.GEMINI_API_KEY,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    all_docs = []
    for path in file_paths:
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        all_docs.extend(chunks)

    vectorstore = Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
    vectorstore.add_documents(all_docs)
    logger.info("Indexed %d chunks from %d files", len(all_docs), len(file_paths))
    return len(all_docs)


def load_sample_data() -> int:
    sample_docs_dir = "./data/sample_docs"
    os.makedirs(sample_docs_dir, exist_ok=True)

    file_paths = []
    for name, content in SAMPLE_DOCUMENTS:
        path = os.path.join(sample_docs_dir, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        file_paths.append(path)

    logger.info("Created %d sample documents in %s", len(file_paths), sample_docs_dir)
    return load_and_index_documents(file_paths)


def get_document_count() -> int:
    try:
        import os as _os
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )
        vectorstore = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
        return vectorstore._collection.count()
    except Exception:
        return 0
