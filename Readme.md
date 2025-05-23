# LLM-RAG Pipeline and AI Agent Evaluation

This project implements multiple **Retrieval-Augmented Generation (RAG)** pipelines and creates various AI agents using **LangChain**. The agents are evaluated using the **Ragas** library to measure their performance on tasks such as context precision, faithfulness, and answer relevancy.

The project supports multiple LLMs, including **OpenAI's GPT**, **Google's Gemini models**, and **custom RAG pipelines**. It also includes a Flask server for RAG evaluation.

---

## Features

- **RAG Pipelines**: Implements multiple RAG pipelines for document retrieval and question answering.
- **AI Agents**: Creates AI agents using LangChain for different LLMs (e.g., GPT, Gemini).
- **Evaluation with Ragas**: Evaluates the performance of AI agents using Ragas metrics such as:
  - Context Precision
  - Faithfulness
  - Answer Relevancy
  - Context Recall
  - Answer Correctness
  - Answer Similarity
- **Flask Server**: A Flask-based server for running RAG evaluations.
- **Input/Output Management**: Handles input datasets, JSON files for questions and ground truths, and outputs the results in JSON format.

---

## Requirements

The project requires the following environment variables to be set in a `.env` file:

```plaintext
OPENAI_API_KEY=<Your OpenAI API Key>
GOOGLE_API_KEY=<Your GEMINI API Key>
GEMINI_API_KEY=<Your Gemini API Key>
```

---

## Folder Structure

The project is organized into the following folders:

### 1. **`flask_server`**
   - Contains Flask code for running RAG evaluation as a web service.
   - Allows users to send requests for evaluating RAG pipelines and receive results.

### 2. **`gemini`**
   - Contains code for implementing RAG pipelines and AI agents using **Google's Gemini models**.
   - Uses the `langchain-google-genai` package for integration with Gemini.

### 3. **`gpt`**
   - Contains code for implementing RAG pipelines and AI agents using **OpenAI's GPT models**.
   - Uses the `langchain` package for integration with OpenAI.

### 4. **`input/dataset`**
   - Contains datasets (e.g., PDFs, text files) that are fed into the LLM models or agents for processing.

### 5. **`input/json`**
   - Contains JSON files with:
     - **Questions**: Questions to be answered by the LLM agents.
     - **Ground Truth**: Expected answers for evaluation using Ragas.

### 6. **`output`**
   - Contains JSON files generated by the LLM agents.
   - Each output file includes:
     - Questions
     - Contexts retrieved by the RAG pipeline
     - Answers generated by the LLM
     - Ground truth for evaluation

### 7. **`others`**
   - Contains older setups and experiments created while learning LangChain and Ragas.
   - May include deprecated or experimental code.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory and add the following keys:

```plaintext
OPENAI_API_KEY=<Your OpenAI API Key>
GOOGLE_API_KEY=<Your Google API Key>
GEMINI_API_KEY=<Your Gemini API Key>
```

### 5. Run the Flask Server

Navigate to the `flask_server` directory and start the Flask server:

```bash
cd flask_server
python app.py
```

The server will be available at `http://127.0.0.1:5000`.

---

## Usage

### 1. Running RAG Pipelines

- **Gemini**: Run the RAG pipeline using Gemini models by executing the code in the `gemini` folder.
- **GPT**: Run the RAG pipeline using OpenAI's GPT models by executing the code in the `gpt` folder.

### 2. Evaluating AI Agents

- Use the `input/json` folder to provide questions and ground truth for evaluation.
- The output will be saved in the `output` folder as a JSON file.

### 3. Flask Server

- Use the Flask server to evaluate RAG pipelines via API requests.
- Send a POST request with the required input data to the server endpoint.

---

## Example Workflow

1. Place your dataset (e.g., `resume.pdf`) in the `input/dataset` folder.
2. Add a JSON file with questions and ground truth in the `input/json` folder.
3. Run the RAG pipeline (e.g., `gemini` or `gpt`).
4. The output JSON file will be saved in the `output` folder.
5. Evaluate the output using Ragas metrics.

---

## Evaluation Metrics

The project uses the **Ragas** library to evaluate the performance of AI agents. The following metrics are calculated:

- **Context Precision**: Measures how precise the retrieved context is.
- **Faithfulness**: Measures how faithful the generated answer is to the retrieved context.
- **Answer Relevancy**: Measures how relevant the answer is to the question.
- **Context Recall**: Measures how much of the relevant context is retrieved.
- **Answer Correctness**: Measures how correct the generated answer is.
- **Answer Similarity**: Measures the similarity between the generated answer and the ground truth.

---

## Dependencies

The project uses the following Python libraries:

- `langchain`
- `langchain-google-genai`
- `langchain-community`
- `ragas`
- `flask`
- `dotenv`
- `plotly`

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Contributing

Contributions are welcome! If you have ideas for improving the project or adding new features, feel free to open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For any questions or issues, please contact:

- **Name**: Paras
- **Email**: dhimanparas20@gmail.com
