# Mistral Bot

The Mistral Bot is a tool designed to provide information by answering user queries using state-of-the-art language models and vector stores. This README will guide you through the setup and usage of the Mistral Bot.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [License](#license)

## Prerequisites

Before you can start using the Mistral Bot, make sure you have the following prerequisites installed on your system:

- Python 3.6 or higher
- Required Python packages (you can install them using pip): You can install everything in the requirements.txt using
pip install -r requirements.txt
  - langchain
  - chainlit
  - sentence-transformers
  - faiss
  - PyPDF2 (for PDF document loading)

## Installation

1. Clone this repository to your local machine.

    ```bash
    git clone https://github.com/muhammedkhaled95/langchain-Mistral-RAG-App.git
    cd langchain-Mistral-RAG-App
    ```

2. Create a Python virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

3. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Download the required [language model](https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF) and data. Please refer to the Langchain documentation for specific instructions on how to download and set up the language model and vector store.

5. Set up the necessary paths and configurations in your project, including the `DB_FAISS_PATH` variable and other configurations as per your needs.

## Getting Started

To get started with the Mistral Bot, you need to:

1. Set up your environment and install the required packages as described in the Installation section.

2. Configure your project by updating the `DB_FAISS_PATH` variable and any other custom configurations in the code.

3. Prepare the language model and data as per the Langchain documentation.

4. Start the bot by running the provided Python script or integrating it into your application.

## Usage

The Mistral Bot can be used for answering your data related queries. To use the bot, you can follow these steps:

1. Start the bot by running your application or using the provided Python script.

2. Send a query to the bot.

3. The bot will provide a response based on the information available in its database.

4. If sources are found, they will be provided alongside the answer.

5. The bot can be customized to return specific information based on the query and context provided.

## Links

- [CapybaraHermes-2.5-Mistral-7B-GGUF](https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF)
- [chainlit](https://github.com/Chainlit/chainlit)
- [faiss](https://github.com/facebookresearch/faiss)
- [Chroma](https://www.trychroma.com)
- [Qdrant](https://qdrant.tech)
- [LangChain](https://python.langchain.com/v0.1/docs/get_started/introduction.html)
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [ctransformers](https://github.com/marella/ctransformers)
- [vLLM](https://github.com/vllm-project/vllm)

## Acknowledgments

This project is based on [llama2 Medical Bot](https://github.com/alansary/langchain-medical-bot) by [https://github.com/alansary]. The original project is licensed under the MIT License.

Modifications have been made to fit the needs of this project.

## License

This project is licensed under the MIT License.

---

Happy coding with Mistral Bot! 🚀
