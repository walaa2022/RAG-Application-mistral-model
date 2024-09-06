from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import os

# Set the environment variable to allow dangerous deserialization
"""
By setting ALLOW_DANGEROUS_DESERIALIZATION to "true", you are effectively allowing your application to 
process serialized data without some of the usual safety checks. This can be useful for debugging or 
specific scenarios where you trust the source of the data, but it can also expose your application to security risks if not managed carefully.
"""
os.environ['ALLOW_DANGEROUS_DESERIALIZATION'] = "true"

#Getting the vector database location in the application.
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Setting the custom prompt which has 2 variables as its dynamic content ['context', 'question']
# Context: is the top similar context we got from the vector database.
# Question: is the original user que
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    custom_prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """

    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )

    return prompt


# Retrieval QA Chain
# The default chain_type="stuff" uses ALL of the text from the documents in the prompt.
# map_reduce: It separates texts into batches (as an example, you can define batch size in llm=OpenAI(batch_size=5)), feeds each batch with the question to LLM separately, and comes up with the final answer based on the answers from each batch.
# refine : It separates texts into batches, feeds the first batch to LLM, and feeds the answer and the second batch to LLM. It refines the answer by going through all the batches.
# map-rerank: It separates texts into batches, feeds each batch to LLM, returns a score of how fully it answers the question, and comes up with the final answer based on the high-scored answers from each batch.
def retrieval_qa_chain(llm, prompt, db):
    """
    llm=llm: Specifies the language model to use.
    chain_type='stuff': Sets the chain type to 'stuff', meaning all retrieved documents will be used in a single prompt.
    retriever=db.as_retriever(search_kwargs={'k': 2}): Configures the retriever to fetch documents from the database. search_kwargs={'k': 2} indicates that the retriever will fetch the top 2 documents.
    eturn_source_documents=True: Indicates that the source documents should be returned along with the answers. This can be useful for understanding the context of the generated answers.
    chain_type_kwargs={'prompt': prompt}: Passes additional arguments to the chain type, such as the prompt or question to be used.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    return qa_chain


# Loading the model
# If local = True, we use locally downloaded LLM. (Free)
# If local = False, we use OpenAI's GPT models using an API call. (Paid)
def load_llm(local: bool):
    
    if local:
        # Load the locally downloaded model here
        llm = CTransformers(
            # model="TheBloke/Llama-2-7B-Chat-GGML",
            model="capybarahermes-2.5-mistral-7b.Q3_K_M.gguf",
            model_type="mistral",
            max_new_tokens = 1024,
            temperature = 0.5
        )
        
        return llm

    else:
        # This is using a paid LLM like GPT4o from OpenAI
        return None 

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm(local=True)
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})

    return response

# chainlit code
# This decorator registers an asynchronous function start() to be called when a chat session starts.
@cl.on_chat_start
async def start():
    chain = qa_bot() # Initializes a chatbot chain we have created above.
    msg = cl.Message(content="Starting the bot...") # Creates a message to inform the user that the bot is starting.
    await msg.send() #Sends the initial message to the user.
    msg.content = "Hi, Welcome to the bot. What is your question?" # Updates the content of the initial message to provide a greeting and prompt.
    await msg.update() # Updates the message on the frontend to reflect the new content.

    cl.user_session.set("chain", chain) #Saves the chain object in the user session so that it can be accessed later.


#This decorator registers an asynchronous function main() to be called when a message is received from the user.
@cl.on_message
async def main(message: cl.Message):
    try:
        chain = cl.user_session.get("chain")  # Retrieves the chatbot chain from the user session
        """
        cl.AsyncLangchainCallbackHandler is a class from the Chainlit library used to handle callbacks asynchronously during the interaction with a language model or chain.
        It provides mechanisms to manage and process intermediate results and final answers from language models.
        
        Parameters:
        1) stream_final_answer=True: This parameter indicates that the final answer should be streamed back in real-time. 
        It allows the handler to send updates as the answer is being generated, rather than waiting until the entire answer is ready.

        2) answer_prefix_tokens=["FINAL", "ANSWER"]: This parameter specifies tokens that are used to determine when the final answer has been reached in the streaming process. 
        The handler will use these tokens as indicators to finalize the answer and process any subsequent actions.
        """
        ###################################################################################
        # You can visualize the intermediate steps by clicking on the dropdown for any step
        ###################################################################################
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True,
            answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True

        res = await chain.ainvoke(message.content, callbacks=[cb])  # Invokes the chatbot chain with the user's message
        answer = res["result"]  # Extracts the result from the response
        sources = res["source_documents"]  # Extracts source documents from the response

        if sources:
            answer += f"\nSources:" + str(sources)  # Appends sources to the answer if available
        else:
            answer += "\nNo sources found"  # Adds a note if no sources are found

        await cl.Message(content=answer).send()  # Sends the response back to the user
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()  # Sends an error message if an exception occurs