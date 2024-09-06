#langchain_community.embeddings is deprecated, use langchain_huggingface instead
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

# These lines define two path constants. DATA_PATH is where the PDF files are stored, 
# and DB_FAISS_PATH is where the FAISS vector database will be saved.
DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

"""
Create the vector database
The process of creating a vector database for the external data source (documents and such) is as follows:
1- Create the document loader object of type (DirectoryLoader)
2- Load the documents using the document loader object and save them to variable (documents)
3- Create a text splitter object of type (RecursiveCharacterTextSplitter) 
4- Split the documents into text chunks using the text splitter object and save them to variable (texts).
5- Create the embeddings model object that will be used for converting the text chunks into vector representation
6- Create the vector store (vector database) using the text chunks and the embedding model passed to FAISS.
"""
def create_vector_db():
    # First, we need to define a documents loader object which will be used
    # to load all the documents that will be saved in the vector database.
    loader = DirectoryLoader(
        # DATA_PATH: is the location where all the documents are.
        # glob: is basically telling the loader that any document in the DATA_PATH
        # and end with .pdf will be included in the document loading operation.
        # loader_cls: specifies which class should be used to load individual files within the directory
        DATA_PATH,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    
    #Now, we use the loader object we've created to load the documents.
    documents = loader.load()
    
    # This creates a RecursiveCharacterTextSplitter object. It will be used to split the text into chunks of 500 characters, 
    # with an overlap of 50 characters between chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Using text_splitter object we've created above to split the documents using split_documents() function and passing the documents to it.
    texts = text_splitter.split_documents(documents)
    
    # This creates a HuggingFaceEmbeddings object, which will be used to generate embeddings. 
    # It's using the 'sentence-transformers/all-MiniLM-L6-v2' model and specifying to run on CPU.
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    
    """
    1. `FAISS`: 
    FAISS (Facebook AI Similarity Search) is a library developed by Facebook for efficient similarity search and clustering 
    of dense vectors. In this context, it's being used to create a vector store.

    2. `from_documents()`: 
    This is a class method of FAISS that creates a new FAISS index from a list of documents and an embedding function.

    3. `texts`: 
    This is the first argument to `from_documents()`. It's the list of text chunks that were created earlier by 
    splitting the original documents. Each of these chunks will be converted into a vector representation and stored in the FAISS index.

    4. `embeddings`: 
    This is the second argument to `from_documents()`. It's the HuggingFaceEmbeddings object we created earlier. 
    This object knows how to convert text into vector representations (embeddings) using the specified model 
    ('sentence-transformers/all-MiniLM-L6-v2' in this case).

    What happens behind the scenes:

    1. For each text chunk in `texts`, the `embeddings` object is used to generate a vector representation.
    2. These vector representations are then added to a new FAISS index.
    3. The FAISS index allows for efficient similarity search later on. When you want to find documents similar to a query, 
    you can convert the query to a vector using the same embedding model, then use FAISS to quickly find the most similar vectors in the index.

    The result `db` is a FAISS vector store that contains all the vector representations of your text chunks, 
    organized in a way that allows for fast similarity search.
    """
    db = FAISS.from_documents(texts, embeddings)
    
    # Saving the vector store locally in the path provided (DB_FAISS_PATH)
    db.save_local(DB_FAISS_PATH)
    
    # Executing create_vector_db() in case this script is run directly. We need to do this once, no need 
    # to run this process for the same documents more than once.
if __name__ == "__main__":
    create_vector_db()