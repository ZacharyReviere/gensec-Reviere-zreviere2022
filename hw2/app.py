import os
import logging
import readline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import chainlit as cl

# fau-gensec changes:
from langchain_core.prompts import ChatPromptTemplate

# Configure the chat model used by the Chainlit RAG app.
llm = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL"))

# Open the persisted vector database and create a retriever from it.
vectorstore = Chroma(
     persist_directory="./rag_data/.chromadb",
     embedding_function=GoogleGenerativeAIEmbeddings(
         model="models/gemini-embedding-001",
         google_api_key=os.environ["GOOGLE_API_KEY"],
         vertexai=False
     )
)

retriever = vectorstore.as_retriever()

# prompt = hub.pull("rlm/rag-prompt")
# Use a local prompt so the answer style is clear and concise.
prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""
)

print("RAG prompt:", prompt)


def format_docs(docs):
    """Join retrieved document contents into one prompt context string."""
    # Combine retrieved documents into the context string expected by the prompt.
    return "\n\n".join(doc.page_content for doc in docs)


def load_pdf(file_path: str, source_name: str) -> int:
    """Extract a PDF's text, split it, and add it to the persistent RAG database.

    Args:
        file_path: Local path to the uploaded PDF supplied by Chainlit.
        source_name: Original filename to retain in each chunk's source metadata.

    Returns:
        The number of document chunks added to Chroma.

    Raises:
        ValueError: If the PDF contains no extractable text, such as a scan
            requiring OCR. PDF parsing and embedding errors also propagate.
    """
    documents = PyPDFLoader(file_path).load()
    documents = [doc for doc in documents if doc.page_content.strip()]
    if not documents:
        raise ValueError("The PDF contains no extractable text.")
    for doc in documents:
        doc.metadata["source"] = source_name
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    vectorstore.add_documents(documents=chunks)
    return len(chunks)

# Retrieve context, fill the prompt, call the model, and parse plain text.
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@cl.on_chat_start
async def on_chat_start():
    """Send the initial Chainlit logo and welcome message."""
    logo = cl.Image(name="logo", display="inline", url="https://codelabs.cs.pdx.edu/images/pdx-cs-logo.png")
    await cl.Message(content="", elements=[logo]).send()

    welcome_text = (
        "**Welcome to the PSU Generative Security Chatbot!**\n\n"
        "Ask me anything from my set of documents.\n\n"
        "Attach PDF files to add them to the document collection, then ask a question.\n\n"
    )
    await cl.Message(content=welcome_text).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Index attached PDFs before answering, or acknowledge an upload alone.

    Files are added to the existing shared, persistent document collection.
    Report individual upload failures without preventing other files or a
    text question from being processed.
    """
    for file in message.elements or []:
        if file.mime != "application/pdf" and not file.name.lower().endswith(".pdf"):
            await cl.Message(content=f"Unsupported file: {file.name}. Please attach a PDF.").send()
            continue
        try:
            chunk_count = await cl.make_async(load_pdf)(file.path, file.name)
        except Exception:
            logging.exception("Failed to load PDF %s", file.name)
            await cl.Message(
                content=f"Could not load {file.name}. Check that it is a readable PDF "
                "with selectable text, and try again."
            ).send()
        else:
            await cl.Message(content=f"Loaded {file.name} ({chunk_count} chunks).").send()

    user_query = message.content
    if not user_query.strip():
        return
    answer = rag_chain.invoke(user_query)
    await cl.Message(content=answer).send()

if __name__ == "__main__":
    cl.run()
