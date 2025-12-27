from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

API_KEY = ""
gemini_model = "gemini-2.5-flash-lite"
folder = './sample_rag'
pdf_file = 'The ethylbenzene production.pdf'

data = PyPDFLoader(pdf_file)

llm = ChatGoogleGenerativeAI(
    model=gemini_model,
    google_api_key=API_KEY,
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

doc = data.load()

split_string = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split = split_string.split_documents(doc)

embeding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)

# for i in split:
# vect=Chroma.from_documents(documents=doc,embedding=embeding,persist_directory=folder)


vect = Chroma(persist_directory=folder, embedding_function=embeding)


def ragg():
    retreive = vect.as_retriever()  # search_kwargs={"k":3}

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # print(retreive)

    promt = """consider you are a chemical engineer and reply the questions more accurate to the chemical perspective , if the question is not related to chemical reply 'i dont know '

    Context:{context}
    Question:{question}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(promt)

    rag_chain = ({
                     "context": retreive | format_docs, "question": RunnablePassthrough()}
                 | prompt
                 | llm
                 | StrOutputParser()
                 )
    return rag_chain