import streamlit as st
import re
import random
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from deep_translator import GoogleTranslator
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain


pdf_paths = [
    "Hoteles_Final1.7.pdf",
    "Tequileras_1.2Final.pdf",
    "Oxxo&ATM_Final1.1.pdf",
    "Antros&Tragos&Gas_Final1.2.pdf",
    "Farmacias&BUS_1.2Final.pdf",
    "Motomandados&UBER_PreFinal.pdf",
    "informacion de tequila.pdf",
    # Add more paths as needed
]


def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def detectar_links(texto):
    raw_links = re.findall(r'https?://\S+', texto)
    # Limpiar caracteres basura como ). o ,
    return [re.sub(r'[\)\]\>,\.]+$', '', link) for link in raw_links]


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def summarize_response(response_content, max_tokens=8000):
    if len(response_content) <= max_tokens:
        return response_content
    else:
        return response_content[:max_tokens] + "..."


def get_conversation_chain(vectorstore):
    try:
        if "conversation_chain" not in st.session_state:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo-1106",
                temperature=0.48,
                openai_api_key=st.secrets["openai"]["openai_api_key"]
            )

            memory = ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True,
                max_messages=6,
            )

            # Solo usa el LLM directamente (sin prompt personalizado)
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory,
                verbose=True
            )

            st.session_state.conversation_chain = conversation_chain
        else:
            conversation_chain = st.session_state.conversation_chain

        return conversation_chain

    except KeyError as e:
        st.error(f"Error: {e}. Se está trabajando en eso.")
        return None
    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {e}")
        return None

def translate_text(text, target_lang):
    translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
    return translated

def handle_userinput(user_question, target_language="es"):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Traducir el texto del link según el idioma del usuario
    texto_link_traducido = translate_text("Click here to view", target_language)

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            contenido = message.content
            links = detectar_links(contenido)

            for link in links:
                link_sanitizado = re.sub(r'[\)\]\>,\.]+$', '', link)
                contenido = contenido.replace(link, f"<a href='{link_sanitizado}' target='_blank'>{texto_link_traducido}</a>")

            st.write(bot_template.replace("{{MSG}}", contenido), unsafe_allow_html=True)



def start_chat(translated_text_2):
    # Aquí se puede agregar mensajes iniciales del bot para comenzar la conversación
    initial_message =  translated_text_2
    
    
    # Mostrar el mensaje inicial del bot en el chat
    st.write(bot_template.replace("{{MSG}}", initial_message), unsafe_allow_html=True)


def render_components_nav():
    bootstrap_css = """
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.3/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    """

    custom_css = """
    <style>
    @media (max-width: 768px) {
        .nav-link {
            display: none;
        }
        .img2 {
            display: inline-block;
        }
    }
    </style>
    """

    bootstrap_css_2 = """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    """

    navbar_html = """
    <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background: rgb(255,255,255);background: linear-gradient(0deg, rgba(255,255,255,1) 12%, rgba(73,203,65,1) 40%);">
        <a class="navbar-brand" href="#" target="_blank">
        <img src='https://i.ibb.co/6vNgMsb/output-onlinepngtools-2.png' width='125' /></a>
        <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <a href="https://instagram.com/b2_studioss?igshid=OGQ5ZDc2ODk2ZA=="><img class="img2" src='https://i.ibb.co/288YzSp/B2New.jpg' width='30' /></a>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav">
                <li class="nav-item active">
                    <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="https://youtube.com/dataprofessor" target="_blank">Instagram</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="https://twitter.com/thedataprof" target="_blank">Facebook</a>
                </li>
            </ul>
            <ul class="navbar-nav ml-auto">
                <li class="nav-item">
                    <a class="nav-link" href="" target="_blank"><img class="img2" src='https://i.ibb.co/288YzSp/B2New.jpg' width='50' /></a>
                </li>
            </ul>
        </div>
    </nav>
    """

    hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """

    # Render the HTML and CSS using Markdown
    st.markdown(bootstrap_css, unsafe_allow_html=True)
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown(bootstrap_css_2, unsafe_allow_html=True)
    st.markdown(navbar_html, unsafe_allow_html=True)
    st.markdown(hide_st_style, unsafe_allow_html=True)

def pre_page():
    
    st.set_page_config(page_title="Find Tequila", page_icon="https://i.ibb.co/whtfXgN/IMG-3702-2.jpg")
    st.markdown('<div class="container">', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

# Agregar contenido al primer elemento (texto)
    with col1:
        
        tittle_select = "Selecciona el idioma | Select Language"

        target_language = st.selectbox(f":green[{tittle_select}]", ["en", "es", "fr", "de", "it", "ja", "ko", "zh-CN"])

        if target_language == "es":
            st.markdown("<img src='https://i.ibb.co/rtRT5Z0/Post-de-Instagram-Lista-de-Servicios-Minimalista-Rosa.png' style='width: 100%; max-width: 390px; padding-bottom: 2rem'/>", unsafe_allow_html=True)
        else:
            st.markdown("<img src='https://i.ibb.co/N2TYLGH/Post-de-Instagram-Lista-de-Servicios-Minimalista-Rosa-1.png' style='width: 100%; max-width: 390px; padding-bottom: 2rem'/>", unsafe_allow_html=True)


        # Interfaz de usuario
        if st.button(":blue[Chat PREMIUM]"):
            st.spinner("Cargando...")
            st.warning('No disponible | Not aviable', icon="⚠️")
            
    
        if target_language == "es":
                st.markdown("<img src='https://i.ibb.co/cTKHLF6/Post-de-Instagram-Lista-de-Servicios-Minimalista-Rosa-3.png' style='width: 100%; max-width: 390px; padding-bottom: 2rem'/>", unsafe_allow_html=True)
        else:
                st.markdown("<img src='https://i.ibb.co/rfcyrJV/Post-de-Instagram-Lista-de-Servicios-Minimalista-Rosa-2.png' style='width: 100%; max-width: 390px; padding-bottom: 2rem'/>", unsafe_allow_html=True)

        if st.button(":green[Chat FREE]"):
            st.spinner("Cargando...")
            st.session_state.pagina_actual = "inicio"
            
    
    render_components_nav()

def pagina_inicio_Free():
    
    st.set_page_config(page_title="Find Tequila", page_icon="https://i.ibb.co/whtfXgN/IMG-3702-2.jpg")
    st.markdown('<div class="container">', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

# Agregar contenido al primer elemento (texto)
    with col1:
        text_to_translate = "Pagina Principal"
        additional_text = "Has click en iniciar el chat, si te interesa algun anuncio haz click en el anuncio o preguntale a Agavi"
        tittle_select = "Selecciona el idioma | Select Language"

        target_language = st.selectbox(f":green[{tittle_select}]", ["en", "es", "fr", "de", "it", "ja", "ko", "zh-CN"])


        translated_text_1 = translate_text(text_to_translate, target_language)
        translated_text_2 = translate_text(additional_text, target_language)

        st.write(
            f"<h1 style='text-align: center; gap: 0rem;'>{translated_text_1}</h1>",
            unsafe_allow_html=True
        )

        st.write(
            f"<p style='text-align: center;'>{translated_text_2}</p>",
            unsafe_allow_html=True
        )

        images = ["https://monchitime.com/www/wp-content/uploads/2013/03/CONGELADO.jpg",
                  "https://leisureandlux.mx/wp-content/uploads/2020/08/Jos%C3%A9-Cuervo-Tradicional-Cristalino-1-1024x578.jpg",
                  "https://i.pinimg.com/736x/e1/4c/da/e14cda25ec5058b5b0b11754a10911d8.jpg"]
        
        images2 = ["https://www.dondeir.com/wp-content/uploads/2017/10/Hornitos-y-DO%CC%81NDE-IR.jpg",
                    "https://http2.mlstatic.com/D_NQ_NP_694544-MLU70044830952_062023-O.webp",
                    "https://m.media-amazon.com/images/I/91wKd8MrmUL.jpg"]

    random_image = random.choice(images)
    random_image2 = random.choice(images2)    
    # Agregar contenido al segundo elemento (imagen)
    with col2:
        st.write(
    "<div style='display: flex; justify-content: center; padding-bottom: 2rem;'>"
    f"<img src='{random_image}' style='width: 100%; max-width: 370px;'  />"
    "</div>",
    "<div style='display: flex; justify-content: center; padding-bottom: 2rem;'>"
    f"<a href='https://youtube.com/dataprofessor'><img src='{random_image2}' style='width: 100%; max-width: 370px;' ' /></a>"
    "</div>",
    unsafe_allow_html=True
    )
    if st.button(":green[Iniciar chat | Start Chat]"):
            st.session_state.pagina_actual = "main"
    
    render_components_nav()

def main_Free():
    load_dotenv()
    st.set_page_config(page_title="Find Tequila",
                       page_icon="https://i.ibb.co/whtfXgN/IMG-3702-2.jpg")
    
    
    st.write(css, unsafe_allow_html=True)
    st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

    render_components_nav()
    


    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.markdown('<div class="container">', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        text_to_translate = "Mi nombre es Agavi, ¿Como puedo ayudarte?"
        text_to_translate2 = "Hola, estoy listo para ayudarte.." 
        #additional_text = "Por favor, introduce tus datos para continuar."
        tittle_select = "Selecciona el idioma | Select Language"

        target_language = st.selectbox(f":green[{tittle_select}]", ["en", "es", "fr", "de", "it", "ja", "ko", "zh-CN"])

        translated_text_1 = translate_text(text_to_translate, target_language)
        translated_text_2 = translate_text(text_to_translate2, target_language)
        #translated_text_3 = translate_text(additional_text, target_language)

        st.write(
            f"<h1 style='text-align: center; gap: 0rem;'>{translated_text_1}</h1>",
            unsafe_allow_html=True
        )

    # Agregar contenido al segundo elemento (imagen)
    with col2:
        st.write(
    "<div style='display: flex; justify-content: center; padding-bottom: 2rem;'>"
    "<img src='https://i.ibb.co/TgcShwr/AgaviNew.jpg' width='200' />"
    "</div>",
    unsafe_allow_html=True
    )


    # Process the PDFs from the defined paths
    with st.spinner("Pensando... | Thinking..."):
        raw_text = get_pdf_text(pdf_paths)          # Extrae texto de los PDFs
        text_chunks = get_text_chunks(raw_text)     # Lo divide correctamente
        vectorstore = get_vectorstore(text_chunks)  # Construye índice FAISS
        st.session_state.conversation = get_conversation_chain(vectorstore)
        start_chat(translated_text_2)
        

    # Create a container to hold the chat messages
    chat_container = st.empty()
    

    # Create a container for the user input at the bottom
    input_container = st.empty()


    
    # Place the input field inside the input container
    with input_container:
        prompt = st.chat_input("Type Here.. | Pregunta Aqui... ")
        
        
        
    
    

    # Process user input and update chat
    if prompt:
        handle_userinput(prompt)

    

    # Add spacing at the end to push the input container to the bottom
    st.markdown('<style>div.css-1aumxhk { margin-top: auto; }</style>', unsafe_allow_html=True)

def main_Pay():
    load_dotenv()
    st.set_page_config(page_title="Find Tequila",
                       page_icon="https://i.ibb.co/whtfXgN/IMG-3702-2.jpg")
    
    
    st.write(css, unsafe_allow_html=True)
    st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

    render_components_nav()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.markdown('<div class="container">', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        text_to_translate = "Mi nombre es Agavi, ¿Como puedo ayudarte?"
        additional_text = "Necesito ayuda personalizada | Click Aqui 🧑🏻‍💻"
        tittle_select = "Selecciona el idioma | Select Language"

        target_language = st.selectbox(f":green[{tittle_select}]", ["en", "es", "fr", "de", "it", "ja", "ko", "zh-CN"])

        translated_text_1 = translate_text(text_to_translate, target_language)
        translated_text_2 = translate_text(additional_text, target_language)

        st.write(
            f"<h1 style='text-align: center; gap: 0rem;'>{translated_text_1}</h1>",
            unsafe_allow_html=True
        )

    # Agregar contenido al segundo elemento (imagen)
    with col2:
        st.write(
    "<div style='display: flex; justify-content: center; padding-bottom: 2rem;'>"
    "<img src='https://i.ibb.co/TgcShwr/AgaviNew.jpg' width='200' />"
    "</div>",
    unsafe_allow_html=True
    )
    
    st.write(
    f"<a style='color: green' href='https://wa.me/3741011240'><h4 style='text-align: center;'>{translated_text_2}</h4></a>",
    unsafe_allow_html=True
    )


    # Process the PDFs from the defined paths
    raw_text = get_pdf_text(pdf_paths)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    st.session_state.conversation = get_conversation_chain(vectorstore)

    # Create a container to hold the chat messages
    chat_container = st.empty()

    # Show chat messages from chat history if available
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                chat_container.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                summarized_response = summarize_response(message.content)
                chat_container.write(bot_template.replace("{{MSG}}", summarized_response), unsafe_allow_html=True)

    # Show chat messages of the test chat(s)
    

    # Create a container for the user input at the bottom
    input_container = st.empty()


    
    # Place the input field inside the input container
    with input_container:
        prompt = st.chat_input("Type Here.. | Pregunta Aqui... ")
        
        
        
    
    

    # Process user input and update chat
    if prompt:
        handle_userinput(prompt)
    #Aqui vamos a agregar una funcion que salgan anuncios


    # Add spacing at the end to push the input container to the bottom
    st.markdown('<style>div.css-1aumxhk { margin-top: auto; }</style>', unsafe_allow_html=True)


if __name__ == '__main__':
    
    if "pagina_actual" not in st.session_state:
        st.session_state.pagina_actual = "prepage"

    if st.session_state.pagina_actual == "prepage":
        #pre_page()
        pagina_inicio_Free()
    elif st.session_state.pagina_actual == "inicio":
        pagina_inicio_Free()
    else:
        main_Free()
